import torch

import numpy as np
import datasets
import os
import wandb
import hydra
import omegaconf
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    get_scheduler,
)
from transformers.utils import logging

from models.convert import llama_to_cwic
from utils.data_utils import PackedCollator
from utils.loss_utils import (
    KDLossModule,
    MSELossModule,
    FlopLossModule,
    get_total_active
)
from utils.torch_utils import grad_nan_to_num
from models.modelling_cwic import CWICForCausalLM
from utils.misc_utils import str_to_int_list


logger = logging.get_logger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayerHook:
    def __init__(self):
        self.last = None

    def __call__(self, norm, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        self.last = out

    def get(self):
        x = self.last
        self.last = None
        return x



@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):
    logger.info(f"Starting CWIC distillation training on device {str(DEVICE)}")

    # Load the teacher model
    teacher_model = LlamaForCausalLM.from_pretrained(
        config.teacher_model,
        device_map=DEVICE,
    )
    teacher_model.eval()
    logger.info(f"Loaded teacher model {config.teacher_model}!")

    # Initialize the student model from the teacher
    if config.student_model is not None:
        student_model = CWICForCausalLM.from_pretrained(
            config.student_model,
            device_map=DEVICE,
        )
        logger.info(f"Loaded student model {config.student_model}!")
    else:
        student_model = llama_to_cwic(
            teacher_model,
            **config.model,
            mse_layers=config.mse_layers,
        )
        logger.info(f"Initialized student model from teacher!")
    # we keep the teacher model in its original format (important for some models)
    # but we want the student in float32
    student_model = student_model.to(torch.float32)
    student_model.train()
    student_model.gradient_checkpointing_enable()
    logger.info("Student model is ready for training!")

    # add the hooks to capture hidden states
    mse_layers = str_to_int_list(config.mse_layers)
    teacher_hooks = {i: LayerHook() for i in mse_layers}
    student_hooks = {i: LayerHook() for i in mse_layers}
    for i in mse_layers:
        teacher_model.model.layers[i].register_forward_hook(teacher_hooks[i])
        student_model.model.layers[i].register_forward_hook(student_hooks[i])

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.dataset.path)

    # load the dataset
    total_batch_size = config.batch_size * config.grad_accum_steps
    dataset = datasets.load_dataset(**config.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=total_batch_size,
        shuffle=False,
        collate_fn=PackedCollator(tokenizer, config.max_length, DEVICE),
    )
    logger.info(f"Loaded dataset {config.dataset.path} with batch_size {config.batch_size} and max_length {config.max_length}!")

    # load the optimizer
    training_params = list(student_model.parameters())
    training_params.remove(student_model.model.embed_tokens.weight)
    training_params.remove(student_model.lm_head.weight)
    optimizer = torch.optim.AdamW(
        training_params,
        **config.optimizer
    )
    lr_scheduler = get_scheduler(optimizer=optimizer, **config.lr_scheduler)
    logger.info(
        f"Initialized AdamW optimizer and {config.lr_scheduler.name} learning rate scheduler!"
    )

    # load the loss functions
    kd_loss_fn = KDLossModule(config.kd_chunk_size)
    scaled_mse_fn = MSELossModule()
    flop_loss_fn = FlopLossModule()

    # compile if requested
    old_student_model_model = student_model.model
    if config.compile:
        teacher_model.model = torch.compile(teacher_model.model, fullgraph=True)
        student_model.model = torch.compile(student_model.model, fullgraph=False)
        kd_loss_fn = torch.compile(kd_loss_fn, fullgraph=True)
        scaled_mse_fn = torch.compile(scaled_mse_fn, fullgraph=True)
        flop_loss_fn = torch.compile(flop_loss_fn, fullgraph=True)
        logger.info("Added compilation to the models and loss functions!")

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.run_name,
        config=omegaconf.OmegaConf.to_container(config, resolve=True),
    )
    logger.info("Starting Training!")

    pbar = tqdm(desc="CWIC Distillation")
    step = 0
    seen_tokens = 0
    prev_ratio = None

    for total_batch in dataloader:
        
        compute_gain = config.end_compute_reduction - config.start_compute_reduction
        target_ratio = config.start_compute_reduction + compute_gain * np.clip(
            step / config.compute_reduction_steps, a_min=0.0, a_max=1.0
        )

        mini_batches = [{} for _ in range(config.grad_accum_steps)]
        for k, v in total_batch.items():
            for b, v_mini in enumerate(torch.chunk(v, config.grad_accum_steps, dim=0)):
                mini_batches[b][k] = v_mini

        total_aux = {}
        total_active = 0.0
        total_dense = 0.0
        for batch in mini_batches:
            
            mask = (batch["input_ids"] != tokenizer.pad_token_id).float()
            seen_tokens += mask.sum().item()

            with torch.autocast(device_type=str(DEVICE), dtype=torch.bfloat16):
                with torch.no_grad():
                    teacher_output = teacher_model(
                        input_ids=batch["input_ids"],
                        use_cache=False,
                        attention_mask=batch["attention_mask"],
                    )

                student_model.prepare_tracking()
                student_output = student_model(
                    input_ids=batch["input_ids"],
                    statistics_mask=mask,
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )

                kl_loss, rkl, fkl = kd_loss_fn(
                    student_output.logits,
                    teacher_output.logits,
                    mask=mask
                )
                flop_loss = flop_loss_fn(
                    student_output.active_parameters,
                    student_output.dense_parameters,
                    target_ratio=target_ratio,
                    mask=mask,
                    base_ratio=prev_ratio,
                )

                mse_loss = 0.0
                for l in mse_layers:
                    
                    teacher_states = teacher_hooks[l].get()
                    student_states = student_hooks[l].get()

                    mse_loss = mse_loss + scaled_mse_fn(
                        student_states,
                        teacher_states,
                        student_model.cross_projections[str(l)],
                        eps=student_model.config.rms_norm_eps,
                        scale=config.mse_weight,
                        mask=mask
                    )
                mse_loss = mse_loss / len(mse_layers)

            active, dense = get_total_active(
                student_output.active_parameters,
                student_output.dense_parameters,
                mask=mask,
            )
            total_active = total_active + active.detach()
            total_dense = total_dense + dense.detach()

            loss = (
                kl_loss +
                config.flop_weight * flop_loss +
                mse_loss
            )

            aux = {
                "loss": loss,
                "kl_loss": kl_loss,
                "rkl": rkl,
                "fkl": fkl,
                "flop_loss": flop_loss,
                "mse_loss": mse_loss,
            }
            for k, v in aux.items():
                if k not in total_aux.keys():
                    total_aux[k] = v
                else:
                    total_aux[k] = total_aux[k] + v

            loss.backward()

        prev_ratio = (total_dense / total_active).detach().view(1)

        total_aux = {k: v.item() / config.grad_accum_steps for k, v in total_aux.items()}
        total_aux["flop_reduction"] = prev_ratio.item()
        total_aux["target_flop_reduction"] = target_ratio
        total_aux["lr"] = lr_scheduler.get_last_lr()[0]
        total_aux["seen_tokens"] = seen_tokens

        grad_nan_to_num(student_model)
        total_aux["grad_norm"] = torch.nn.utils.clip_grad_norm_(
            student_model.parameters(), config.max_grad_norm
        ).item()

        optimizer.step()
        optimizer.zero_grad(True)
        lr_scheduler.step()

        student_model.clip_thresholds()

        pbar.update(1)
        pbar.set_postfix(
            {
                "kl_loss": total_aux["kl_loss"],
                "FRR": total_aux["flop_reduction"],
                "FRR_targ": total_aux["target_flop_reduction"],
            }
        )

        wandb.log(total_aux)

        step += 1

        if step % config.checkpoint_interval == 0:
            with torch.no_grad():
                logger.info(f"Saving checkpoint at step {step}...")

                ckpt_path = os.path.join("checkpoints", config.run_name, f"{step:08}_{target_ratio:.2f}x".replace(".", "p"))

                tmp_model = student_model.model
                student_model.model = old_student_model_model

                student_model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

                student_model.model = tmp_model

                logger.info(f"Checkpoint saved to {ckpt_path}!")


if __name__ == "__main__":
    main()
