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
from utils.data_utils import DeviceCollator
from utils.loss_utils import (
    kd_loss_fn,
    flop_loss_fn,
)


logger = logging.get_logger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):
    logger.info(f"Starting CWIC distillation training on device {str(DEVICE)}")

    # Load the dataset
    dataset = datasets.load_dataset(**config.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=DeviceCollator(DEVICE),
    )
    logger.info(f"Loaded dataset {config.dataset.path} with batch size {config.batch_size}!")

    # Load the teacher model
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)
    teacher_model = LlamaForCausalLM.from_pretrained(
        config.teacher_model,
        device_map=DEVICE,
    )
    teacher_model.eval()
    logger.info(f"Loaded teacher model {config.teacher_model}!")

    # convert the teacher model to CWIC format
    student_model = llama_to_cwic(teacher_model, **config.model)
    # we keep the teacher model in its original format (important for some models)
    # but CWIC works best in float32
    student_model = student_model.to(torch.float32)
    student_model.train()
    student_model.gradient_checkpointing_enable()
    logger.info("Converted teacher model to CWIC format!")

    optimizer = torch.optim.AdamW(student_model.parameters(), **config.optimizer)
    lr_scheduler = get_scheduler(optimizer=optimizer, **config.lr_scheduler)
    logger.info(
        f"Initialized AdamW optimizer and {config.lr_scheduler.name} learning rate scheduler!"
    )

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.run_name,
        config=omegaconf.OmegaConf.to_container(config, resolve=True),
    )
    logger.info("Starting Training!")

    pbar = tqdm(desc="CWIC Distillation")
    step = 0
    seen_tokens_combined = 0

    for batch in dataloader:
        mask = batch["pad_mask.npy"].float()
        seen_tokens_combined+=mask.sum().item()

        with torch.no_grad():
            teacher_output = teacher_model(
                input_ids=batch["input_ids.npy"],
            )
        student_output = student_model(
            input_ids=batch["input_ids.npy"],
            statistics_mask=mask,
        )

        compute_gain = config.end_compute_reduction - config.start_compute_reduction
        target_ratio = config.start_compute_reduction + compute_gain * np.clip(
            step / config.compute_reduction_steps, a_min=0.0, a_max=1.0
        )

        kd_loss = kd_loss_fn(
            student_output.logits, teacher_output.logits.to(student_output.logits.dtype), mask=mask
        )
        flop_loss, flop_reduction = flop_loss_fn(
            student_output.active_parameters,
            student_output.dense_parameters,
            target_ratio=target_ratio,
            mask=mask,
        )

        loss = kd_loss + flop_loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(True)
        lr_scheduler.step()

        student_model.clip_thresholds()

        pbar.update(1)
        pbar.set_postfix(
            {
                "kd_loss": kd_loss.item(),
                "FRR": flop_reduction.item(),
                "FRR_targ": target_ratio,
            }
        )

        wandb.log(
            {
                "kd_loss/combined": loss.item(),
                "kd_loss/logits/twoway": kd_loss.item(),
                "flop_loss/combined": flop_loss.item(),
                "flop_ratios/combined": flop_reduction.item(),
                "target_flop_reduction": target_ratio,
                "seen_tokens/combined": seen_tokens_combined
            },
        )

        step += 1

        if step % config.checkpoint_interval == 0:
            with torch.no_grad():
                logger.info(f"Saving checkpoint at step {step}...")

                ckpt_path = os.path.join("checkpoints", config.run_name, f"{step:08}.pt")

                student_model.save_pretrained(ckpt_path)
                teacher_tokenizer.save_pretrained(ckpt_path)

                logger.info(f"Checkpoint saved to {ckpt_path}!")


if __name__ == "__main__":
    main()
