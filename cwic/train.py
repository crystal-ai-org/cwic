import torch

import datasets
import argparse
import wandb
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


DATASET = 'aklein4/chat-compilation-benchmark-5x-Llama-3.2-Instruct-Shuffled'


def main(args):
    
    # Load the dataset
    dataset = datasets.load_dataset(
        DATASET,
        split='train',
        streaming=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DeviceCollator(),
    )
    logger.info(f"Loaded dataset {DATASET} with batch size {args.batch_size}!")

    # Load the teacher model
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    teacher_model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
    )
    teacher_model.eval()
    logger.info(f"Loaded teacher model {args.model_name}!")

    # convert the teacher model to CWIC format
    student_model = llama_to_cwic(teacher_model)
    # we keep the teacher model in its original format (important for some models)
    # but CWIC works best in float32
    student_model = student_model.to(torch.float32)
    student_model.train()
    student_model.gradient_checkpointing_enable()
    logger.info("Converted teacher model to CWIC format!")

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=1e-5,
    )
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=1000
    )
    logger.info("Initialized optimizer and learning rate scheduler!")

    # wandb.init(
    #     project="CWIC-torch",
    #     config={
    #         "dataset": DATASET,
    #         "batch_size": args.batch_size,
    #         "model_name": args.model_name,
    #         "learning_rate": 1e-5,
    #         "num_warmup_steps": 500,
    #         "num_training_steps": 1000,
    #     }
    # )

    logger.info("Starting training...")
    for batch in dataloader:
        mask = batch['pad_mask.npy'].float()

        with torch.no_grad():
            teacher_output = teacher_model(
                input_ids=batch['input_ids.npy'],
            )
        student_output = student_model(
            input_ids=batch['input_ids.npy'],
            statistics_mask=mask,
        )

        kd_loss = kd_loss_fn(
            student_output.logits,
            teacher_output.logits.to(student_output.logits.dtype),
            mask=mask
        )
        flop_loss = flop_loss_fn(
            student_output.active_params,
            student_output.dense_params,
            target_ratio=2.0,
            mask=mask
        )

        loss = kd_loss + flop_loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(True)
        lr_scheduler.step()

        print(f"KD Loss: {kd_loss.item()}, FLOP Loss: {flop_loss.item()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CWIC the training script.")

    parser.add_argument('--dataset', type=str, default=DATASET, help='Dataset to use for training.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for training.')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct', help='Model name to use for training.')

    args = parser.parse_args()

    main(args)
