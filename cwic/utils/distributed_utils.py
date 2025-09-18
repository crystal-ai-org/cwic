import torch
from torch.distributed import fsdp

import os


def shard_llm(model, mesh):

    for layer in model.model.layers:
        fsdp.fully_shard(layer, mesh=mesh)

    fsdp.fully_shard(model.model.embed_tokens, mesh=mesh)
    fsdp.fully_shard(model.lm_head, mesh=mesh)

    fsdp.fully_shard(model, mesh=mesh)