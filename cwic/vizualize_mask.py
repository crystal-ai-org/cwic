import torch

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.modelling_cwic import CWICForCausalLM
from transformers import AutoTokenizer

CHECKPOINT = "/home/ubuntu/cwic/checkpoints/llama-1B_1B-tokens/00003000"
MODULE = "model.layers.7.self_attn.o_proj" # "model.layers.7.self_attn.qkv_proj"

PROMPT = """
Grape juice is obtained from crushing and blending grapes into a liquid. In the wine industry, grape juice that contains 7–23 percent of pulp, skins, stems and seeds is often referred to as must. The sugars in grape juice allow it to be used as a sweetener, and fermented and made into wine, brandy, or vinegar.

In North America, the most common grape juice is purple and made from Concord grapes while white grape juice is commonly made from Niagara grapes, both of which are varieties of native American grapes, a different species from European wine grapes. In California, Sultana (known there as 'Thompson Seedless') grapes are sometimes diverted from the raisin or table market to produce white juice.
"""


@torch.no_grad()
def main():
    
    model = CWICForCausalLM.from_pretrained(CHECKPOINT, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    handle = model.get_mask_handle(MODULE)

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    outputs = model(input_ids=input_ids)

    mask = handle.data.numpy()[0]

    sequence_mask = mask.reshape(mask.shape[0], -1)
    img = Image.fromarray((sequence_mask * 255).astype(np.uint8))
    img.save("sequence_mask.png")

    avg_mask = mask.mean(axis=0).reshape(-1, mask.shape[-1])
    img = Image.fromarray((avg_mask * 255).astype(np.uint8))
    img.save("avg_mask.png")


if __name__ == "__main__":
    main()