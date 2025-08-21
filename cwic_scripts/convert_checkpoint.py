
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

from transformers import AutoTokenizer

from cwic_huggingface.configuration_cwic import CWICConfig
from cwic_huggingface.modelling_cwic import CWICForCausalLM, CWICDecoderLayer
from cwic_huggingface.modules_cwic import CWICLinear, CWICMLP

CHECKPOINT = "./checkpoints/release_fr_5"
CONFIG = "./configs/llama_3-2_1B_Instruct.json"
SAVE_PATH = "./torch_checkpoints/release_fr_5"


def load(param, x):
    assert param.data.shape == x.shape, f"Shape mismatch: {param.data.shape} vs {x.shape}"
    param.data = x.clone().detach().contiguous()


def extract_keys(
    checkpoint,
    name
):
    c = {}
    for k in checkpoint.keys():

        if k.startswith(name):
            c[k[len(name) + 1:]] = checkpoint[k]

    return c


def load_linear(
    module: CWICLinear,
    checkpoint: dict,
    name: str,
    layer: int
):
    
    c = extract_keys(checkpoint, name)
    if layer is None:
        c = {k: (v[None] if isinstance(v, torch.Tensor) else v) for k, v in c.items()}
        layer = 0

    load(
        module.weight,
        c["W.kernel"][layer]
    )
    if module.bias is not None:
        load(
            module.bias,
            c["bias"][layer]
        )

    div = 1 - c["dist_tracker.beta"][layer].item() ** c["dist_tracker.steps"][layer].item()
    med = c["dist_tracker.med"][layer] / (div + 1e-7)
    upp = c["dist_tracker.upp"][layer] / (div + 1e-7)
    
    load(
        module.dist_tracker.adj_med_computed,
        med
    )
    load(
        module.dist_tracker.adj_std_computed,
        (upp - med) + 1e-7
    )

    load(
        module.thresholds,
        (c["thresholds"][layer] * c["scalar_scaler"][layer].item()).T
    )


def load_mlp(
    module: CWICMLP,
    checkpoint: dict,
    name: str,
    layer: int
):
    
    c = extract_keys(checkpoint, name)

    load_linear(
        module.gate,
        c,
        "gate",
        layer
    )

    load(
        module.up.weight,
        c["up.kernel"][layer]
    )
    # if module.up.bias is not None:
    #     load(
    #         module.up.bias,
    #         c["up.bias"][layer]
    #     )

    load(
        module.down.weight,
        c["down.kernel"][layer].T
    )
    if module.down.bias is not None:
        load(
            module.down.bias,
            c["down.bias"][layer]
        )
    
    div = 1 - c["dist_tracker.beta"][layer].item() ** c["dist_tracker.steps"][layer].item()
    med = c["dist_tracker.med"][layer] / (div + 1e-7)
    upp = c["dist_tracker.upp"][layer] / (div + 1e-7)  
    # todo runnings
    load(
        module.dist_tracker.adj_med_computed,
        med
    )
    load(
        module.dist_tracker.adj_std_computed,
        (upp - med) + 1e-7
    )

    div = 1 - c["mad_tracker.beta"][layer].item() ** c["mad_tracker.steps"][layer].item()
    med = c["mad_tracker.med"][layer] / (div + 1e-7)
    upp = c["mad_tracker.upp"][layer] / (div + 1e-7)  

    # todo runnings

    load(
        module.mad_tracker.adj_med_computed,
        med
    )

    load(
        module.mad_tracker.adj_std_computed,
        (upp - med) + 1e-7
    )

    load(
        module.thresholds,
        c["thresholds"][layer] * c["scalar_scaler"][layer].item() 
    )


def load_layer(
    module: CWICDecoderLayer,
    checkpoint: dict,
    name: str,
    layer: int
):
    
    c = extract_keys(checkpoint, name)

    load(
        module.input_layernorm.weight,
        c["attention_norm_i.scale"][layer]
    )
    load_linear(
        module.self_attn.qkv_proj,
        c,
        "attention.wqkv_i",
        layer
    )
    load_linear(
        module.self_attn.o_proj,
        c,
        "attention.wo_i",
        layer
    )

    load(
        module.post_attention_layernorm.weight,
        c["ffn_norm_i.scale"][layer]
    )
    load_mlp(
        module.mlp,
        c,
        "feed_forward.ffn_i",
        layer
    )


def main():
    print("\n ====================== \n")
    print(f"Converting checkpoint {CHECKPOINT} into {CONFIG}")
    print("")

    checkpoint_raw = np.load(os.path.join(CHECKPOINT, "state_dict.npy"), allow_pickle=True).item()
    
    checkpoint = {}
    for k, v in checkpoint_raw.items():

        if isinstance(v, np.ndarray):
            checkpoint[k] = torch.from_numpy(v)
    print("Checkpoint loaded!")

    config = CWICConfig.from_json_file(CONFIG)
    config.stripe_size = config.hidden_size // checkpoint["h.blocks.attention.wo_i.thresholds"].shape[-1]
    config.head_stripe_size = checkpoint["lm_head_i.W.kernel"].shape[-1] // checkpoint["lm_head_i.thresholds"].shape[-1]
    print(f"Config loaded with stripe_size={config.stripe_size} and head_stripe_size={config.head_stripe_size}!")

    model = CWICForCausalLM(config)
    print("Model loaded!")

    # LM components
    load(
        model.model.embed_tokens.weight,
        F.linear(
            checkpoint["wte.embedding"],
            checkpoint["embed_proj_i.kernel"].T
        )
    )
    load(
        model.model.norm.weight,
        checkpoint["ln_f_i.scale"]
    )
    load_linear(
        model.lm_head,
        checkpoint,
        "lm_head_i",
        None
    )

    # Decoder layers
    for i, layer in enumerate(model.model.layers):
        load_layer(
            layer,
            checkpoint,
            "h.blocks",
            i
        )
    
    print("Weights loaded!")

    model.save_pretrained(SAVE_PATH)
    AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct").save_pretrained(SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}!")

    print("\n SUCCESS! \n")


if __name__ == "__main__":
    main()