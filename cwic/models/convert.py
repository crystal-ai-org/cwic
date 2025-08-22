import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaForCausalLM,
)

from .modelling_cwic import (
    CWICAttention,
    CWICDecoderLayer,
    CWICForCausalLM,
)
from .modelling_cwic import LlamaRMSNorm as CWICRMSNorm
from .configuration_cwic import CWICConfig
from .modules import CWICLinear, CWICMLP
def _convert_norm(
    cwic: CWICRMSNorm,
    llama: LlamaRMSNorm
):
    cwic.weight.data = llama.weight.data


def _convert_linear(
    cwic: CWICLinear,
    linear: nn.Linear
):
    cwic.weight.data[:, :cwic.og_out_features] = linear.weight.data.T
    if cwic.bias is not None:
        cwic.bias.data[:cwic.og_out_features] = linear.bias.data


def _convert_mlp(
    cwic: CWICMLP,
    mlp: LlamaMLP
):
    
    _convert_linear(
        cwic.gate,
        mlp.gate_proj,
    )

    # TODO: handle the bias for these in case of bias in the llama model
    cwic.up.weight.data = mlp.up_proj.weight.data.T
    cwic.down.weight.data = mlp.down_proj.weight.data


def _convert_attn(
    cwic: CWICAttention,
    attn: LlamaAttention
):
    
    llama_qkv = nn.Linear(
        cwic.qkv_proj.in_features,
        cwic.qkv_proj.og_out_features,
    )
    llama_qkv.weight.data = torch.cat(
        [
            attn.q_proj.weight.data,
            attn.k_proj.weight.data,
            attn.v_proj.weight.data
        ],
        dim=0
    )
    _convert_linear(
        cwic.qkv_proj,
        llama_qkv
    )

    _convert_linear(
        cwic.o_proj,
        attn.o_proj
    )


def _convert_layer(
    cwic: CWICDecoderLayer,
    layer: LlamaDecoderLayer
):
    
    _convert_norm(
        cwic.input_layernorm,
        layer.input_layernorm
    )
    _convert_attn(
        cwic.self_attn,
        layer.self_attn
    )

    _convert_norm(
        cwic.post_attention_layernorm,
        layer.post_attention_layernorm
    )
    _convert_mlp(
        cwic.mlp,
        layer.mlp
    )


@torch.no_grad()
def llama_to_cwic(
    llama: LlamaForCausalLM,
    **kwargs,
) -> CWICForCausalLM:
    
    base_config = llama.config.to_dict()
    base_config["tie_word_embeddings"] = False
    base_config.update(kwargs)

    config = CWICConfig(**base_config)
    print(config)
    model = CWICForCausalLM(config).to(llama.device, llama.lm_head.weight.dtype)

    # LM components
    model.model.embed_tokens.weight.data = llama.model.embed_tokens.weight.data
    _convert_norm(
        model.model.norm,
        llama.model.norm
    )
    _convert_linear(
        model.lm_head,
        llama.lm_head
    )

    for i in range(config.num_hidden_layers):
        _convert_layer(
            model.model.layers[i],
            llama.model.layers[i]
        )
    
    for p in model.parameters():
        p.data = p.data.clone().detach().contiguous()

    return model
