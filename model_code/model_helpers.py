import jax.numpy as jnp
import numpy as np

from typing import Tuple, Optional

import math


from model_code.config import CWICConfig


def _compute_default_rope_parameters(
    config: Optional[CWICConfig] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple["np.ndarray", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    base = None
    if len(rope_kwargs) > 0:
        base = float(rope_kwargs["base"])
        dim = int(rope_kwargs["dim"])
    else:
        assert config is not None, "rope_kwargs must be defined or config passed"
        base = config.rope_theta
        partial_rotary_factor = (
            config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        )
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / dim))
    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: CWICConfig, seq_len: Optional[int] = None, **rope_kwargs
) -> Tuple["np.ndarray", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, seq_len, **rope_kwargs)

    factor = config.rope_scaling["factor"]  # `8` in the original implementation
    low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling[
        "high_freq_factor"
    ]  # `4` in the original implementation
    old_context_len = config.rope_scaling[
        "original_max_position_embeddings"
    ]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


def precompute_freqs_cis(
    dim: int,
    end: int,
    config: CWICConfig,
    theta: float = 10000.0,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    # freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    if (
        hasattr(config, "rope_scaling")
        and isinstance(config.rope_scaling, dict)
        and config.rope_scaling.get("rope_type", None) == "llama3"
    ):
        inv_freq, attention_factor = _compute_llama3_parameters(config, None)
    else:

        inv_freq, attention_factor = _compute_default_rope_parameters(config, None)
    t = np.arange(end)  # type: ignore
    inv_freq = np.outer(t, inv_freq).astype(dtype)  # type: ignore
    sin, cos = np.sin(inv_freq), np.cos(inv_freq)
    freqs_cis = np.complex64(cos + 1j * sin) * attention_factor
    return jnp.asarray(freqs_cis)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_emb_hf(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    reshape_xq = xq.astype(jnp.float32)
    reshape_xk = xk.astype(jnp.float32)

    # add head dim
    freqs_cis = jnp.concatenate((freqs_cis, freqs_cis), axis=-1)
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))
    cos = jnp.real(freqs_cis)
    sin = jnp.imag(freqs_cis)
    xq_out = (reshape_xq * cos) + (rotate_half(reshape_xq) * sin)
    xk_out = (reshape_xk * cos) + (rotate_half(reshape_xk) * sin)

    return xq_out.astype(dtype), xk_out.astype(dtype)


def repeat_kv(
    hidden_states: jnp.ndarray,
    n_rep: int,
) -> jnp.ndarray:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :]
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=3)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)
