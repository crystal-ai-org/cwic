import jax
from jax import lax, numpy as jnp, Array

from jax.sharding import PartitionSpec

from flax import nnx
import numpy as np

from typing import Tuple, Any, Optional, Union

from functools import partial


from flax.typing import Array, Dtype, Shape, Sharding
from flax.nnx import make_causal_mask
from model_code.loss_names import LossKeys
from model_code.model_helpers import (
    precompute_freqs_cis,
    apply_rotary_emb_hf,
    repeat_kv,
)

from .CWICMod import CWICDense, CWICFFN
from model_code.config import CWICConfig
from model_code.logical_axis import global_mesh as mesh


class NNXRefractionAttention(nnx.Module):
    def __init__(
        self,
        config: CWICConfig,
        insane: bool,
        partitioning: Sharding,
        rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
        decode: bool | None = None,
    ):

        self.config = config
        self.insane = insane
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.decode = decode

        self.wqkv_i = (
            CWICDense(
                config.hidden_size,
                self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim),
                self.config.cwic_stripe_size,
                self.config.cwic_threshold_learning_scale,
                self.config.cwic_bandwidth,
                self.config.cwic_threshold_shift_cap,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
                    partitioning,
                ),
                thresholds_init=nnx.with_partitioning(
                    nnx.initializers.zeros_init(),
                    partitioning,
                ),
                rngs=rngs,
            )
            if self.insane
            else None
        )
        self.wo_i = (
            CWICDense(
                config.hidden_size,
                config.hidden_size,
                self.config.cwic_stripe_size,
                self.config.cwic_threshold_learning_scale,
                self.config.cwic_bandwidth,
                self.config.cwic_threshold_shift_cap,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
                    partitioning,
                ),
                rngs=rngs,
            )
            if self.insane
            else None
        )

        self.wq = (
            None
            if self.insane
            else nnx.Linear(
                config.hidden_size,
                self.num_heads * self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(
                    jax.nn.initializers.normal(stddev=self.config.initializer_range), partitioning
                ),
                precision=self.precision,
                rngs=rngs,
            )
        )
        self.wk = (
            None
            if self.insane
            else nnx.Linear(
                config.hidden_size,
                self.num_key_value_heads * self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(
                    jax.nn.initializers.normal(stddev=self.config.initializer_range), partitioning
                ),
                precision=self.precision,
                rngs=rngs,
            )
        )
        self.wv = (
            None
            if self.insane
            else nnx.Linear(
                config.hidden_size,
                self.num_key_value_heads * self.head_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(
                    jax.nn.initializers.normal(stddev=self.config.initializer_range), partitioning
                ),
                precision=self.precision,
                rngs=rngs,
            )
        )
        self.wo = (
            None
            if self.insane
            else nnx.Linear(
                config.hidden_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(
                    jax.nn.initializers.normal(stddev=self.config.initializer_range), partitioning
                ),
                precision=self.precision,
                rngs=rngs,
            )
        )

        self.resid_dropout = nnx.Dropout(rate=config.resid_pdrop)

        self.causal_mask = nnx.Variable(
            jax.lax.with_sharding_constraint(
                make_causal_mask(
                    jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool"
                ),
                PartitionSpec(None, None),
            )
        )

        self.freqs_cis = nnx.Variable(
            precompute_freqs_cis(
                self.head_dim,
                config.max_sequence_length * 2,
                config,
                theta=config.rope_theta,
                dtype=self.dtype,
            )
        )

        self.cached_key: nnx.Cache[jax.Array] | None = None
        self.cached_value: nnx.Cache[jax.Array] | None = None
        self.cache_index: nnx.Cache[jax.Array] | None = None

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states,
        segment_ids,
        pad_mask,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        print_debug_on_layer: bool = False,
        decode: bool | None = None,
        rngs: nnx.Rngs = nnx.Rngs(18),
        x_gate=None,
    ):
        qkv_aux, o_aux = None, None
        if not self.insane:
            assert self.wq is not None
            assert self.wk is not None
            assert self.wv is not None
            query, key, value = (
                self.wq(hidden_states),
                self.wk(hidden_states),
                self.wv(hidden_states),
            )

            xqkv = jnp.concatenate([query, key, value], axis=-1)

        else:
            assert self.wqkv_i is not None
            xqkv, qkv_aux = self.wqkv_i(hidden_states, where=pad_mask)

            query = xqkv[..., : self.num_heads * self.head_dim]
            key = xqkv[
                ...,
                self.num_heads * self.head_dim : self.num_heads * self.head_dim
                + self.num_key_value_heads * self.head_dim,
            ]
            value = xqkv[..., -(self.num_key_value_heads * self.head_dim) :]

        query = self._split_heads(query, self.num_heads)
        key = self._split_heads(key, self.num_key_value_heads)
        value = self._split_heads(value, self.num_key_value_heads)

        freqs_cis = jnp.take(self.freqs_cis.value, position_ids, axis=0)

        query, key = apply_rotary_emb_hf(query, key, freqs_cis=freqs_cis, dtype=self.dtype)

        # we get these after rope, since it better captures behavior
        # xq_og, xk_og, xv_og = query, key, value

        query_length, key_length = query.shape[1], key.shape[1]
        mask_shift = 0

        decode = (
            (self.cached_key is not None)
            and (self.cache_index is not None)
            and (self.cached_key is not None)
        )
        if decode:
            assert (
                (self.cached_key is not None)
                and (self.cache_index is not None)
                and (self.cached_key is not None)
            )
            mask_shift = self.cache_index.value
            max_decoder_length = self.cached_key.value.shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask.value,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = self.causal_mask.value[:, :, :query_length, :key_length]

        if decode:
            if self.cached_key is None or self.cached_value is None or self.cache_index is None:
                raise ValueError(
                    "Autoregressive cache not initialized, call ``init_cache`` first."
                )
            (
                *batch_dims,
                max_length,
                num_key_value_heads,
                depth_per_head,
            ) = self.cached_key.value.shape
            # shape check of cached keys against query input
            expected_shape = tuple(batch_dims) + (query_length, self.num_heads, depth_per_head)
            if expected_shape != query.shape:
                raise ValueError(
                    "Autoregressive cache shape error, "
                    "expected query shape %s instead got %s." % (expected_shape, query.shape)
                )
            # update key, value caches with our new 1d spatial slices
            cur_index = self.cache_index.value
            zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
            indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
            key = lax.dynamic_update_slice(self.cached_key.value, key, indices)
            value = lax.dynamic_update_slice(self.cached_value.value, value, indices)
            self.cached_key.value = key
            self.cached_value.value = value
            self.cache_index.value += query_length

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        attention_mask = causal_mask
        if not decode:
            attention_mask = jnp.logical_and(
                attention_mask, (segment_ids[..., None] == segment_ids[..., None, :])[:, None]
            )

        # usual dot product attention

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attn_output = jax.nn.dot_product_attention(
            query,
            key,
            value,
            mask=attention_mask,
        )

        attn_output = self._merge_heads(attn_output)
        if not self.insane:
            assert self.wo is not None
            attn_output = self.wo(attn_output)

        else:
            assert self.wo_i is not None
            attn_output, o_aux = self.wo_i(attn_output, where=pad_mask)

        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

        if self.insane:
            aux_sum = jax.tree.map(lambda qkv, o: qkv + o, qkv_aux, o_aux)
        else:
            aux_sum = None

        return (
            attn_output,
            aux_sum,
        )

    def init_cache(self, input_shape: Shape, dtype: Dtype | None = None):
        """Initializes cache for fast autoregressive decoding. When
        ``decode=True``, this method must be called first before performing
        forward inference. When in decode mode, only one token must be passed
        at a time.

        Example usage::

        >>> from flax import nnx
        >>> import jax.numpy as jnp
        ...
        >>> batch_size = 5
        >>> embed_dim = 3
        >>> x = jnp.ones((batch_size, 1, embed_dim)) # single token
        ...
        >>> model_nnx = nnx.MultiHeadAttention(
        ...   num_heads=2,
        ...   in_features=3,
        ...   qkv_features=6,
        ...   out_features=6,
        ...   decode=True,
        ...   rngs=nnx.Rngs(42),
        ... )
        ...
        >>> # out_nnx = model_nnx(x)  <-- throws an error because cache isn't initialized
        ...
        >>> model_nnx.init_cache(x.shape)
        >>> out_nnx = model_nnx(x)
        """

        if dtype is None:
            dtype = self.dtype
        cache_shape_key = (*input_shape[:-1], self.num_key_value_heads, self.head_dim)
        cache_shape_value = (*input_shape[:-1], self.num_key_value_heads, self.head_dim)
        self.cached_key = nnx.Cache(jnp.zeros(cache_shape_key, dtype))
        self.cached_value = nnx.Cache(jnp.zeros(cache_shape_value, dtype))
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))

    def delete_cache(self):
        self.cached_key = None
        self.cached_value = None
        self.cache_index = None


class NNXRefractionMLP(nnx.Module):
    def __init__(
        self,
        config: CWICConfig,
        insane: bool,
        partitioning: Sharding,
        rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
    ):

        self.config = config
        self.insane = insane
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.ffn_i = (
            CWICFFN(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_size,
                self.config.cwic_threshold_learning_scale,
                self.config.cwic_bandwidth,
                self.config.cwic_threshold_shift_cap,
                self.config.cwic_stripe_size,
                self.config.cwic_threshold_learning_scale,
                self.config.cwic_bandwidth,
                self.config.cwic_threshold_shift_cap,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
                    partitioning,
                ),
                gate_thresholds_init=nnx.with_partitioning(
                    nnx.initializers.zeros_init(),
                    partitioning,
                ),
                rngs=rngs,
            )
            if self.insane
            else None
        )

        self.gate = (
            None
            if self.insane
            else nnx.Linear(
                config.hidden_size,
                config.intermediate_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(
                    jax.nn.initializers.normal(stddev=self.config.initializer_range), partitioning
                ),
                precision=self.precision,
                rngs=rngs,
            )
        )
        self.down = (
            None
            if self.insane
            else nnx.Linear(
                config.intermediate_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(
                    jax.nn.initializers.normal(stddev=self.config.initializer_range), partitioning
                ),
                precision=self.precision,
                rngs=rngs,
            )
        )
        self.up = (
            None
            if self.insane
            else nnx.Linear(
                config.hidden_size,
                config.intermediate_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(
                    jax.nn.initializers.normal(stddev=self.config.initializer_range), partitioning
                ),
                precision=self.precision,
                rngs=rngs,
            )
        )

        self.dropout = nnx.Dropout(rate=self.config.resid_pdrop)

    def __call__(
        self,
        x: Array,
        where,
        deterministic: bool = True,
        x_gate=None,
    ) -> Tuple[Array, dict | None]:

        # run teacher
        if self.insane:
            assert self.ffn_i is not None
            yi, aux = self.ffn_i(x, where=where)
            yi = self.dropout(yi, deterministic=deterministic)
            return yi, aux
        else:
            assert self.up is not None
            assert self.gate is not None
            assert self.down is not None
            up = self.up(x)
            gate = self.gate(x)
            inter = up * nnx.silu(gate)
            y_no_drop = self.down(inter)
            y = self.dropout(y_no_drop, deterministic=deterministic)

            # run the student

            return y, None


class NNXRefractionBlock(nnx.Module):

    def __init__(
        self,
        config: CWICConfig,
        insane: bool,
        partitioning: Sharding,
        rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
    ):

        self.config = config
        self.insane = insane
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.attention = NNXRefractionAttention(
            self.config,
            self.insane,
            partitioning,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=rngs,
        )

        if self.config.intermediate_size > 0:
            self.feed_forward = NNXRefractionMLP(
                self.config,
                self.insane,
                partitioning,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                rngs=rngs,
            )
        self.attention_norm = (
            nnx.RMSNorm(
                num_features=self.config.hidden_size,
                epsilon=self.config.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_scale=True,
                rngs=rngs,
            )
            if self.insane
            else nnx.RMSNorm(
                self.config.hidden_size,
                epsilon=self.config.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_scale=True,
                rngs=rngs,
            )
        )
        self.ffn_norm = (
            nnx.RMSNorm(
                num_features=self.config.hidden_size,
                epsilon=self.config.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_scale=True,
                rngs=rngs,
            )
            if self.insane
            else nnx.RMSNorm(
                self.config.hidden_size,
                epsilon=self.config.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_scale=True,
                rngs=rngs,
            )
        )

    def __call__(
        self,
        hidden_and_loss_and_por: Tuple[Array, Any, Array],
        segment_ids,
        pad_mask,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        print_debug_on_layer: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[Array, Any, Array]:

        hidden = hidden_and_loss_and_por[0]

        aux_prev = hidden_and_loss_and_por[1]
        layer_num = hidden_and_loss_and_por[2]
        layer_num += 1

        # perform attention
        if not self.insane:
            (
                attn_out,
                _,
            ) = self.attention(
                self.attention_norm(hidden),
                segment_ids,
                pad_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                print_debug_on_layer=print_debug_on_layer,
            )
            hidden += attn_out
        else:
            insane = hidden

            (
                attn_out_i,
                attn_aux,
            ) = self.attention(
                self.attention_norm(insane),
                segment_ids,
                pad_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                print_debug_on_layer=print_debug_on_layer,
                x_gate=insane,
            )

            # add attention to residual
            assert attn_aux is not None
            aux_prev = jax.tree.map(
                lambda a, b: a + b,
                aux_prev,
                {k: attn_aux.get(k, 0) for k in aux_prev.keys()},
            )
            hidden += attn_out_i

        assert self.config.intermediate_size > 0

        # get mlp
        feed_forward_hidden_state, mlp_aux = self.feed_forward(
            self.ffn_norm(hidden),
            pad_mask,
            deterministic=deterministic,
            x_gate=hidden,
        )
        if self.insane:
            assert mlp_aux is not None
            # add mlp to residual
            aux_prev = jax.tree.map(
                lambda a, b: a + b,
                aux_prev,
                {k: mlp_aux.get(k, 0) for k in aux_prev.keys()},
            )
        hidden += feed_forward_hidden_state

        return (hidden, aux_prev, layer_num)


class NNXRefractionBlockCollection(nnx.Module):

    def __init__(
        self,
        config: CWICConfig,
        insane: bool,
        partitioning: Sharding,
        precision: Optional[Union[jax.lax.Precision, str]],
        rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
    ):

        self.config = config
        self.insane = insane
        self.precision = precision
        self.dtype = dtype
        self.param_dtype = param_dtype

        @nnx.split_rngs(splits=self.config.num_hidden_layers)
        @partial(
            nnx.vmap, in_axes=0, out_axes=0, transform_metadata={nnx.PARTITION_NAME: None}
        )  # , split_rngs=True)
        def create_blocks(rngs):
            m = NNXRefractionBlock(
                self.config,
                self.insane,
                partitioning,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                rngs=rngs,
            )
            return m

        self.blocks = create_blocks(
            rngs,
        )

    def __call__(
        self,
        hidden_states,
        segment_ids=None,
        pad_mask=None,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        rkey=jax.random.key(7),
    ):

        i = jax.lax.with_sharding_constraint(jnp.zeros(()), None)

        aux0 = {
            LossKeys.FLOPS_BASE: jax.lax.with_sharding_constraint(
                jnp.array(0.0).astype(jnp.float32), None
            ),
            LossKeys.FLOPS_CWIC: jax.lax.with_sharding_constraint(
                jnp.zeros_like(hidden_states[..., 0]).astype(jnp.float32),
                PartitionSpec("dp", *([None] * (hidden_states.ndim - 2))),
            ),
            "in_bandwidth": jax.lax.with_sharding_constraint(jnp.array(0.0), None),
        }

        hidden_states_and_aux = (hidden_states, aux0, i)

        @nnx.split_rngs(splits=self.config.num_hidden_layers)
        @partial(
            nnx.scan, transform_metadata={nnx.PARTITION_NAME: None}
        )  # ,_shardings=(jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None,'dp', None,None)),None))
        @partial(nnx.remat, prevent_cse=False)
        def forward(x, block):

            x = block(
                x,
                segment_ids,
                pad_mask,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
                False,
                output_hidden_states,
            )

            return x, None

        hidden_states_and_aux, _ = forward(hidden_states_and_aux, self.blocks)

        hidden_states, auxi, _ = hidden_states_and_aux

        return hidden_states, auxi


class NNXRefractionModule(nnx.Module):

    def __init__(
        self,
        config: CWICConfig,
        insane: bool,
        rngs,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: Optional[Union[jax.lax.Precision, str]] = None,
    ):
        self.config = config
        self.insane = insane
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_dim = self.config.hidden_size
        partitioning = PartitionSpec(None, "dp")
        partitioning2 = PartitionSpec("dp", None)

        self.wte = nnx.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=nnx.with_partitioning(
                jax.nn.initializers.normal(stddev=self.config.initializer_range), partitioning2
            ),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(rate=self.config.embd_pdrop)
        self.h = NNXRefractionBlockCollection(
            self.config,
            self.insane,
            partitioning2,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=rngs,
        )
        self.ln_f = (
            nnx.RMSNorm(
                num_features=self.config.hidden_size,
                epsilon=self.config.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_scale=True,
                rngs=rngs,
            )
            if self.insane
            else nnx.RMSNorm(
                num_features=self.config.hidden_size,
                epsilon=self.config.rms_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_scale=True,
                rngs=rngs,
            )
        )

        self.lm_head = (
            CWICDense(
                self.config.hidden_size,
                self.config.vocab_size,
                self.config.cwic_stripe_size_lm_head,
                self.config.cwic_threshold_learning_scale,
                self.config.cwic_bandwidth,
                self.config.cwic_threshold_shift_cap,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
                    partitioning2,
                ),
                rngs=rngs,
            )
            if self.insane
            else nnx.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(
                    jax.nn.initializers.normal(stddev=self.config.initializer_range), partitioning2
                ),
                precision=self.precision,
                rngs=rngs,
            )
        )

        def eye_init(key: Any, shape: Shape, dtype=jnp.float_) -> jax.Array:
            return jnp.zeros(shape[:-2], dtype=dtype)[..., None, None] + jnp.eye(
                shape[-2], shape[-1], dtype=dtype
            )

        self.embed_proj_i = (
            nnx.Linear(
                self.config.hidden_size,
                self.config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(eye_init, partitioning2),
                precision=self.precision,
                rngs=rngs,
            )
            if self.insane
            else None
        )

    @nnx.jit
    def __call__(
        self,
        input_ids,
        segment_ids,
        pad_mask,
        position_ids,
        key,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        gen_mask=None,
    ):
        input_embeds = self.wte(input_ids.astype("i4"))

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        if self.embed_proj_i is not None:
            hidden_states = self.embed_proj_i(hidden_states)

        hidden_states, aux = self.h(
            hidden_states,
            segment_ids=segment_ids,
            pad_mask=pad_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = self.ln_f(hidden_states.astype(jnp.float32)).astype(self.dtype)

        if not self.insane:
            assert isinstance(self.lm_head, nnx.Linear)
            lm_logits = self.lm_head(hidden_states)
        else:
            assert isinstance(self.lm_head, CWICDense)
            head_mask = pad_mask
            # if gen_mask is not None:
            #     head_mask = pad_mask & gen_mask
            lm_logits, lm_head_aux_i = self.lm_head(hidden_states, where=head_mask)

            aux = jax.tree.map(
                lambda a, b: a + b,
                aux,
                {k: lm_head_aux_i.get(k, 0) for k in aux.keys()},
            )

        return lm_logits, aux

    @partial(nnx.jit, static_argnums=[1], static_argnames=["dtype"])
    def init_cache(self, input_shape: Shape, dtype: Dtype | None = None):
        @nnx.split_rngs(splits=self.config.num_hidden_layers)
        @nnx.vmap
        def helper(x: NNXRefractionAttention):
            x.init_cache(input_shape, dtype=dtype)

        helper(self.h.blocks.attention)

    @partial(nnx.jit)
    def reset_cache(self):
        assert self.h.blocks.attention.cache_index is not None
        self.h.blocks.attention.cache_index.value *= 0

    @partial(nnx.jit)
    def delete_cache(self):

        @nnx.split_rngs(splits=self.config.num_hidden_layers)
        @nnx.vmap
        def helper(x: NNXRefractionAttention):
            x.delete_cache()

        helper(self.h.blocks.attention)

    def set_init_mode(self, mode: bool):
        self.set_attributes(init_mode=mode, raise_if_not_found=False)

    def train(self, **attributes):
        super().train(train_mode=True, **attributes)

    def eval(self, **attributes):
        super().eval(train_mode=False, **attributes)

    @partial(nnx.jit)
    def clamp_thresholds(self):

        @nnx.split_rngs(splits=self.config.num_hidden_layers)
        @nnx.vmap
        def helper(x: NNXRefractionBlock):
            if x.attention.wqkv_i is not None:
                x.attention.wqkv_i.clamp_thresholds()
            if x.attention.wo_i is not None:
                x.attention.wo_i.clamp_thresholds()
            if x.feed_forward.ffn_i is not None:
                x.feed_forward.ffn_i.gate.clamp_thresholds()
                x.feed_forward.ffn_i.clamp_thresholds()

        helper(self.h.blocks)

        if isinstance(self.lm_head, CWICDense):
            self.lm_head.clamp_thresholds()


@nnx.jit
def infer_nnx_model(model: NNXRefractionModule, toks, seg_ids, pad_mask, pos, key):
    model.eval()

    vals = model(toks, seg_ids, pad_mask, pos, key)

    return vals
