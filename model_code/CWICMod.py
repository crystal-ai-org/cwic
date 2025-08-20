import math
from typing import Optional

from einops import rearrange

from model_code.loss_names import LossKeys
from model_code.utils import replace_grad

from jax.sharding import PartitionSpec


import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

from flax import nnx

from flax.typing import (
    Dtype,
    Initializer,
    PrecisionLike,
)



EPS = 1e-7

USE_TRITON_KERNELS_IN_DECODING = False


""" Kernel Functions """


def soft_jump(x, gate, threshold, bandwidth, where_vals, batch_dims=1, ste=False, train_mode=True):
    dtype = x.dtype
    x = x.astype(jnp.float32)
    gate = gate.astype(jnp.float32)
    threshold = threshold.astype(jnp.float32)
    bandwidth = bandwidth.astype(jnp.float32)

    # to mean along batch dims
    batch_mean = lambda x: jnp.mean(
        x,
        axis=tuple(range(batch_dims)),
        where=(
            None
            if where_vals is None
            else where_vals.reshape(*where_vals.shape, *([1] * (x.ndim - where_vals.ndim)))
        ),
    )
    if train_mode:
        # calculate the kernel hits
        kernel = jax.nn.sparse_sigmoid((gate - threshold) / (bandwidth / 2) + EPS)
        nog_kernel = jax.nn.sparse_sigmoid(
            (jax.lax.stop_gradient(gate) - threshold) / (bandwidth / 2) + EPS
        )

        # get the occupance (with grads to gate)
        occupance = replace_grad((gate >= threshold).astype(jnp.float32), kernel)

        # calculate the activation frequency (no grads)
        frequency = jax.lax.stop_gradient(batch_mean(occupance))

        # calculate the final output
        y = x * occupance
        if ste:
            # thresholds get exact grads, x gets STE scaled by frequency
            y = replace_grad(y, jax.lax.stop_gradient(x) * nog_kernel + x)
        else:
            y = replace_grad(
                y, x * jax.lax.stop_gradient(occupance) + jax.lax.stop_gradient(x) * nog_kernel
            )

        # hits are same shape as frequency, showing which kernels were in (0,1)
        hits = jax.lax.stop_gradient(
            batch_mean(
                (jnp.abs(gate - threshold) < (bandwidth / 2)).astype(jnp.float32),
            )
        )

        return (
            y.astype(dtype),
            occupance.astype(jnp.float32),
            hits.astype(dtype),
            frequency.astype(dtype),
        )
    else:
        occupance = (gate >= threshold).astype(jnp.float32)

        # calculate the activation frequency (no grads)

        frequency = jax.lax.stop_gradient(batch_mean(occupance)).astype(dtype)
        # frequency = jnp.zeros(occupance.shape[batch_dims:],dtype=dtype)

        # calculate the final output
        y = x * occupance

        # hits are same shape as frequency, showing which kernels were in (0,1)
        # hits = frequency
        hits = jax.lax.stop_gradient(
            batch_mean(
                (jnp.abs(gate - threshold) < (bandwidth / 2)).astype(dtype),
            )
        )

        return (
            y.astype(dtype),
            occupance.astype(jnp.float32),
            hits,
            frequency,
        )


""" Loss Functions """


def get_usage_penalty(occupance, where_vals):
    # calculate usage penalty based on the effective number of different experts used
    # https://en.wikipedia.org/wiki/Effective_number_of_parties#Quadratic
    counts = occupance.reshape(-1, occupance.shape[-1]).sum(0, where=where_vals.reshape(-1, 1))
    counts = counts.at[0].add(1)  # add a ghost occupance in one position to avoid div by zero
    usage = counts / counts.sum(-1, keepdims=True)
    effective_parties = 1 / (usage**2).sum()
    return jnp.log2(occupance.shape[-1] / effective_parties)


def get_information_loss(occupance, where_vals):
    return (
        occupance.reshape(-1, occupance.shape[-1]).sum(0, where=where_vals.reshape(-1, 1)) > 0
    ).mean()


def get_flop_ratio(base_flops, my_flops, where_vals):
    return base_flops / my_flops.mean(where=where_vals)


""" Modules """


class CWICDense(nnx.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stripe_size: int,
        threshold_learning_scale: float,
        bandwidth: float,
        threshold_shift_cap: float,
        use_bias: bool = True,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = nnx.initializers.variance_scaling(
            1.0, "fan_in", "truncated_normal"
        ),
        thresholds_init: Initializer = nnx.initializers.zeros_init(),
        bias_init: Initializer = nnx.initializers.zeros_init(),
        rngs: nnx.Rngs = nnx.Rngs(),
    ):
        # stripe_size = out_features
        self.out_slice = out_features
        if stripe_size > out_features:
            stripe_size = out_features
        out_features = ((out_features + (stripe_size - 1)) // stripe_size) * stripe_size
        num_stripes = out_features // stripe_size

        # save args
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        self.dtype = dtype if dtype is not None else param_dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.rngs = rngs

        self.base_num_params = self.in_features * self.out_features
        self.base_flops = self.base_num_params

        self.num_stripes = num_stripes

        self.threshold_learning_scale = nnx.Variable(
            jnp.ones((), self.param_dtype) * jnp.sqrt(self.in_features) * threshold_learning_scale
        )
        self.threshold_shift_cap = threshold_shift_cap

        # init params
        self.W = nnx.Linear(
            self.in_features,
            self.out_features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

        self.thresholds = nnx.Param(
            thresholds_init(rngs.params(), (self.in_features, self.num_stripes), self.param_dtype)
        )
        self.thresholds_computed = nnx.Variable(
            thresholds_init(rngs.params(), (self.in_features, self.num_stripes), self.param_dtype)
        )
        self.thresholds_old = nnx.Variable(
            thresholds_init(rngs.params(), (self.in_features, self.num_stripes), self.param_dtype)
        )
        self.bandwidth = nnx.Variable(jnp.ones((), self.param_dtype) * bandwidth)

        self.dist_tracker = DistributionTracker(self.in_features)

        self.out_mu_computed = nnx.Variable(jnp.zeros((self.out_features,), self.dtype))

        # trackers
        self.flop_ratio_trace = nnx.Variable(jnp.ones((), dtype=jnp.float32))
        hist_trace_rows = 0
        self.hist_trace = (
            None
            if hist_trace_rows == 0
            else nnx.Variable(jnp.zeros((hist_trace_rows, self.in_features), self.dtype))
        )

        self.train_mode = False

    def transfer_teacher(self, rngs: nnx.Rngs, kernel: Array, permutation=None):
        # assert kernel.shape == self.W.kernel.value.shape

        self.W.kernel.value = self.W.kernel.value.at[
            ..., : kernel.shape[-2], : kernel.shape[-1]
        ].set(kernel)

    def clamp_thresholds(self):
        max_shift = (
            (self.bandwidth.value[..., None, None]) / self.threshold_learning_scale.value * self.threshold_shift_cap
        )
        self.thresholds.value = self.thresholds.value.at[:].set(
            jnp.clip(
                jnp.clip(self.thresholds.value, 0.0),
                self.thresholds_old.value - max_shift,
                self.thresholds_old.value + max_shift,
            )
        )
        self.thresholds_old.value = self.thresholds.value.copy()

    def __call__(
        self,
        x: Array,
        where=None,
    ):
        if where is None:
            where = jnp.ones_like(x[..., 0]) > 0

        mu, std = self.dist_tracker(x)
        if self.train_mode:
            self.out_mu_computed.value = self.W(mu).astype(self.dtype)

        if self.train_mode:
            # get the thresholds and bandwidth
            thresholds = (
                self.thresholds.value * self.threshold_learning_scale.value * std[..., None]
            )
            self.thresholds_computed.value = thresholds
        else:
            thresholds = self.thresholds_computed.value
        bandwidth = self.bandwidth.value[..., None, None] * std[..., None]

        @partial(nnx.remat)
        def remat_fn(thresholds, bandwidth, Wkernel, out_mu_computed, where):
            # mean preservation: demeaning
            x_off = x - mu.astype(x.dtype)
            check = jnp.abs(x_off)
            z, occupance, hits, freq = soft_jump(
                x_off[..., None],
                check[..., None],
                thresholds,
                bandwidth,
                where,
                batch_dims=(x.ndim - 1),
                ste=True,
                train_mode=self.train_mode,
            )
            occupance = occupance.mean(-1)
            if not self.train_mode:
                hits = jnp.zeros_like(hits)
                freq = jnp.zeros_like(freq)

            if (
                math.prod(x_off.shape[:-1]) == 1
                and not self.train_mode
                and USE_TRITON_KERNELS_IN_DECODING
            ):
                assert "USE_TRITON_KERNELS_IN_DECODING not supported yet"
                # y = smm(
                #     x_off,
                #     kernel,
                #     thresholds.T,
                #     128,
                #     128,
                #     self.out_features // self.num_stripes,
                # )
            else:
                y = jnp.einsum(
                    "... i n, i n q -> ... n q",
                    # mean preservation: add the mean levelset
                    z + mu[...,None] if self.train_mode else z,
                    rearrange(
                        Wkernel, "i (stripes q) -> i stripes q", stripes=z.shape[-1]
                    ),
                    preferred_element_type=z.dtype,
                    precision=self.precision,
                )
                y = rearrange(y, "... n q -> ... (n q)")

            # mean preservation: add the mean levelset
            y = y if self.train_mode else y + out_mu_computed
            y = y[..., : self.out_slice]

            my_flops = self.out_features * occupance.sum(-1)

            in_bandwidth = hits.mean() if self.train_mode else 0.0
            if self.hist_trace is None:
                occupance=None

            return y, occupance,  my_flops, in_bandwidth
        
        y, occupance,  my_flops, in_bandwidth = remat_fn(
            thresholds, bandwidth, self.W.kernel.value, self.out_mu_computed.value, where
        )
        sparse_flops = my_flops
        # compute flops

        out = (
            jnp.astype(y, self.dtype),
            {
                LossKeys.FLOPS_BASE: self.base_flops,
                LossKeys.FLOPS_CWIC: my_flops,
                f"sparse_{LossKeys.FLOPS_CWIC}": sparse_flops,
                "in_bandwidth": in_bandwidth,
            },
        )

        self.flop_ratio_trace.value = get_flop_ratio(self.base_flops, my_flops, where)

        if self.hist_trace is not None:
            assert occupance is not None
            hist = jnp.where(where[..., None], occupance, 0.0).reshape(-1, occupance.shape[-1])[
                : self.hist_trace.value.shape[0]
            ]
            self.hist_trace.value = hist

        return out


class CWICFFN(nnx.Module):

    def __init__(
        self,
        in_features: int,
        inter_features: int,
        out_features: int,
        threshold_learning_scale: float,
        bandwidth: float,
        threshold_shift_cap: float,
        gate_stripe_size: int,
        gate_threshold_learning_scale: float,
        gate_bandwidth: float,
        gate_threshold_shift_cap: float,
        use_bias: bool = True,
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = nnx.initializers.variance_scaling(
            1.0, "fan_in", "truncated_normal"
        ),
        gate_thresholds_init: Initializer = nnx.initializers.zeros_init(),
        bias_init: Initializer = nnx.initializers.zeros_init(),
        rngs: nnx.Rngs = nnx.Rngs(),
        **kwargs,
    ):

        # save argsadj_med,adj_std
        self.in_features = in_features
        self.inter_features = inter_features
        self.out_features = out_features
        self.use_bias = use_bias

        self.dtype = dtype if dtype is not None else param_dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.rngs = rngs

        self.base_num_params = self.inter_features * (2 * self.in_features + self.out_features)
        self.base_flops = self.base_num_params

        self.threshold_learning_scale = nnx.Variable(
            jnp.ones((), self.param_dtype) * jnp.sqrt(self.in_features) * threshold_learning_scale
        )

        self.gate = CWICDense(
            in_features,
            inter_features,
            gate_stripe_size,
            gate_threshold_learning_scale,
            gate_bandwidth,
            gate_threshold_shift_cap,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            thresholds_init=gate_thresholds_init,
            bias_init=bias_init,
            rngs=rngs,
        )

        self.up = nnx.Linear(
            self.in_features,
            self.inter_features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

        self.down = nnx.Linear(
            self.inter_features,
            self.out_features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

        self.thresholds = nnx.Param(
            (jnp.zeros((self.inter_features,), self.param_dtype))
            / self.threshold_learning_scale.value
        )
        self.threshold_shift_cap = threshold_shift_cap
        self.thresholds_old = nnx.Variable(self.thresholds.value.copy())
        self.bandwidth = nnx.Variable(jnp.ones((), self.param_dtype) * bandwidth)

        self.mad_tracker = DistributionTracker(self.inter_features)
        self.dist_tracker = DistributionTracker(self.inter_features)

        # trackers
        self.flop_ratio_trace = nnx.Variable(jnp.ones((), dtype=jnp.float32))
        hist_trace_rows = 0
        self.hist_trace = (
            None
            if hist_trace_rows == 0
            else nnx.Variable(
                jnp.zeros(
                    (hist_trace_rows, self.inter_features),
                    self.param_dtype,
                )
            )
        )

    def transfer_teacher(
        self,
        rngs: nnx.Rngs,
        up: Array,
        down: Array,
        gate: Array,
    ):
        self.gate.transfer_teacher(rngs, gate)

        assert up.shape == self.up.kernel.value.shape
        self.up.kernel.value = self.up.kernel.value.at[:].set(jnp.array(up, copy=True))

        assert down.shape == self.down.kernel.value.shape
        self.down.kernel.value = self.down.kernel.value.at[:].set(jnp.array(down, copy=True))

    def clamp_thresholds(self):
        max_shift = (
            self.bandwidth.value[..., None]/ self.threshold_learning_scale.value  * self.threshold_shift_cap
        )
        self.thresholds.value = self.thresholds.value.at[:].set(
            jnp.clip(
                jnp.clip(self.thresholds.value, 0.0),
                self.thresholds_old.value - max_shift,
                self.thresholds_old.value + max_shift,
            )
        )
        self.thresholds_old.value = self.thresholds.value.copy()

    def __call__(self, x: Array, where=None):
        if where is None:
            where = jnp.ones_like(x[..., 0]) > 0

        pre_sil, gate_aux = self.gate(x, where)
        sil = jax.nn.silu(pre_sil).astype(x.dtype)

        check = jnp.abs(sil)

        # ONLY for scaling bandwidth
        std = self.dist_tracker(sil)[1]
        mad = self.mad_tracker(check)[0]

        thresholds = self.thresholds.value * self.threshold_learning_scale.value * mad
        bandwidth = jax.lax.stop_gradient(self.bandwidth.value[..., None] * std)

        z, occupance, hits, freq = soft_jump(sil, check, thresholds, bandwidth, where, ste=True)

        y = self.down(self.up(x) * z)

        my_flops = gate_aux[LossKeys.FLOPS_CWIC] + (
            self.in_features + self.out_features
        ) * occupance.sum(-1)
        sparse_flops = my_flops

        in_bandwidth = gate_aux["in_bandwidth"] + hits.mean()

        out = (
            jnp.astype(y, self.dtype),
            {
                LossKeys.FLOPS_BASE: self.base_flops,
                LossKeys.FLOPS_CWIC: my_flops,
                f"sparse_{LossKeys.FLOPS_CWIC}": sparse_flops,
                "in_bandwidth": in_bandwidth,
            },
        )

        self.flop_ratio_trace.value = get_flop_ratio(
            self.base_flops - self.gate.base_flops,
            my_flops - gate_aux[LossKeys.FLOPS_CWIC],
            where,
        )

        if self.hist_trace is not None:
            hist = jnp.where(where[..., None], occupance, 0.0).reshape(-1, occupance.shape[-1])[
                : self.hist_trace.value.shape[0]
            ]
            self.hist_trace.value = hist

        return out


class DistributionTracker(nnx.Module):

    def __init__(
        self,
        hidden_size,
        quantile_bs: int = 4,
        upper_quantile: float = 0.841,
        beta: float = 0.99,
        param_dtype=jnp.float32,
    ):
        self.hidden_size = hidden_size
        self.quantile_bs = quantile_bs
        self.upper_quantile = upper_quantile
        self.param_dtype = param_dtype

        self.beta = nnx.Variable(jnp.array(beta, self.param_dtype))
        self.steps = nnx.Variable(jnp.zeros((), self.param_dtype))

        self.med = nnx.Variable(jnp.zeros((hidden_size,), self.param_dtype))
        self.upp = nnx.Variable(jnp.zeros((hidden_size,), self.param_dtype))

        self.train_mode = False

        self.adj_med_computed = nnx.Variable(jnp.zeros((hidden_size,), self.param_dtype))
        self.adj_std_computed = nnx.Variable(jnp.zeros((hidden_size,), self.param_dtype))

    def __call__(
        self,
        x,
    ):
        if self.train_mode:
            # broadcast where_vals to x shape
            assert x.shape[-1] == self.hidden_size

            # reduce batch size
            x = x[: self.quantile_bs]

            # reshape batched vectors
            x = x.reshape(-1, self.hidden_size).astype(self.param_dtype)

            # get x distribution
            new_med = jnp.median(x, 0)
            new_upp = jnp.quantile(x, self.upper_quantile, 0)

            # calculate update size
            delta = 1.0 if self.train_mode else 0.0
            beta = self.beta.value**delta

            # calculate new values
            med = beta * self.med.value + (1.0 - beta) * new_med
            upp = beta * self.upp.value + (1.0 - beta) * new_upp

            old_med = self.med.value.copy()
            old_upp = self.upp.value.copy()
            old_steps = self.steps.value.copy()

            self.med.value = self.med.value.at[...].set(jax.lax.stop_gradient(med))
            self.upp.value = self.upp.value.at[...].set(jax.lax.stop_gradient(upp))
            self.steps.value = self.steps.value.at[...].add(jax.lax.stop_gradient(delta))

            trig = old_steps > 1.0
            med = jnp.where(trig, old_med, med)
            upp = jnp.where(trig, old_upp, upp)
            steps = jnp.where(trig, old_steps, self.steps.value)

            # adjust for bias
            div = 1 - self.beta.value**steps
            adj_med = med / (div + EPS)
            adj_upp = upp / (div + EPS)
            adj_med, adj_std = (
                jax.lax.stop_gradient(adj_med),
                jax.lax.stop_gradient(adj_upp - adj_med + EPS),
            )
            self.adj_med_computed.value, self.adj_std_computed.value = adj_med, adj_std

        else:
            adj_med, adj_std = self.adj_med_computed.value, self.adj_std_computed.value

        return adj_med, adj_std
