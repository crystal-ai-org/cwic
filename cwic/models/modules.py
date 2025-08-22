from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D
from transformers.modeling_layers import GradientCheckpointingLayer

try:
    from cwic_triton.triton_torch import smm
except:
    smm = None
from utils.torch_utils import attach_gradient


logger = logging.get_logger(__name__)


class CWICLinear(GradientCheckpointingLayer):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stripe_size: int | None,
        bias: Optional[bool] = True,
        threshold_lr_scale: float = 1.0,
        threshold_init: float = 0.1,
        threshold_minimum: float = 1e-3,
        bandwidth: float = 0.1,
        stats_beta: float = 0.99,
        median_iters: int = 3,
        eps: float = 1e-7,
        do_checkpointing: bool = False,
    ):
        super().__init__()

        # basic configs
        self.in_features = in_features
        self.bandwidth = bandwidth
        self.eps = eps
        self.do_checkpointing = do_checkpointing

        # handle stripe sizes
        stripe_size = min(stripe_size, out_features) if stripe_size is not None else out_features
        self.stripe_size = stripe_size
        if self.stripe_size % 16 != 0:
            logger.warning(
                f"`stripe_size` {self.stripe_size} is not divisible by 16. This means that the accelerated kernel with not be used for inference."
            )

        if out_features % stripe_size == 0:
            self.og_out_features = out_features
            self.out_features = out_features

        else:
            self.og_out_features = out_features

            out_features = stripe_size * (1 + out_features // stripe_size)
            self.out_features = out_features

            logger.warning(
                f"`out_features` {self.og_out_features} is not divisible by `stripe_size` {stripe_size}. Adjusting `out_features` to be {out_features}"
            )

        assert (
            self.out_features % self.stripe_size == 0
        ), f"out_features {self.out_features} is not divisible by stripe_size {self.stripe_size}"
        self.num_stripes = self.out_features // self.stripe_size

        # note that this is transposed compared to nn.Linear for inference kernel compatibility
        self.weight = nn.Parameter(torch.randn(in_features, out_features) / (in_features**0.5))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

        # when the argument threshold_lr_scale is 1.0, the thresholds move at the same 'speed' as the weights
        self.threshold_lr_scale = threshold_lr_scale * (in_features**0.5)
        self.threshold_minimum = threshold_minimum / self.threshold_lr_scale

        self.thresholds = nn.Parameter(
            torch.zeros(self.num_stripes, self.in_features)
            + (threshold_init / self.threshold_lr_scale)
        )

        self.distribution_tracker = RobustDistributionTracker(
            self.in_features, beta=stats_beta, num_iters=median_iters, eps=eps
        )

        self.cached_weight = None
        self.cached_post_mu = None
        self.cached_thresholds = None

    def _get_weight(self) -> torch.Tensor:

        f = lambda x: x.T.view(1, self.num_stripes, self.stripe_size, self.in_features)

        if self.training:
            self.cached_weight = None
            return f(self.weight)

        else:
            # Use cached weight during inference if available
            if self.cached_weight is None:
                self.cached_weight = f(self.weight).detach().contiguous()

            return self.cached_weight

    def _get_post_mu(self, mu) -> torch.Tensor:

        f = lambda x, y: torch.einsum("a b, a -> b", x, y)

        if self.training:
            self.cached_post_mu = None
            return f(self.weight, mu)

        else:
            # Use cached weight during inference if available
            if self.cached_post_mu is None:
                self.cached_post_mu = f(self.weight, mu).detach().contiguous()

            return self.cached_post_mu

    def _get_thresholds(self, std) -> torch.Tensor:

        f = lambda x, s: (x * s[None] * self.threshold_lr_scale)[None]

        if self.training:
            self.cached_thresholds = None
            return f(self.thresholds, std)

        else:
            # Use cached weight during inference if available
            if self.cached_thresholds is None:
                self.cached_thresholds = f(self.thresholds, std).detach().contiguous()

            return self.cached_thresholds

    def __call__(self, *args, **kwargs):
        if self.do_checkpointing:
            return GradientCheckpointingLayer.__call__(self, *args, **kwargs)
        else:
            return nn.Module.__call__(self, *args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        statistics_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shape annotations:

         - `B`: batch size
         - `N`: number of stripes
         - `S`: stripe size
         - `I`: input features
         - `O`: output features
        """

        # [I], [I]
        mu, std = self.distribution_tracker(x, statistics_mask=statistics_mask)

        # [B, 1, I]
        og_shape = x.shape[:-1]
        batched = math.prod(og_shape) > 1
        x = x.view(-1, 1, self.in_features)

        thresholds = self._get_thresholds(std) # [1, N, I]

        # [B, 1, I]
        x_demeaned = x - mu[None, None]
        x_gate = x_demeaned.abs()

        if smm is None or self.training or batched or self.stripe_size % 16 != 0:

            bandwidth = (self.bandwidth * std[None, None]) + self.eps  # [1, 1, I]

            # [B, N, I], [B, N, I]
            if self.training:
                x_masked, mask = step_with_grads(x_demeaned, x_gate, thresholds, bandwidth)
            else:
                mask = (x_gate > thresholds).to(x.dtype)
                x_masked = x_demeaned * mask

            # [B, N, I, 1]
            x = (x_masked + mu[None, None]).unsqueeze(-1)

            # [1, N, S, I]
            w = self._get_weight()

            # [B, N, S, 1]
            y = torch.einsum("B N S I, B N I P -> B N S P", w, x)

        else:

            mask = (x_gate > thresholds).to(x.dtype)

            y = smm(x_demeaned, self.weight, thresholds, stripe_size=self.stripe_size).view(
                -1, self.out_features
            )
            y = y + self._get_post_mu(mu)

            if self.bias is not None:
                y = y + self.bias[None]

        # [B, O]
        y = y.view(-1, self.out_features)
        if self.bias is not None:
            y = y + self.bias[None]

        # move y back to the original shape
        y = y.view(*og_shape, self.out_features)

        # correct the shape to the corrected output size
        if self.og_out_features != self.out_features:
            y = y[..., : self.og_out_features]

        # calculate the parameter usage
        active_params = self.stripe_size * mask.view(*og_shape, -1).sum(dim=-1)
        dense_params = self.stripe_size * torch.ones_like(mask).view(*og_shape, -1).sum(dim=-1)

        return y, dense_params, active_params


class CWICMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        inter_features: int,
        out_features: int,
        stripe_size: int | None,
        hidden_act: str = "silu",
        bias: bool = True,
        threshold_lr_scale: float = 1.0,
        threshold_init: float = 0.1,
        threshold_minimum: float = 1e-3,
        bandwidth: float = 0.1,
        stats_beta: float = 0.99,
        median_iters: int = 3,
        eps: float = 1e-7,
    ):
        super().__init__()

        self.in_features = in_features
        self.inter_features = inter_features
        self.out_features = out_features

        self.bandwidth = bandwidth
        self.eps = eps

        self.gate = CWICLinear(
            in_features,
            inter_features,
            stripe_size,
            bias=bias,
            threshold_lr_scale=threshold_lr_scale,
        )

        self.up = Conv1D(inter_features, in_features)
        self.down = nn.Linear(inter_features, out_features, bias)

        self.threshold_lr_scale = threshold_lr_scale * (in_features**0.5)
        self.threshold_minimum = threshold_minimum / self.threshold_lr_scale

        self.thresholds = nn.Parameter(
            torch.zeros(self.inter_features) + (threshold_init / self.threshold_lr_scale)
        )

        self.act_fn = ACT2FN[hidden_act]

        self.distribution_tracker = RobustDistributionTracker(
            inter_features, beta=stats_beta, num_iters=median_iters, eps=eps
        )
        self.mad_tracker = RobustDistributionTracker(
            self.inter_features, beta=stats_beta, num_iters=median_iters, eps=eps
        )

    def forward(
        self,
        x: torch.Tensor,
        statistics_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        z, gate_dense_params, gate_active_params = self.gate(x, statistics_mask=statistics_mask)
        z = self.act_fn(z)

        std = self.distribution_tracker(z, statistics_mask=statistics_mask)[1]
        mad = self.mad_tracker(z.abs(), statistics_mask=statistics_mask)[0]

        thresholds = (self.thresholds * self.threshold_lr_scale * mad).view(
            *[1 for _ in range(x.ndim - 1)], -1
        )
        bandwidth = (self.bandwidth * std).view(*[1 for _ in range(x.ndim - 1)], -1) + self.eps

        z_masked, mask = step_with_grads(z, z.abs(), thresholds, bandwidth)

        y = self.down(self.up(x) * z_masked)

        active_params = gate_active_params + (self.in_features + self.out_features) * mask.sum(
            dim=-1
        )
        dense_params = gate_dense_params + (
            self.in_features + self.out_features
        ) * torch.ones_like(mask).sum(dim=-1)

        return y, dense_params, active_params


def step_with_grads(
    x: torch.Tensor, x_gate: torch.Tensor, thresholds: torch.Tensor, bandwidth: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    mask = (x_gate > thresholds).to(x.dtype)

    kernel = F.hardsigmoid(6 * (x_gate - thresholds) / bandwidth)
    nog_kernel = F.hardsigmoid(6 * (x_gate.detach() - thresholds) / bandwidth)

    mask = attach_gradient(kernel, mask)
    nog_mask = attach_gradient(nog_kernel, mask)

    out = attach_gradient(
        x,
        x.detach() * nog_mask,
    )

    return out, mask


class RobustDistributionTracker(nn.Module):

    def __init__(
        self,
        hidden_size,
        beta: float = 0.99,
        num_iters: int = 3,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.beta = beta
        self.num_iters = num_iters
        self.eps = eps

        self.register_buffer("steps", torch.zeros((), dtype=torch.float32), persistent=True)

        self.register_buffer(
            "med", torch.zeros((hidden_size,), dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "aad", torch.zeros((hidden_size,), dtype=torch.float32), persistent=True
        )

    def forward(
        self,
        x,
        statistics_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()

        if self.training and not torch.is_grad_enabled():
            with torch.no_grad():

                self.steps += 1.0
                debiaser = 1 / (1 - self.beta**self.steps)

                x = x.view(-1, self.hidden_size)
                if statistics_mask is not None:
                    statistics_mask = statistics_mask.view(-1, 1).to(x.dtype).detach()
                else:
                    statistics_mask = torch.ones_like(x[:, :1])

                new_med = geometric_median(
                    x,
                    num_iters=self.num_iters,
                    dim=0,
                    mask=statistics_mask,
                    eps=self.eps,
                    verbose=False,  # (self.steps < 1.5).all()
                )
                self.med.copy_(self.beta * self.med + (1 - self.beta) * new_med)
                med_debiased = self.med * debiaser

                new_aad = ((x - med_debiased[None]) * statistics_mask).mean(
                    0
                ) / statistics_mask.mean(0)
                self.aad.copy_(self.beta * self.aad + (1 - self.beta) * new_aad)
                aad_debiased = self.aad * debiaser

                # assuming that x is gaussian, we scale the AAD to get the STD
                return med_debiased, aad_debiased / math.sqrt(2 * math.pi)

        debiaser = 1 / (self.eps + (1 - self.beta**self.steps))

        med_debiased = self.med * debiaser
        aad_debiased = self.aad * debiaser

        return med_debiased, aad_debiased / math.sqrt(2 * math.pi)


def geometric_median(x, num_iters, dim, mask=None, eps=1e-7, verbose=False):
    assert num_iters >= 0

    if mask is None:
        mask = torch.ones_like(x)

    x = x * mask
    scale = 1 / (mask.mean(dim, keepdim=True) + eps)

    mu = x.mean(dim, keepdim=True) * scale

    if verbose:
        print(f"Target Median: {torch.median(x, dim=dim).values}")
        print(f"Initial Mu: {mu.squeeze(dim)}")

    for _ in range(num_iters):
        if verbose:
            print(f"Iteration {_} Mu: {mu.squeeze(dim)}")

        w = 1 / ((x - mu).abs() + eps)
        w = w / (w.mean(dim, keepdim=True) + eps)

        mu = (x * w).mean(dim, keepdim=True) * scale

    if verbose:
        print(f"Final Mu: {mu.squeeze(dim)}")

    return mu.squeeze(dim)
