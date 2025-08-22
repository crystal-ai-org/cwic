from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D


logger = logging.get_logger(__name__)


class CWICLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_stripes: int,
        bias: Optional[bool] = True,
        threshold_lr_scale: float = 1.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_stripes = num_stripes

        if out_features % num_stripes == 0:
            self.og_out_features = out_features
            self.out_features = out_features
            self.stripe_size = out_features // num_stripes

        else:
            self.og_out_features = out_features
            
            out_features = num_stripes * (1 + out_features // num_stripes)
            self.out_features = out_features
            self.stripe_size = out_features // num_stripes

            logger.warning(
                f"`out_features` {self.og_out_features} is not divisible by `num_stripes`. {num_stripes} Adjusting `out_features` to be {out_features}"
            )

        # note that this is transposed compared to nn.Linear for inference kernel compatibility
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) / (in_features ** 0.5)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features)
            )

        self.thresholds = nn.Parameter(
            torch.zeros(self.num_stripes, self.in_features)
        )

        # when the argument threshold_lr_scale is 1.0, the thresholds move at the same 'speed' as the weights
        self.threshold_lr_scale = threshold_lr_scale * (in_features ** 0.5)

        self.distribution_tracker = RobustDistributionTracker(self.in_features)


    def _get_weight(self) -> torch.Tensor:
        return self.weight.T.view(1, self.num_stripes, self.stripe_size, self.in_features)


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
        
        mu, std = self.distribution_tracker(
            x,
            statistics_mask=statistics_mask
        )

        # [B, 1, I]
        og_shape = x.shape[:-1]
        x = x.view(-1, 1, self.in_features)
        
        thresholds = (
            self.thresholds[None]
            * self.threshold_lr_scale
            * std[None, None] 
        ) # [1, N, I]

        mask = (
            (x - mu[None]).abs() > thresholds
        ).to(x.dtype) # [B, N, I]

        x = (
            (x - mu[None]) * mask
            + mu[None, None]
        ).unsqueeze(-1) # [B, N, I, 1]

        # [1, N, S, I]
        w = self._get_weight()

        # [B, N, S, 1]
        y = torch.einsum(
            "B N S I, B N I P -> B N S P",
            w, x
        )

        # [B, O]
        y = y.view(-1, self.out_features)
        if self.bias is not None:
            y = y + self.bias[None]

        # move y back to the original shape
        y = y.view(*og_shape, self.out_features)

        # correct the shape to the corrected output size
        if self.og_out_features != self.out_features:
            y = y[..., :self.og_out_features]

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
        num_stripes: int,
        hidden_act: str = "silu",
        bias: bool = True,
        threshold_lr_scale: float = 1.0
    ):
        super().__init__()

        self.in_features = in_features
        self.inter_features = inter_features
        self.out_features = out_features

        self.gate = CWICLinear(
            in_features,
            inter_features,
            num_stripes,
            bias=bias,
            threshold_lr_scale=threshold_lr_scale
        )

        self.up = Conv1D(
            inter_features,
            in_features
        )
        self.down = nn.Linear(
            inter_features,
            out_features,
            bias
        )

        self.thresholds = nn.Parameter(
            torch.zeros(inter_features)
        )

        self.threshold_lr_scale = threshold_lr_scale * (in_features ** 0.5)
        
        self.act_fn = ACT2FN[hidden_act]

        self.distribution_tracker = RobustDistributionTracker(inter_features)
        self.mad_tracker = RobustDistributionTracker(inter_features)


    def forward(
        self,
        x: torch.Tensor,
        statistics_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        z, gate_dense_params, gate_active_params = self.gate(x, statistics_mask=statistics_mask)
        z = self.act_fn(z)

        mu, std = self.distribution_tracker(
            z,
            statistics_mask=statistics_mask
        )
        mu_abs = self.mad_tracker(
            z.abs(),
            statistics_mask=statistics_mask
        )[0]

        thresholds = (
            self.thresholds
            * self.threshold_lr_scale
            * mu_abs
        ).view(*[1 for _ in range(x.ndim - 1)], -1)

        mask = (z.abs() > thresholds).to(z.dtype)
        z = z * mask

        y = self.down(
            self.up(x) * z
        )

        active_params = gate_active_params + (self.in_features + self.out_features) * mask.sum(dim=-1)
        dense_params = gate_dense_params + (self.in_features + self.out_features) * torch.ones_like(mask).sum(dim=-1)

        return y, dense_params, active_params


class RobustDistributionTracker(nn.Module):

    def __init__(
        self,
        hidden_size,
        beta: float = 0.99,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.beta = beta

        self.register_buffer(
            "steps",
            torch.zeros((), dtype=torch.float32),
            persistent=True
        )

        self.register_buffer(
            "med",
            torch.zeros((hidden_size,), dtype=torch.float32),
            persistent=True
        )
        self.register_buffer(
            "aad",
            torch.zeros((hidden_size,), dtype=torch.float32),
            persistent=True
        )


    def forward(
        self,
        x,
        statistics_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()

        if self.training and torch.is_grad_enabled():
            with torch.no_grad():

                self.steps += 1.0
                debiaser = 1 / (1 - self.beta ** self.steps)

                x = x.view(-1, self.hidden_size)
                if statistics_mask is not None:
                    statistics_mask = statistics_mask.view(-1, 1).to(x.dtype).detach()
                else:
                    statistics_mask = torch.ones_like(x[:, :1])

                new_med = geometric_median(
                    x,
                    num_iters=3,
                    dim=0,
                    mask=statistics_mask,
                )
                self.med.copy_(
                    self.beta * self.med + (1 - self.beta) * new_med
                )
                med_debiased = self.med * debiaser

                new_aad = (
                    ((x - med_debiased) * statistics_mask).mean() / statistics_mask.mean()
                )
                self.aad.copy_(
                    self.beta * self.aad + (1 - self.beta) * new_aad
                )
                aad_debiased = self.aad * debiaser

                return med_debiased, aad_debiased

        debiaser = 1 / (1e-5 + (1 - self.beta ** self.steps))

        med_debiased = self.med * debiaser
        aad_debiased = self.aad * debiaser

        return med_debiased, aad_debiased


def geometric_median(
    x,
    num_iters,
    dim,
    mask=None,
    eps=1e-7,
):
    assert num_iters >= 0

    if mask is None:
        mask = torch.ones_like(x)
    
    x = x * mask
    scale = 1 / (mask.mean(dim, keepdim=True) + eps)

    mu = x.mean(dim, keepdim=True) * scale

    for _ in range(num_iters):

        w = 1 / ((x - mu).abs() + eps)
        w = w / (w.mean(dim, keepdim=True) + eps)

        mu = (x * w).mean(dim, keepdim=True) * scale

    return mu.squeeze(dim)
