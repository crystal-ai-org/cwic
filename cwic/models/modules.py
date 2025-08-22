import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D
try:
    from cwic_triton.triton_torch import smm
except:
    smm=None
from utils.torch_utils import attach_gradient


logger = logging.get_logger(__name__)


class CWICLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stripe_size: int,
        bias: Optional[bool] = True,
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
        stripe_size = min(stripe_size, out_features)
        self.stripe_size = stripe_size
        
        self.bandwidth = bandwidth
        self.eps = eps

        if out_features % stripe_size == 0:
            self.og_out_features = out_features
            self.out_features = out_features

        else:
            self.og_out_features = out_features
            
            out_features = stripe_size * (1 + out_features // stripe_size)
            self.out_features = out_features

            logger.warning(
                f"`out_features` {self.og_out_features} is not divisible by `stripe_size`. {stripe_size} Adjusting `out_features` to be {out_features}"
            )
        self.num_stripes = self.out_features//self.stripe_size

        # note that this is transposed compared to nn.Linear for inference kernel compatibility
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) / (in_features ** 0.5)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features)
            )

        # when the argument threshold_lr_scale is 1.0, the thresholds move at the same 'speed' as the weights
        self.threshold_lr_scale = threshold_lr_scale * (in_features ** 0.5)
        self.threshold_minimum = threshold_minimum / self.threshold_lr_scale

        self.thresholds = nn.Parameter(
            torch.zeros(self.num_stripes, self.in_features) +
            (threshold_init / self.threshold_lr_scale)
        )

        self.dist_tracker = RobustDistributionTracker(
            self.in_features,
            beta=stats_beta,
            num_iters=median_iters,
            eps=eps
        )

        self.cached_weight = None
        self.cached_post_mu = None
        self.cached_thresholds = None



    def _get_weight(self) -> torch.FloatTensor:

        f = lambda x: x.T.view(1, self.num_stripes, self.stripe_size, self.in_features)

        if self.training:
            self.cached_weight = None
            return f(self.weight)
        
        else:
            # Use cached weight during inference if available
            if self.cached_weight is None:
                self.cached_weight = f(self.weight).detach().contiguous()

            return self.cached_weight

    def _get_post_mu(self) -> torch.FloatTensor:

        f = lambda x,y: torch.einsum("a b, a -> b",x,y)

        if self.training:
            self.cached_post_mu = None
            return f(self.weight,self.dist_tracker.adj_med_computed)
        
        else:
            # Use cached weight during inference if available
            if self.cached_post_mu is None:
                self.cached_post_mu = f(self.weight,self.dist_tracker.adj_med_computed).detach().contiguous()

            return self.cached_post_mu
        
    
    def _get_thresholds(self, std) -> torch.FloatTensor:

        f = lambda x, s: (x * s[None] *self.threshold_lr_scale)[None]

        if self.training:
            self.cached_thresholds = None
            return f(self.thresholds, std)
        
        else:
            # Use cached weight during inference if available
            if self.cached_thresholds is None:
                self.cached_thresholds = f(self.thresholds, std).detach().contiguous()

            return self.cached_thresholds


    def forward(
        self,
        x: torch.Tensor,
        statistics_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Shape annotations:

         - `B`: batch size
         - `N`: number of stripes
         - `S`: stripe size
         - `I`: input features
         - `O`: output features
        """
        
        mu, std = self.dist_tracker(
            x,
            statistics_mask=statistics_mask
        )

        # [B, 1, I]
        og_shape = x.shape[:-1]
        x = x.view(-1, 1, self.in_features)
        
        thresholds = self._get_thresholds(std) # [1, N, I]

        # [B, 1, I]
        x_demeaned = x - mu[None, None]  
        x_gate = x_demeaned.abs()
        
        # [B, N, I], [B, N, I]
        if self.training:

            bandwidth = (
                self.bandwidth
                * std[None, None]
            ) + self.eps # [1, 1, I]
            x_masked, mask = step_with_grads(
                x_demeaned,
                x_gate,
                thresholds,
                bandwidth
            )
                

            # [B, N, I, 1]
            x = (x_masked + mu[None, None]).unsqueeze(-1)

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

        else:
            mask = (
                x_gate > thresholds
            ).to(x.dtype)
            if math.prod(og_shape)==1 and smm is not None and self.stripe_size%16==0:
                y = smm(
                    x_demeaned,
                    self.weight,
                    thresholds,
                    stripe_size=self.stripe_size
                ).view(-1, self.out_features)
                y = y + self._get_post_mu()
            else:
                x = x_demeaned * mask

                y = torch.einsum(
                    "b n s i, b n i p -> b n s p",
                    self._get_weight(),
                    x.unsqueeze(-1),
                ).view(-1, self.out_features)
                y = y + self._get_post_mu()
            if self.bias is not None:
                y = y + self.bias[None]

            y = y.view(*og_shape, self.out_features)

            # correct the shape to the corrected output size
            if self.og_out_features != self.out_features:
                y = y[..., :self.og_out_features]

        # calculate the parameter usage
        active_params = self.stripe_size * mask.view(*og_shape, -1).sum(dim=-1)
        dense_params = self.stripe_size * torch.ones_like(mask).view(*og_shape, -1).sum(dim=-1)

        return y, (dense_params, active_params)


class CWICMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        inter_features: int,
        out_features: int,
        stripe_size: int,
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

        self.threshold_lr_scale = threshold_lr_scale * (in_features ** 0.5)
        self.threshold_minimum = threshold_minimum / self.threshold_lr_scale

        self.thresholds = nn.Parameter(
            torch.zeros(self.inter_features) +
            (threshold_init / self.threshold_lr_scale)
        )

        self.act_fn = ACT2FN[hidden_act]

        self.dist_tracker = RobustDistributionTracker(
            inter_features,
            beta=stats_beta,
            num_iters=median_iters,
            eps=eps
        )
        self.mad_tracker = RobustDistributionTracker(
            self.inter_features,
            beta=stats_beta,
            num_iters=median_iters,
            eps=eps
        )


    def forward(
        self,
        x: torch.Tensor,
        statistics_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        z, (gate_dense_params, gate_active_params) = self.gate(x, statistics_mask=statistics_mask)
        z = self.act_fn(z)

        std = self.dist_tracker(
            z,
            statistics_mask=statistics_mask
        )[1]
        mad = self.mad_tracker(
            z.abs(),
            statistics_mask=statistics_mask
        )[0]
        thresholds = (
            self.thresholds
            * self.threshold_lr_scale
            * mad
        ).view(*[1 for _ in range(x.ndim - 1)], -1)
        bandwidth = (
            self.bandwidth
            * std
        ).view(*[1 for _ in range(x.ndim - 1)], -1) + self.eps

        z_masked, mask = step_with_grads(
            z,
            z.abs(),
            thresholds,
            bandwidth
        )

        y = self.down(
            self.up(x) * z_masked
        )

        active_params = gate_active_params + (self.in_features + self.out_features) * mask.sum(dim=-1)
        dense_params = gate_dense_params + (self.in_features + self.out_features) * torch.ones_like(mask).sum(dim=-1)

        return y, (dense_params, active_params)


def step_with_grads(
    x: torch.Tensor,
    x_gate: torch.Tensor,
    thresholds: torch.Tensor,
    bandwidth: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    mask = (x_gate > thresholds).to(x.dtype)
    
    kernel = F.hardsigmoid(3 * (x_gate - thresholds) / bandwidth*2)
    nog_kernel = F.hardsigmoid(3 * (x_gate.detach() - thresholds) / bandwidth*2)

    mask = attach_gradient(kernel, mask)
    nog_mask = attach_gradient(nog_kernel, mask)

    out = attach_gradient(
        x, x.detach() * nog_mask,
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
        self.num_iters = num_iters
        self.eps = eps

        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
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
            "upp",
            torch.zeros((hidden_size,), dtype=torch.float32),
            persistent=True
        )
        self.register_buffer("adj_med_computed", torch.zeros((hidden_size,), dtype=torch.float32))
        self.register_buffer("adj_std_computed", torch.zeros((hidden_size,), dtype=torch.float32))



    def forward(
        self,
        x,
        statistics_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.detach()

        if self.training:
            with torch.no_grad():

                x = x.view(-1, self.hidden_size)
                if statistics_mask is not None:
                    statistics_mask = statistics_mask.view(-1, 1).to(x.dtype).detach()
                else:
                    statistics_mask = torch.ones_like(x[:, :1])
                current_step_size =statistics_mask.mean()
                sbeta=(self.beta**current_step_size)
                self.steps += current_step_size
                debiaser = 1 / (1 - self.beta ** self.steps)

                new_med = geometric_median(
                    x,
                    num_iters=self.num_iters,
                    dim=0,
                    mask=statistics_mask,
                    eps=self.eps,
                    verbose=False# (self.steps < 1.5).all()
                )
                old_std=self.upp-self.med
                self.med=sbeta * self.med + (1 - sbeta) * new_med
                
                med_debiased = self.med * debiaser

                new_std = (
                    ((x - med_debiased[None]).abs() * statistics_mask).mean(0) / statistics_mask.mean(0)
                ) / np.sqrt(2 * np.pi)
                self.upp=self.med+(sbeta * old_std + (1 - sbeta) * new_std)
                
                aad_debiased = (self.upp-self.med) * debiaser
                adj_med= med_debiased
                adj_std=aad_debiased
            
                self.adj_med_computed, self.adj_std_computed = adj_med, adj_std

        else:
            adj_med, adj_std = self.adj_med_computed, self.adj_std_computed
            

        return adj_med, adj_std



def geometric_median(
    x,
    num_iters,
    dim,
    mask=None,
    eps=1e-7,
    verbose=False
):
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
        w = w  * mask
        w = w / w.mean(dim, keepdim=True)

        mu = (x * w).mean(dim, keepdim=True)

    if verbose:
        print(f"Final Mu: {mu.squeeze(dim)}")

    return mu.squeeze(dim)
