import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D
from cwic_triton.triton_torch import smm


class DistributionTracker(nn.Module):

    def __init__(
        self,
        hidden_size,
        quantile_bs: int = 4,
        upper_quantile: float = 0.841,
        beta: float = 0.99,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.quantile_bs = quantile_bs
        self.upper_quantile = upper_quantile

        self.register_buffer("beta",torch.tensor(beta, dtype=torch.float32))
        self.register_buffer("steps",torch.zeros((), dtype=torch.float32))

        self.register_buffer("med",torch.zeros((hidden_size,), dtype=torch.float32))
        self.register_buffer("upp",torch.zeros((hidden_size,), dtype=torch.float32))

        self.register_buffer("adj_med_computed",torch.zeros((hidden_size,), dtype=torch.float32))
        self.register_buffer("adj_std_computed",torch.zeros((hidden_size,), dtype=torch.float32))

    def __call__(
        self,
        x,
    ):
        if self.training:
            # broadcast where_vals to x shape
            assert x.shape[-1] == self.hidden_size

            # reduce batch size
            x = x[: self.quantile_bs]

            # reshape batched vectors
            x = x.reshape(-1, self.hidden_size).float()

            # get x distribution
            new_med = torch.median(x, 0).values.detach()
            new_upp = torch.quantile(x, self.upper_quantile, 0).detach()

            # calculate update size
            delta = 1.0
            beta = self.beta**delta

            # calculate new values
            med = beta * self.med + (1.0 - beta) * new_med
            upp = beta * self.upp + (1.0 - beta) * new_upp

            old_med = self.med.clone()
            old_upp = self.upp.clone()
            old_steps = self.steps.clone()

            self.med = med
            self.upp =upp
            self.steps =self.steps + delta

            trig = old_steps > 1.0
            med = torch.where(trig, old_med, med)
            upp = torch.where(trig, old_upp, upp)
            steps = torch.where(trig, old_steps, self.steps)

            # adjust for bias
            div = 1 - self.beta**steps
            adj_med = med / (div + 1e-7)
            adj_upp = upp / (div + 1e-7)
            adj_med, adj_std = (
                (adj_med),
                (adj_upp - adj_med +1e-7),
            )
            self.adj_med_computed, self.adj_std_computed = adj_med, adj_std

        else:
            adj_med, adj_std = self.adj_med_computed, self.adj_std_computed

        return adj_med, adj_std
    
class CWICLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stripe_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.stripe_size = stripe_size
        self.num_stripes = out_features // stripe_size
        if out_features % stripe_size != 0:
            raise ValueError(f"out_features {out_features} must be divisible by stripe_size {stripe_size}")

        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) / (in_features ** 0.5)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features)
            )

        self.dist_tracker = DistributionTracker(in_features)

        self.thresholds = nn.Parameter(
            torch.zeros(self.num_stripes, self.in_features)
        )

        self.cached_weight = None
        self.cached_post_mu = None
        self.cached_thresholds = None


    def _get_weight(self) -> torch.FloatTensor:

        f = lambda x: x.T.view(1, self.num_stripes, self.stripe_size, self.in_features)

        if torch.is_grad_enabled():
            self.cached_weight = None
            return f(self.weight)
        
        else:
            # Use cached weight during inference if available
            if self.cached_weight is None:
                self.cached_weight = f(self.weight).detach().contiguous()

            return self.cached_weight

    def _get_post_mu(self) -> torch.FloatTensor:

        f = lambda x,y: torch.einsum("a b, a -> b",x,y)

        if torch.is_grad_enabled():
            self.cached_post_mu = None
            return f(self.weight,self.dist_tracker.adj_med_computed)
        
        else:
            # Use cached weight during inference if available
            if self.cached_post_mu is None:
                self.cached_post_mu = f(self.weight,self.dist_tracker.adj_med_computed).detach().contiguous()

            return self.cached_post_mu
        
    
    def _get_thresholds(self, std) -> torch.FloatTensor:

        f = lambda x, s: (x * s[None])[None]

        if torch.is_grad_enabled():
            self.cached_thresholds = None
            return f(self.thresholds, std)
        
        else:
            # Use cached weight during inference if available
            if self.cached_thresholds is None:
                self.cached_thresholds = f(self.thresholds, std).detach().contiguous()

            return self.cached_thresholds


    def forward(
        self,
        x: torch.FloatTensor
    ) -> tuple[torch.FloatTensor,tuple[torch.FloatTensor,torch.FloatTensor]]:
        og_shape = x.shape[:-1]
        mu, std = self.dist_tracker(x)
        x = x.view(-1, 1, self.in_features) - mu[None, None, :]
        xgate = x.abs()
        thresh = self._get_thresholds(std)
        mask = (
            xgate > thresh
        ).to(x.dtype)
        if self.training:
            bw=std[None, None, :]*0.1

            kernel = F.hardsigmoid(3*(xgate-thresh)/bw)
            mask = mask + (kernel -kernel.detach())
            nog_kernel = F.hardsigmoid(3*(xgate.detach()-thresh)/bw)
            grad_term = x + x.detach() * nog_kernel
            x = (x * mask).detach() + (grad_term -grad_term.detach())
            x = x + mu[None, None, :]

            y = torch.einsum(
                "b n s i, b n i p -> b n s p",
                self._get_weight(),
                x.unsqueeze(-1),
            ).view(-1, self.out_features)
        else:
            if math.prod(og_shape)==1:
                y = smm(
                    x,
                    self.weight,
                    thresh,
                    stripe_size=self.stripe_size
                ).view(-1, self.out_features)
                y = y + self._get_post_mu()
            else:
                x = x * mask

                y = torch.einsum(
                    "b n s i, b n i p -> b n s p",
                    self._get_weight(),
                    x.unsqueeze(-1),
                ).view(-1, self.out_features)
                y = y + self._get_post_mu()


        if self.bias is not None:
            y = y + self.bias[None]

        return y.view(*og_shape, self.out_features), (torch.zeros(og_shape,device=y.device,dtype=torch.float32)+self.in_features*self.out_features,torch.zeros(og_shape,device=y.device,dtype=torch.float32)+self.in_features*self.out_features*mask.mean(-1).mean(-1).view(*og_shape))


class CWICMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        inter_features: int,
        out_features: int,
        stripe_size: int,
        hidden_act: str = "silu",
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.inter_features = inter_features
        self.out_features = out_features

        self.gate = CWICLinear(
            in_features=in_features,
            out_features=inter_features,
            stripe_size=stripe_size,
            bias=bias
        )

        self.up = Conv1D(
            inter_features,
            in_features
        )
        assert not bias, "CWICMLP does not support bias due to the use of Conv1D"
        
        self.down = nn.Linear(
            inter_features,
            out_features,
            bias
        )

        self.dist_tracker = DistributionTracker(inter_features)
        self.mad_tracker = DistributionTracker(inter_features)

        self.thresholds = nn.Parameter(
            torch.zeros(inter_features)
        )
        
        self.act_fn = ACT2FN[hidden_act]


    def forward(
        self,
        x: torch.FloatTensor
    ) -> torch.FloatTensor:
        og_shape = x.shape[:-1]

        x = x.view(-1, x.shape[-1])
        z, (dense_parameters, active_parameters) = self.gate(x)
        z = self.act_fn(z)

        zgate = z.abs()

        _, std = self.dist_tracker(z)

        mad, _ = self.mad_tracker(zgate)

        thresh = self.thresholds[None] * mad
        mask = (
            zgate > thresh
        ).to(z.dtype)

        if self.training:
            bw=std[None, None, :]*0.1

            kernel = F.hardsigmoid(3*(zgate-thresh)/bw)
            mask = mask + (kernel - kernel.detach())
            nog_kernel = F.hardsigmoid(3*(zgate.detach()-thresh)/bw)
            grad_term = z + z.detach() * nog_kernel
            z = (z * mask).detach() + (grad_term -grad_term.detach())
        else:
            z = z * mask


        y = self.down(
            self.up(x) * z
        ).view(*og_shape, -1)

        return y, (dense_parameters + (self.in_features + self.out_features) * self.inter_features,active_parameters + (self.in_features + self.out_features) * self.inter_features * mask.mean(-1).view(*og_shape))
