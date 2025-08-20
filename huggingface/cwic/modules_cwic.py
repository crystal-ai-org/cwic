import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D


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

        self.mu = nn.Parameter(
            torch.zeros(in_features)
        )
        self.std = nn.Parameter(
            torch.ones(in_features)
        )

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
            return f(self.weight,self.mu)
        
        else:
            # Use cached weight during inference if available
            if self.cached_post_mu is None:
                self.cached_post_mu = f(self.weight,self.mu).detach().contiguous()

            return self.cached_post_mu
        
    
    def _get_thresholds(self) -> torch.FloatTensor:

        f = lambda x, s: (x * s[None])[None]

        if torch.is_grad_enabled():
            self.cached_thresholds = None
            return f(self.thresholds, self.std)
        
        else:
            # Use cached weight during inference if available
            if self.cached_thresholds is None:
                self.cached_thresholds = f(self.thresholds, self.std).detach().contiguous()

            return self.cached_thresholds


    def forward(
        self,
        x: torch.FloatTensor
    ) -> tuple[torch.FloatTensor,tuple[torch.FloatTensor,torch.FloatTensor]]:
        og_shape = x.shape[:-1]

        x = x.view(-1, 1, self.in_features) - self.mu[None, None, :]
        mask = (
            x.abs() > self._get_thresholds()
        ).to(x.dtype)
        if torch.is_grad_enabled():
            x = (x * mask) + self.mu[None, None, :]

            y = torch.einsum(
                "b n s i, b n i p -> b n s p",
                self._get_weight(),
                x.unsqueeze(-1),
            ).view(-1, self.out_features)
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

        mask = (
            z.abs() > self.thresholds[None]
        ).to(z.dtype)

        y = self.down(
            self.up(x) * (z * mask)
        ).view(*og_shape, -1)

        return y, (dense_parameters + (self.in_features + self.out_features) * self.inter_features,active_parameters + (self.in_features + self.out_features) * self.inter_features * mask.mean(-1).view(*og_shape))
