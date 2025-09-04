import torch
import torch.nn as nn
import torch.nn.functional as F


def unsqueeze_to_batch(x, target):
    while x.dim() < target.dim():
        x = x[None]

    return x


def expand_to_batch(x, target):

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[None]
        num_unsqueeze += 1

    x = x.repeat(
        *(
            [target.shape[i] for i in range(num_unsqueeze)] +
            [1] * (x.dim() - num_unsqueeze)
        )
    )

    return x


class _ScaleGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def scale_gradient(x, scale):
    return _ScaleGradient.apply(x, scale)


class _DebugGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output)
        return grad_output


def debug_gradient(x):
    return _DebugGradient.apply(x)


class _AttachGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, val, att):
        return val.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

def attach_gradient(val: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
    att = att.expand(*val.shape)
    return _AttachGradient.apply(val, att)


def grad_nan_to_num(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = torch.nan_to_num(p.grad, 0.0, 0.0, 0.0)
            