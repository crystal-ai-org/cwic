import torch
import torch.nn as nn
import torch.nn.functional as F


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


# class _AttachGradient(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, tgt, src):
#         out = src.clone()
#         out.requires_grad = True
#         return out
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, grad_output

# def attach_gradient(tgt, src):
#     return _AttachGradient.apply(tgt, src)

def attach_gradient(tgt, src):
    return src + (tgt - tgt.detach())
