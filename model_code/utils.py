from typing import Tuple
from jax import Array
import jax


@jax.custom_vjp
def replace_grad(out: Array, to_grad: Array):
    return out.astype(to_grad.dtype)


def replace_grad_fwd(out: Array, to_grad: Array):
    return out.astype(to_grad.dtype), ()


def replace_grad_bwd(res: Tuple, g: Array):
    return None, g


replace_grad.defvjp(replace_grad_fwd, replace_grad_bwd)


def attach_grad(out, to_grad):
    return replace_grad(out, out + to_grad)
