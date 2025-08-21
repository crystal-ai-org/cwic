from functools import partial
import math
import timeit
import torch
import triton
import triton.language as tl


from typing import Optional

import triton
import triton.language as tl


def init_to_zero(*names):
    def init_func(nargs):
        for name in names:
            nargs[name].zero_()

    return init_func


# NOTE: will need to warm up kernels each time, triton autotune caching isn't a thing right now

configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    # Llama 3 variants can use BLOCK_N >= 1024
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
]
kernels={}
def get_sparse_gemv_kernel_for_stripe_size(stripe_size:int):
    global kernels
    if stripe_size in kernels:
        return kernels[stripe_size]
    @triton.autotune(
        configs=[config for config in configs if stripe_size%config.kwargs["BLOCK_N"]==0],
        key=["BATCHSIZE"],
    )
    @triton.jit
    def sparse_gemv_kernel(
        X,
        A,
        threshold,

        Y,
        # Matrix dimensions
        N: tl.constexpr,
        M: tl.constexpr,
        Sn: tl.constexpr,
        # Meta-parameters
        BATCHSIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        start_n = tl.program_id(0)
        start_m = tl.program_id(1)
        rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        A_ptr = A + (rm[:, None] * N + rn[None, :])
        X_ptr = X + rm
        T_ptr = threshold + rm + M * ((start_n * BLOCK_N * Sn) // N)

        Y_ptr = Y + rn

        if BATCHSIZE == 1:
            thresh0 = tl.load(
                T_ptr, mask=rm < M, other=10.0, eviction_policy="evict_first"
            )  # no reuse thesh across threadblocks
            x0 = tl.load(
                X_ptr, mask=rm < M, other=0.0, eviction_policy="evict_last"
            )  # reuse x across threadblocks
            
            idx = tl.abs(x0) > thresh0 
            # selectively load weight rows
            a = tl.load(
                A_ptr, mask=idx[:, None], other=0.0, eviction_policy="evict_first"
            )  # only load weights once per threadblock
            acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

            # rematerialize rm and rn to save registers
            rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

            tl.atomic_add(Y_ptr, acc0, mask=rn < N)
    kernels[stripe_size]=sparse_gemv_kernel
    return sparse_gemv_kernel




def smm(x: torch.Tensor, w: torch.Tensor, t: torch.Tensor, 
        stripe_size: int = 1024) -> torch.Tensor:
    """
    Sparse matrix multiplication using Triton kernel.
    
    Args:
        x: Input tensor of shape (..., M)
        w: Weight matrix of shape (M, N) 
        t: Threshold tensor of shape (..., M)
        block_size_n: Block size for N dimension
        block_size_m: Block size for M dimension
        stripe_size: Stripe size for processing
        num_warps: Number of warps for Triton kernel
        
    Returns:
        Output tensor of shape (..., N)
    """
    # Ensure input is 2D for processing
    original_shape = x.shape
    assert math.prod(x.shape[:-1])==1
    x = x.view(-1, x.shape[-1])
    t = t.view(-1, t.shape[-1])
    
    batch_size = x.shape[0] if x.dim() == 2 else 1
    M = x.shape[-1]  # Input feature dimension
    _, N = w.shape   # Output feature dimension
    
    # Create output tensor
    if x.dim() == 1:
        out_shape = (N,)
    else:
        out_shape = (batch_size, N)
    
    output = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
    
    # Calculate grid dimensions
    def grid(meta):
        return (
            triton.cdiv(N, meta['BLOCK_N']),
            triton.cdiv(M, meta['BLOCK_M']),
        )
    
    
    stride_w_m, stride_w_n = w.stride(0), w.stride(1)
    
    # Launch kernel
    get_sparse_gemv_kernel_for_stripe_size(stripe_size)[grid](
        x, w, t, output,
        N=N, M=M, Sn=(N + stripe_size - 1) // stripe_size, BATCHSIZE=batch_size,
    )
    
    # Reshape output to match expected shape
    if len(original_shape) > 1:
        output = output.view(original_shape[:-1] + (N,))
    
    return output

# Example usage:
if __name__ == "__main__":
    # Test the function
    M, N = 512,2048
    stripe_size=1024
    x = torch.randn((1,M), device='cuda', dtype=torch.float32)
    w = torch.randn(M, N, device='cuda', dtype=torch.float32)/M**0.5
    t = torch.zeros((N//stripe_size,M), device='cuda', dtype=torch.float32)+0.5
    
    result = smm(x, w, t,stripe_size=stripe_size)
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {w.shape}")
    print(f"Output shape: {result.shape}")
