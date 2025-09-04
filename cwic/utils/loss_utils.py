from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_utils import attach_gradient


class KDLossModule(nn.Module):

    def __init__(
        self,
        chunk_size: Optional[int] = None
    ):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return kd_loss_fn(student_logits, teacher_logits, mask, self.chunk_size)


def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    if chunk_size is None:
        return _kd_loss_fn(
            student_logits.view(-1, student_logits.shape[-1]),
            teacher_logits.view(-1, teacher_logits.shape[-1]),
            mask.view(-1) if mask is not None else None,
        )
    
    bs = student_logits.shape[0]

    num_chunks = bs // chunk_size
    assert bs % chunk_size == 0, f"Batch size {bs} must be divisible by chunk_size {chunk_size}"

    loss = 0
    rkl = 0
    fkl = 0
    for s_logits, t_logits, m in zip(
        torch.chunk(student_logits, num_chunks, dim=0),
        torch.chunk(teacher_logits, num_chunks, dim=0),
        torch.chunk(mask, num_chunks, dim=0) if mask is not None else [None]*num_chunks,
    ):

        curr_loss, curr_rkl, curr_fkl = torch.utils.checkpoint.checkpoint(
            _kd_loss_fn,
            s_logits.view(-1, s_logits.shape[-1]),
            t_logits.view(-1, t_logits.shape[-1]),
            m.view(-1) if m is not None else None,
            use_reentrant=False,
        )

        loss = loss + curr_loss
        rkl = rkl + curr_rkl
        fkl = fkl + curr_fkl

    return loss / num_chunks, rkl / num_chunks, fkl / num_chunks


def _kd_loss_fn(student_logits, teacher_logits, mask=None):
    teacher_logits = teacher_logits.to(student_logits.dtype)
    mask = mask.to(student_logits.dtype) if mask is not None else None

    if mask is not None:
        student_logits = student_logits * mask[..., None]
        teacher_logits = teacher_logits * mask[..., None]

    student_logits = F.log_softmax(student_logits, dim=-1)
    teacher_logits = F.log_softmax(teacher_logits, dim=-1)

    rkl = F.kl_div(
        input=student_logits,
        target=teacher_logits,
        log_target=True,
        reduction="batchmean",
    )
    fkl = F.kl_div(
        input=teacher_logits,
        target=student_logits,
        log_target=True,
        reduction="batchmean",
    )

    loss = (rkl + fkl) / 2

    if mask is not None:
        return loss / mask.mean(), rkl / mask.mean(), fkl / mask.mean()
    else:
        return loss, rkl, fkl


class MSELossModule(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return scaled_mse_fn(pred, target, mask)


def scaled_mse_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    pred = pred.view(-1, pred.shape[-1])
    target = target.view(-1, target.shape[-1]).to(pred.dtype)
    if mask is not None:
        mask = mask.view(-1, 1).to(pred.dtype)
    else:
        mask = torch.ones_like(pred[..., :1])

    mu = (
        (target * mask).mean(0)
        / mask.mean(0)
    )
    std = torch.sqrt(
        ((target - mu).pow(2) * mask).mean(0)
        / mask.mean(0)
    )

    pred = (pred - mu[None]) / std[None]
    target = (target - mu[None]) / std[None]

    return (
        ((pred - target).pow(2) * mask).mean()
        / mask.mean()
    )


class FlopLossModule(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(
        self,
        active_params: torch.Tensor,
        dense_params: torch.Tensor,
        target_ratio: float = 2.0,
        mask: Optional[torch.Tensor] = None,
        base_ratio: Optional[torch.Tensor] = None,
    ):
        return flop_loss_fn(
            active_params,
            dense_params,
            target_ratio,
            mask=mask,
            base_ratio=base_ratio
        )


def flop_loss_fn(
    active_params: torch.Tensor,
    dense_params: torch.Tensor,
    target_ratio: float = 2.0,
    mask: Optional[torch.Tensor] = None,
    base_ratio: Optional[torch.Tensor] = None,
):

    if mask is not None:
        active_per_token = (active_params * mask).sum() / mask.sum()
        dense_per_token = (dense_params * mask).sum() / mask.sum()
    else:
        active_per_token = active_params.mean()
        dense_per_token = dense_params.mean()

    ratio = dense_per_token / active_per_token
    if base_ratio is not None:
        ratio = attach_gradient(base_ratio.detach(), ratio)

    return torch.clip(ratio - target_ratio, max=0) ** 2


def get_total_active(
    active_params: torch.Tensor,
    dense_params: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    
    if mask is not None:
        active = (active_params * mask).sum()
        dense = (dense_params * mask).sum()
    else:
        active = active_params.sum()
        dense = dense_params.sum()

    return active, dense
