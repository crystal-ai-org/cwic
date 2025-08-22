from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

        
def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bs = student_logits.shape[0]

    loss = 0
    for i in range(bs):
        
        curr_loss = torch.utils.checkpoint.checkpoint(
            _kd_loss_fn,
            student_logits[i],
            teacher_logits[i],
            (mask[i] if mask is not None else None),
            use_reentrant=False,
        )
        loss = loss + curr_loss

    return loss / bs


def _kd_loss_fn(
    student_logits,
    teacher_logits,
    mask=None
):
    teacher_logits = teacher_logits.to(student_logits.dtype)

    if mask is not None:
        student_logits = student_logits * mask[..., None]
        teacher_logits = teacher_logits * mask[..., None]

    student_logits = F.log_softmax(student_logits, dim=-1)
    teacher_logits = F.log_softmax(teacher_logits, dim=-1)

    rkl = F.kl_div(
        input=student_logits,
        target=teacher_logits,
        log_target=True,
    )
    fkl = F.kl_div(
        input=teacher_logits,
        target=student_logits,
        log_target=True,
    )

    loss = (rkl + fkl) / 2
    
    if mask is not None:
        return loss / mask.mean()
    else:
        return loss


def flop_loss_fn(
    active_params: torch.Tensor,
    dense_params: torch.Tensor,
    target_ratio: float = 2.0,
    mask: Optional[torch.Tensor] = None,
):
    
    if mask is not None:
        active_per_token = (active_params * mask).sum() / mask.sum()
        dense_per_token = (dense_params * mask).sum() / mask.sum()
    else:
        active_per_token = active_params.mean()
        dense_per_token = dense_params.mean()

    ratio = dense_per_token / active_per_token

    return torch.clip(ratio - target_ratio, max=0) ** 2, ratio
