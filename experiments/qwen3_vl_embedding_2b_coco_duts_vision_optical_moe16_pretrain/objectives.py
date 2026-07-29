from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.objectives import (
    SegmentationAccumulator,
    segmentation_loss,
)


def feature_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    cosine_weight: float,
    smooth_l1_weight: float,
    smooth_l1_beta: float,
    router_balance: torch.Tensor,
    router_balance_weight: float,
    router_importance: torch.Tensor | None = None,
    router_importance_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if student.shape != teacher.shape:
        raise RuntimeError(
            f"Student feature {tuple(student.shape)} != PCA teacher "
            f"{tuple(teacher.shape)}"
        )
    if student.ndim != 2 or student.shape[-1] != 224:
        raise RuntimeError(
            f"Feature distillation expects valid packed [tokens,224], got "
            f"{tuple(student.shape)}"
        )
    student32 = student.float()
    teacher32 = teacher.float()
    cosine = (
        1.0
        - F.cosine_similarity(student32, teacher32, dim=-1, eps=1e-8)
    ).mean()
    smooth_l1 = F.smooth_l1_loss(
        student32,
        teacher32,
        beta=float(smooth_l1_beta),
    )
    importance = (
        student32.new_zeros(())
        if router_importance is None
        else router_importance.float()
    )
    total = (
        float(cosine_weight) * cosine
        + float(smooth_l1_weight) * smooth_l1
        + float(router_balance_weight) * router_balance.float()
        + float(router_importance_weight) * importance
    )
    return total, {
        "cosine_loss": cosine,
        "smooth_l1_loss": smooth_l1,
        "router_balance": router_balance.float(),
        "router_importance": importance,
    }


class FeatureAccumulator:
    def __init__(self, *, smooth_l1_beta: float = 0.1) -> None:
        if smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.tokens = 0
        self.samples = 0
        self.loss_sum = 0.0
        self.cosine_sum = 0.0
        self.smooth_l1_sum = 0.0
        self.mse_sum = 0.0

    @torch.no_grad()
    def update(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        *,
        loss: float,
        samples: int,
    ) -> None:
        if student.shape != teacher.shape:
            raise RuntimeError("Feature accumulator received mismatched tensors")
        count = int(student.shape[0])
        student32 = student.float()
        teacher32 = teacher.float()
        self.tokens += count
        self.samples += int(samples)
        self.loss_sum += float(loss) * count
        self.cosine_sum += float(
            F.cosine_similarity(
                student32,
                teacher32,
                dim=-1,
                eps=1e-8,
            ).sum()
        )
        self.smooth_l1_sum += float(
            F.smooth_l1_loss(
                student32,
                teacher32,
                beta=self.smooth_l1_beta,
                reduction="none",
            ).mean(dim=-1).sum()
        )
        self.mse_sum += float(
            (student32 - teacher32).square().mean(dim=-1).sum()
        )

    def compute(self) -> dict[str, Any]:
        if not self.tokens:
            raise RuntimeError("Cannot compute feature metrics without valid tokens")
        return {
            "loss": self.loss_sum / self.tokens,
            "cosine_similarity": self.cosine_sum / self.tokens,
            "smooth_l1": self.smooth_l1_sum / self.tokens,
            "mse": self.mse_sum / self.tokens,
            "tokens": self.tokens,
            "samples": self.samples,
        }


__all__ = [
    "FeatureAccumulator",
    "SegmentationAccumulator",
    "feature_distillation_loss",
    "segmentation_loss",
]
