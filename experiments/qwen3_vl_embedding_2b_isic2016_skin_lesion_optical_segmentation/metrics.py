from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.objectives import (
    SegmentationAccumulator,
)


class ISICSegmentationAccumulator:
    """Challenge-oriented metrics plus the repository's common metrics."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.common = SegmentationAccumulator(threshold=threshold)
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.sample_sensitivity: list[float] = []
        self.sample_specificity: list[float] = []

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        loss: float | torch.Tensor | None = None,
    ) -> None:
        self.common.update(logits, target, loss=loss)
        prediction = logits.float().sigmoid().ge(self.threshold)
        truth = target.float().ge(0.5)
        tp = (prediction & truth).flatten(1).sum(1)
        tn = ((~prediction) & (~truth)).flatten(1).sum(1)
        fp = (prediction & (~truth)).flatten(1).sum(1)
        fn = ((~prediction) & truth).flatten(1).sum(1)
        self.true_positive += int(tp.sum())
        self.true_negative += int(tn.sum())
        self.false_positive += int(fp.sum())
        self.false_negative += int(fn.sum())
        sensitivity = tp.float() / (tp + fn).clamp_min(1).float()
        specificity = tn.float() / (tn + fp).clamp_min(1).float()
        self.sample_sensitivity.extend(sensitivity.cpu().tolist())
        self.sample_specificity.extend(specificity.cpu().tolist())

    def compute(self) -> dict[str, Any]:
        result = dict(self.common.compute())
        sensitivity = self.true_positive / max(
            1, self.true_positive + self.false_negative
        )
        specificity = self.true_negative / max(
            1, self.true_negative + self.false_positive
        )
        result.update(
            {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_pixel_accuracy": 0.5 * (sensitivity + specificity),
                "mean_sample_sensitivity": float(
                    np.mean(self.sample_sensitivity)
                ),
                "mean_sample_specificity": float(
                    np.mean(self.sample_specificity)
                ),
                "true_positive_pixels": self.true_positive,
                "true_negative_pixels": self.true_negative,
                "false_positive_pixels": self.false_positive,
                "false_negative_pixels": self.false_negative,
                "threshold": self.threshold,
            }
        )
        return result


def per_sample_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> list[dict[str, float]]:
    probability = logits.float().sigmoid()
    prediction = probability.ge(threshold)
    truth = target.float().ge(0.5)
    tp = (prediction & truth).flatten(1).sum(1).float()
    tn = ((~prediction) & (~truth)).flatten(1).sum(1).float()
    fp = (prediction & (~truth)).flatten(1).sum(1).float()
    fn = ((~prediction) & truth).flatten(1).sum(1).float()
    union = tp + fp + fn
    iou = torch.where(union > 0, tp / union, torch.ones_like(union))
    dice_denominator = 2.0 * tp + fp + fn
    dice = torch.where(
        dice_denominator > 0,
        2.0 * tp / dice_denominator,
        torch.ones_like(dice_denominator),
    )
    sensitivity = tp / (tp + fn).clamp_min(1)
    specificity = tn / (tn + fp).clamp_min(1)
    mae = (probability - target.float()).abs().flatten(1).mean(1)
    return [
        {
            "mean_iou": float(iou[index]),
            "mean_dice": float(dice[index]),
            "mae": float(mae[index]),
            "sensitivity": float(sensitivity[index]),
            "specificity": float(specificity[index]),
        }
        for index in range(logits.shape[0])
    ]


__all__ = ["ISICSegmentationAccumulator", "per_sample_metrics"]
