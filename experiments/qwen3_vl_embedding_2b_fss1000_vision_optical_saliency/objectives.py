from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = logits.float().sigmoid()
    target = target.float()
    intersection = (probabilities * target).flatten(1).sum(1)
    denominator = probabilities.flatten(1).sum(1) + target.flatten(1).sum(1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def soft_iou_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probability = logits.float().sigmoid()
    target = target.float()
    intersection = (probability * target).flatten(1).sum(1)
    union = (
        probability + target - probability * target
    ).flatten(1).sum(1)
    return (1.0 - (intersection + eps) / (union + eps)).mean()


def boundary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dice loss on 3x3 morphological boundaries.

    The operation remains differentiable for prediction probabilities and
    directly emphasizes the thin/complex contours that dominate the observed
    FSS-1000 student failures.
    """
    probability = logits.float().sigmoid()
    target = target.float()
    predicted_boundary = (
        F.max_pool2d(probability, 3, stride=1, padding=1)
        + F.max_pool2d(-probability, 3, stride=1, padding=1)
    )
    target_boundary = (
        F.max_pool2d(target, 3, stride=1, padding=1)
        + F.max_pool2d(-target, 3, stride=1, padding=1)
    )
    intersection = (predicted_boundary * target_boundary).flatten(1).sum(1)
    denominator = (
        predicted_boundary.flatten(1).sum(1)
        + target_boundary.flatten(1).sum(1)
    )
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    bce_weight: float,
    dice_weight: float,
    soft_iou_weight: float = 0.0,
    boundary_weight: float = 0.0,
    teacher_logits: torch.Tensor | None = None,
    mask_kd_weight: float = 0.0,
    mask_kd_temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.shape != target.shape:
        raise RuntimeError(f"Logits {tuple(logits.shape)} != masks {tuple(target.shape)}")
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
    dice = dice_loss(logits, target)
    soft_iou = soft_iou_loss(logits, target)
    boundary = boundary_dice_loss(logits, target)
    total = (
        bce_weight * bce
        + dice_weight * dice
        + soft_iou_weight * soft_iou
        + boundary_weight * boundary
    )
    kd = logits.new_zeros(())
    if mask_kd_weight > 0:
        if teacher_logits is None or teacher_logits.shape != logits.shape:
            raise RuntimeError("Mask KD is enabled but shape-compatible teacher logits are absent")
        temperature = float(mask_kd_temperature)
        teacher_probability = (teacher_logits.float() / temperature).sigmoid()
        kd = F.binary_cross_entropy_with_logits(
            logits.float() / temperature, teacher_probability
        ) * temperature * temperature
        total = total + mask_kd_weight * kd
    return total, {
        "bce": bce,
        "dice_loss": dice,
        "soft_iou_loss": soft_iou,
        "boundary_loss": boundary,
        "mask_kd": kd,
    }


class SegmentationAccumulator:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.iou: list[float] = []
        self.dice: list[float] = []
        self.absolute_error_sum = 0.0
        self.pixel_correct = 0
        self.pixel_count = 0
        self.loss_sum = 0.0
        self.sample_count = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        loss: float | torch.Tensor | None = None,
    ) -> None:
        probability = logits.float().sigmoid()
        truth = target.float()
        prediction = probability.ge(self.threshold)
        binary_truth = truth.ge(0.5)
        intersection = (prediction & binary_truth).flatten(1).sum(1).float()
        union = (prediction | binary_truth).flatten(1).sum(1).float()
        pred_sum = prediction.flatten(1).sum(1).float()
        truth_sum = binary_truth.flatten(1).sum(1).float()
        iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
        dice = torch.where(
            pred_sum + truth_sum > 0,
            2.0 * intersection / (pred_sum + truth_sum),
            torch.ones_like(intersection),
        )
        self.iou.extend(iou.cpu().tolist())
        self.dice.extend(dice.cpu().tolist())
        self.absolute_error_sum += float((probability - truth).abs().sum().item())
        self.pixel_correct += int(prediction.eq(binary_truth).sum().item())
        self.pixel_count += int(target.numel())
        batch = int(target.shape[0])
        self.sample_count += batch
        if loss is not None:
            self.loss_sum += float(loss) * batch

    def compute(self) -> dict[str, float | int]:
        if not self.sample_count:
            raise RuntimeError("Cannot compute segmentation metrics without samples")
        return {
            "loss": self.loss_sum / self.sample_count,
            "mean_iou": float(np.mean(self.iou)),
            "mean_dice": float(np.mean(self.dice)),
            "mean_f1": float(np.mean(self.dice)),
            "mae": self.absolute_error_sum / self.pixel_count,
            "pixel_accuracy": self.pixel_correct / self.pixel_count,
            "samples": self.sample_count,
            "pixels": self.pixel_count,
        }
