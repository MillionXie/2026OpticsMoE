from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .settings import Settings


def soft_dice_loss(probability: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def editing_objective(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    settings: Settings,
) -> dict[str, torch.Tensor]:
    palette_logits = outputs["palette_logits"].float()
    edit_logits = outputs["edit_logits"].float()
    target_classes = batch["target_classes"].long()
    edit_target = batch["edit_mask"].float()
    preserve_target = batch["preserve_mask"].float()
    task_target = batch["task_index"].long()

    pixel_ce = F.cross_entropy(palette_logits, target_classes, reduction="none")
    pixel_weight = 1.0 + settings.changed_pixel_weight * edit_target
    palette_loss = (pixel_ce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)

    edit_probability = torch.sigmoid(edit_logits)
    mask_bce = F.binary_cross_entropy_with_logits(edit_logits, edit_target)
    mask_dice = soft_dice_loss(edit_probability, edit_target)
    edit_mask_loss = mask_bce + mask_dice

    preservation_loss = (
        (edit_probability * preserve_target).sum()
        / preserve_target.sum().clamp_min(1.0)
    )
    task_loss = F.cross_entropy(outputs["task_logits"].float(), task_target)

    edge_rows = task_target.eq(3)
    if edge_rows.any():
        black_probability = palette_logits[edge_rows].softmax(dim=1)[:, 7]
        black_target = target_classes[edge_rows].eq(7).float()
        edge_loss = soft_dice_loss(black_probability, black_target)
    else:
        edge_loss = palette_loss.new_zeros(())

    total = (
        settings.palette_loss_weight * palette_loss
        + settings.edit_mask_loss_weight * edit_mask_loss
        + settings.preservation_loss_weight * preservation_loss
        + settings.task_loss_weight * task_loss
        + settings.edge_loss_weight * edge_loss
        + settings.ccd_loss_weight * outputs["ccd_operating_loss"].float()
        + settings.router_balance_weight * outputs["router_balance_loss"].float()
    )
    return {
        "total": total,
        "palette": palette_loss,
        "edit_mask": edit_mask_loss,
        "preservation": preservation_loss,
        "task": task_loss,
        "edge": edge_loss,
        "ccd": outputs["ccd_operating_loss"].float(),
        "router_balance": outputs["router_balance_loss"].float(),
    }


__all__ = ["editing_objective", "soft_dice_loss"]
