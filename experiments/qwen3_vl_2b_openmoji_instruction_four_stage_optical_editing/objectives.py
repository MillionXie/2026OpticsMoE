from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .settings import Settings


def dice_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)).mean()


def editing_objective(
    outputs: dict[str, torch.Tensor], batch: dict[str, Any], settings: Settings
) -> dict[str, torch.Tensor]:
    category_logits = outputs["category_logits"].float()
    edit_logits = outputs["edit_logits"].float()
    target = batch["target_grid"].long()
    edit = batch["edit_grid"].float()
    preserve = batch["preserve_grid"].float()
    cell_ce = F.cross_entropy(category_logits, target, reduction="none")
    weight = (
        1.0
        + settings.changed_cell_weight * edit
        + settings.foreground_cell_weight * target.gt(0).float()
    )
    category_loss = (cell_ce * weight).sum() / weight.sum().clamp_min(1.0)
    edit_probability = torch.sigmoid(edit_logits)
    edit_bce = F.binary_cross_entropy_with_logits(
        edit_logits,
        edit,
        pos_weight=edit_logits.new_tensor(8.0),
    )
    edit_loss = edit_bce + dice_loss(edit_probability, edit)
    preservation_loss = (edit_probability * preserve).sum() / preserve.sum().clamp_min(1.0)
    task_loss = F.cross_entropy(outputs["task_logits"].float(), batch["task_index"].long())
    total = (
        settings.category_loss_weight * category_loss
        + settings.edit_loss_weight * edit_loss
        + settings.preservation_loss_weight * preservation_loss
        + settings.task_loss_weight * task_loss
        + settings.ccd_loss_weight * outputs["ccd_operating_loss"].float()
        + settings.router_balance_weight * outputs["router_balance_loss"].float()
    )
    return {
        "total": total,
        "category": category_loss,
        "edit": edit_loss,
        "preservation": preservation_loss,
        "task": task_loss,
        "ccd": outputs["ccd_operating_loss"].float(),
        "router_balance": outputs["router_balance_loss"].float(),
    }


__all__ = ["dice_loss", "editing_objective"]

