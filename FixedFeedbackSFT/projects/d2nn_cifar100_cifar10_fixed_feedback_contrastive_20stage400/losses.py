from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """SupCon over [B,V,D] normalized embeddings with all same-class views positive."""
    if features.ndim != 3:
        raise ValueError(f"Expected [B,V,D] features, got {tuple(features.shape)}")
    batch, views, _ = features.shape
    if views < 2:
        raise ValueError("Supervised contrastive loss requires at least two views")
    flat = F.normalize(features.reshape(batch * views, -1), dim=-1, eps=1e-12)
    flat_labels = labels.reshape(-1).repeat_interleave(views)
    logits = flat @ flat.T / float(temperature)
    self_mask = torch.eye(len(flat), device=flat.device, dtype=torch.bool)
    positive_mask = flat_labels[:, None].eq(flat_labels[None, :]) & ~self_mask
    if torch.any(positive_mask.sum(dim=1) == 0):
        raise ValueError("Every SupCon anchor must have at least one positive")
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    loss = -(log_probability * positive_mask).sum(dim=1) / positive_mask.sum(dim=1)
    return loss.mean()


def leave_one_out_prototype_logits(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build each anchor's own-class prototype without using the anchor itself."""
    if features.ndim != 2:
        raise ValueError(f"Expected [N,D] features, got {tuple(features.shape)}")
    unique, inverse = torch.unique(labels.long(), sorted=True, return_inverse=True)
    classes = len(unique)
    sums = torch.zeros(classes, features.shape[1], device=features.device, dtype=features.dtype)
    sums.index_add_(0, inverse, features)
    counts = torch.bincount(inverse, minlength=classes).to(features.dtype)
    if torch.any(counts < 2):
        raise ValueError("Leave-one-out prototypes require at least two embeddings per class")
    prototype_sums = sums.unsqueeze(0).expand(len(features), -1, -1).clone()
    rows = torch.arange(len(features), device=features.device)
    prototype_sums[rows, inverse] -= features
    prototype_counts = counts.unsqueeze(0).expand(len(features), -1).clone()
    prototype_counts[rows, inverse] -= 1.0
    prototypes = F.normalize(prototype_sums / prototype_counts.unsqueeze(-1), dim=-1, eps=1e-12)
    logits = torch.einsum("nd,ncd->nc", F.normalize(features, dim=-1, eps=1e-12), prototypes)
    return logits / float(temperature), inverse


def contrastive_transfer_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    contrastive_temperature: float,
    prototype_temperature: float,
    supcon_weight: float,
    prototype_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    supcon = supervised_contrastive_loss(features, labels, contrastive_temperature)
    flat = features.reshape(-1, features.shape[-1])
    flat_labels = labels.repeat_interleave(features.shape[1])
    logits, local_targets = leave_one_out_prototype_logits(flat, flat_labels, prototype_temperature)
    prototype = F.cross_entropy(logits, local_targets)
    total = float(supcon_weight) * supcon + float(prototype_weight) * prototype
    accuracy = (logits.argmax(dim=1) == local_targets).float().mean()
    return total, {"supcon": supcon, "prototype": prototype, "batch_prototype_accuracy": accuracy}
