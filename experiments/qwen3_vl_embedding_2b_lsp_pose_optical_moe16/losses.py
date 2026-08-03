from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_heatmap_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise RuntimeError(
            f"Heatmap shapes must match [B,J,H,W], got {prediction.shape} and {target.shape}"
        )
    if visible.shape != prediction.shape[:2]:
        raise RuntimeError("Visibility mask must be [B,J]")
    weights = visible.to(prediction.dtype)[..., None, None]
    denominator = weights.sum().clamp_min(1.0) * prediction.shape[-1] * prediction.shape[-2]
    return ((prediction.float() - target.float()).square() * weights).sum() / denominator


def softargmax_coordinates(heatmaps: torch.Tensor, image_size: int) -> torch.Tensor:
    """Differentiable [B,J,H,W] heatmaps -> [B,J,2] image-space coordinates."""
    if heatmaps.ndim != 4:
        raise RuntimeError("softargmax_coordinates expects [B,J,H,W]")
    batch, joints, height, width = heatmaps.shape
    probabilities = F.softmax(heatmaps.float().reshape(batch, joints, -1), dim=-1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=heatmaps.device, dtype=torch.float32),
        torch.arange(width, device=heatmaps.device, dtype=torch.float32),
        indexing="ij",
    )
    x = (probabilities * grid_x.reshape(-1)).sum(-1)
    y = (probabilities * grid_y.reshape(-1)).sum(-1)
    scale_x = float(image_size) / float(width)
    scale_y = float(image_size) / float(height)
    return torch.stack(((x + 0.5) * scale_x - 0.5, (y + 0.5) * scale_y - 0.5), dim=-1)


def hardargmax_coordinates(heatmaps: torch.Tensor, image_size: int) -> torch.Tensor:
    if heatmaps.ndim != 4:
        raise RuntimeError("hardargmax_coordinates expects [B,J,H,W]")
    batch, joints, height, width = heatmaps.shape
    indices = heatmaps.reshape(batch, joints, -1).argmax(dim=-1)
    x = (indices % width).float()
    y = torch.div(indices, width, rounding_mode="floor").float()
    return torch.stack(
        ((x + 0.5) * image_size / width - 0.5,
         (y + 0.5) * image_size / height - 0.5),
        dim=-1,
    )


def masked_coordinate_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visible: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    coordinates = softargmax_coordinates(prediction, image_size)
    safe_target = torch.nan_to_num(target.float(), nan=0.0)
    values = F.smooth_l1_loss(
        coordinates / image_size,
        safe_target / image_size,
        reduction="none",
        beta=0.05,
    ).sum(-1)
    weights = visible.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)

