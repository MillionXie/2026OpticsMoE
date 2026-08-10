from __future__ import annotations

import torch
from torch.nn import functional as F

from .datasets import FLIP_PERMUTATION


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


def masked_heatmap_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    visible: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft-target heatmap distillation between student and teacher.

    Both heatmaps are [B,J,H,W]. The frozen teacher's heatmaps are used
    directly as soft spatial regression targets (they carry the teacher's
    localization uncertainty, unlike the fixed-sigma Gaussian ground truth),
    with a per-pixel MSE masked to visible joints. A spatial-softmax version
    would normalise away amplitude and shrink to ~1/(H*W) scale, making the
    term negligible, so direct heatmap MSE (the standard pose-distillation
    objective) is used instead. `temperature` is reserved for future
    softening and defaults to 1.0 (identity).
    """
    if student.shape != teacher.shape or student.ndim != 4:
        raise RuntimeError(
            f"Distillation heatmaps must match [B,J,H,W], got {tuple(student.shape)} "
            f"and {tuple(teacher.shape)}"
        )
    if visible.shape != student.shape[:2]:
        raise RuntimeError("Visibility mask must be [B,J]")
    del temperature
    weights = visible.to(student.dtype)[..., None, None]
    denominator = weights.sum().clamp_min(1.0) * student.shape[-1] * student.shape[-2]
    return ((student.float() - teacher.float()).square() * weights).sum() / denominator


def warp_cached_heatmaps(
    cached: torch.Tensor,
    canon_boxes: torch.Tensor,
    jit_boxes: torch.Tensor,
    flipped: torch.Tensor,
    image_size: int,
    heatmap_size: int,
) -> torch.Tensor:
    """Warp cached teacher heatmaps into the augmented student view.

    ``cached`` holds teacher heatmaps ``[B,J,H,W]`` computed on the canonical
    (un-augmented) center crop of each training image. The student instead sees
    a randomly jittered crop (plus optional horizontal flip) of the same image,
    so each cached heatmap is resampled with the similarity transform that maps
    the canonical crop box onto the jittered crop box, then flipped (with the
    matching joint permutation) for samples where the augmentation flipped the
    image. Returns warped heatmaps ``[B,J,H,W]`` in the student's 224-space.
    """
    batch, joints, height, width = cached.shape
    device = cached.device
    canon = canon_boxes.to(device).float()
    jit = jit_boxes.to(device).float()
    cl, ct, cr, cb = canon.unbind(1)
    jl, jt, jr, jb = jit.unbind(1)
    canon_w = (cr - cl).clamp_min(1.0)
    canon_h = (cb - ct).clamp_min(1.0)
    jit_w = (jr - jl).clamp_min(1.0)
    jit_h = (jb - jt).clamp_min(1.0)
    scale_x = (image_size / jit_w) / (image_size / canon_w)
    scale_y = (image_size / jit_h) / (image_size / canon_h)
    tx = (cl - jl) * (image_size / jit_w)
    ty = (ct - jt) * (image_size / jit_h)
    ratio = heatmap_size / image_size
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device), indexing="ij",
    )
    # Heatmap pixel j represents 224-space coordinate j * (image_size/heatmap_size)
    # (matches make_heatmaps: peak at keypoint * heatmap_size/image_size).
    x_aug = xs * (image_size / heatmap_size)   # [H,W] augmented-224 coords
    y_aug = ys * (image_size / heatmap_size)
    flip_f = flipped.to(device).float()[:, None, None]
    # For flipped samples the augmented image is a reflection of resize(jit_crop):
    # x_jit = (image_size-1) - x_aug ; otherwise x_jit = x_aug.
    x_jit = (image_size - 1 - x_aug) * flip_f + x_aug * (1.0 - flip_f)
    y_jit = y_aug
    x_canon = (x_jit - tx[:, None, None]) / scale_x[:, None, None]
    y_canon = (y_jit - ty[:, None, None]) / scale_y[:, None, None]
    x_src = x_canon * ratio
    y_src = y_canon * ratio
    grid = torch.stack([
        2.0 * (x_src + 0.5) / width - 1.0,
        2.0 * (y_src + 0.5) / height - 1.0,
    ], dim=-1)
    warped = F.grid_sample(
        cached, grid, mode="bilinear", padding_mode="border", align_corners=False,
    )
    flip_mask = flipped.to(device)
    if bool(flip_mask.any()):
        warped[flip_mask] = warped[flip_mask][:, torch.as_tensor(FLIP_PERMUTATION, device=device)]
    return warped


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

