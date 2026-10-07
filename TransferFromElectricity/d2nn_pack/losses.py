"""Pure-optical classification loss via MSE to target intensity patterns.

Each class has a pre-defined Gaussian spot on the detector plane.
Sensor output is per-sample max-normalised to [0, 1].
Targets are also max-normalised to [0, 1] (peak = 1.0).
MSE works at a meaningful scale (~1e-2 to 1e-1) while being scale-invariant.
"""

import math

import torch
import torch.nn.functional as F


def gaussian_spot(
    size: int,
    cx: float,
    cy: float,
    sigma: float = 12.0,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Create a 2D Gaussian spot on a [size, size] grid."""
    ys = torch.arange(size, dtype=torch.float32)
    xs = torch.arange(size, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    dist2 = (gy - cy).square() + (gx - cx).square()
    spot = amplitude * torch.exp(-dist2 / (2.0 * sigma * sigma))
    return spot


def generate_target_patterns(
    num_classes: int = 10,
    size: int = 256,
    sigma: float = 14.0,
) -> torch.Tensor:
    """Generate one target intensity pattern per class.

    Spots are arranged in a near-square grid. Each is max-normalised (peak=1).
    """
    cols = int(math.ceil(math.sqrt(num_classes)))
    rows = int(math.ceil(float(num_classes) / cols))

    margin = size * 0.12
    usable = size - 2 * margin
    patterns = []
    for c in range(num_classes):
        r = c // cols
        col = c % cols
        cy = margin + usable * (r + 0.5) / rows
        cx = margin + usable * (col + 0.5) / cols
        spot = gaussian_spot(size, cx, cy, sigma=sigma)
        # Max-normalise so peak = 1.0
        spot = spot / (spot.max() + 1e-8)
        patterns.append(spot)
    return torch.stack(patterns, dim=0)  # [num_classes, size, size]


def _max_normalise(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample normalise to [0, 1] range: x / max(x)."""
    return x / (x.amax(dim=(-2, -1), keepdim=True) + eps)


def optical_classification_loss(
    sensor_output: torch.Tensor,
    target_patterns: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between max-normalised sensor output and target pattern.

    Scale-invariant — only spatial pattern matters, not absolute intensity.

    Args:
        sensor_output: [B, H, W] raw intensity from detector plane.
        target_patterns: [C, H, W] pre-defined target patterns (peak=1).
        labels: [B] integer class labels.

    Returns:
        scalar MSE loss.
    """
    sensor_norm = _max_normalise(sensor_output)   # [B, H, W] in [0, 1]
    targets = target_patterns[labels]              # [B, H, W] in [0, 1]
    return F.mse_loss(sensor_norm, targets)


@torch.no_grad()
def optical_predict(
    sensor_output: torch.Tensor,
    target_patterns: torch.Tensor,
) -> torch.Tensor:
    """Predict class by smallest MSE to each target (max-normalised).

    Args:
        sensor_output: [B, H, W]
        target_patterns: [C, H, W]

    Returns:
        pred_labels: [B]
    """
    sensor_norm = _max_normalise(sensor_output)   # [B, H, W]
    diff = sensor_norm.unsqueeze(1) - target_patterns.unsqueeze(0)  # [B, C, H, W]
    mse = diff.pow(2).mean(dim=(-2, -1))           # [B, C]
    return mse.argmin(dim=1)


__all__ = [
    "gaussian_spot",
    "generate_target_patterns",
    "optical_classification_loss",
    "optical_predict",
]


__all__ = [
    "gaussian_spot",
    "generate_target_patterns",
    "optical_classification_loss",
    "optical_predict",
]


__all__ = [
    "gaussian_spot",
    "generate_target_patterns",
    "optical_classification_loss",
    "optical_predict",
]
