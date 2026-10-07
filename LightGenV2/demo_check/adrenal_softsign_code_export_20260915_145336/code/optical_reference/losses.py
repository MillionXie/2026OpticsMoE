"""Classification losses shared by training and benchmark entry points."""

from __future__ import annotations

import torch.nn.functional as F


def detector_plane_mse_loss(intensity, target_plane, scale, normalize, eps):
    """Full-plane MSE with optional per-sample total-energy matching."""
    eps = float(eps)
    if eps <= 0:
        raise ValueError("loss.detector_plane_mse_normalization_eps must be positive")
    prediction = intensity
    if bool(normalize):
        prediction_energy = prediction.sum(dim=(-2, -1), keepdim=True)
        target_energy = target_plane.sum(dim=(-2, -1), keepdim=True)
        prediction = prediction * target_energy / (prediction_energy + eps)
    return float(scale) * F.mse_loss(prediction, target_plane)


def detector_region_cross_entropy(detector_energies, targets, eps):
    """NLL over relative nonnegative class-detector energies."""
    eps = float(eps)
    if eps <= 0:
        raise ValueError("loss.detector_ce_eps must be positive")
    probabilities = (detector_energies + eps) / (
        detector_energies.sum(dim=1, keepdim=True) + detector_energies.shape[1] * eps
    )
    return F.nll_loss(probabilities.log(), targets)
