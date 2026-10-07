"""Conditional-VAE losses shared by LightGen and the electronic baseline."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .modeling import TrainOutput


def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    per_dimension = -0.5 * (1.0 + logvar - mean.square() - logvar.exp())
    if free_bits > 0:
        per_dimension = per_dimension.clamp_min(float(free_bits))
    return per_dimension.mean()


def conditional_vae_loss(
    output: TrainOutput,
    target: torch.Tensor,
    *,
    kl_weight: float,
    free_bits: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent_l1 = F.l1_loss(output.predicted_latent.float(), target.float())
    latent_mse = F.mse_loss(output.predicted_latent.float(), target.float())
    kl = kl_divergence(output.posterior_mean.float(), output.posterior_logvar.float(), free_bits)
    total = latent_l1 + 0.25 * latent_mse + float(kl_weight) * kl
    return total, {
        "loss": float(total.detach()),
        "latent_l1": float(latent_l1.detach()),
        "latent_mse": float(latent_mse.detach()),
        "kl": float(kl.detach()),
        "kl_weight": float(kl_weight),
    }


__all__ = ["conditional_vae_loss", "kl_divergence"]
