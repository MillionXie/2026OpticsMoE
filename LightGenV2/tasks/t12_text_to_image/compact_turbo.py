"""Shared operators for the compact, one-call Turbo UNet student."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F


@dataclass(frozen=True)
class CompactTurboConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    detail_weight: float
    num_workers: int

    def validate(self) -> None:
        if min(self.batch_size, self.epochs) <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or min(self.weight_decay, self.detail_weight, self.num_workers) < 0:
            raise ValueError("Invalid compact Turbo optimization settings")


def load_compact_turbo_config(path: str | Path) -> CompactTurboConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    training, loss = raw["training"], raw.get("loss", {})
    config = CompactTurboConfig(
        batch_size=int(training["batch_size"]),
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        detail_weight=float(loss.get("detail_weight", 0.0)),
        num_workers=int(training["num_workers"]),
    )
    config.validate()
    return config


def one_step_denoise(
    unet: torch.nn.Module,
    noise: torch.Tensor,
    condition: torch.Tensor,
    sigma: torch.Tensor,
    *,
    timestep: int = 999,
) -> torch.Tensor:
    """Perform the SD-Turbo Euler update with exactly one UNet invocation."""

    latent = noise * sigma
    scaled = latent / torch.sqrt(sigma.square() + 1)
    timesteps = torch.full((len(noise),), timestep, device=noise.device, dtype=torch.long)
    prediction = unet(
        scaled, timesteps, encoder_hidden_states=condition, return_dict=False
    )[0]
    return latent - sigma * prediction


def latent_gradient_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 matching of horizontal and vertical latent differences."""

    horizontal = F.l1_loss(
        output[..., 1:] - output[..., :-1], target[..., 1:] - target[..., :-1]
    )
    vertical = F.l1_loss(
        output[..., 1:, :] - output[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    )
    return horizontal + vertical


__all__ = [
    "CompactTurboConfig",
    "latent_gradient_loss",
    "load_compact_turbo_config",
    "one_step_denoise",
]
