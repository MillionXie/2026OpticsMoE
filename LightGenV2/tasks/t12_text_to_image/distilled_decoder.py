"""Compact one-pass latent student distilled from SD-Turbo.

The deployment graph intentionally has no CLIP-condition PCA and no diffusion
UNet. It maps a pooled Qwen feature and one seeded noise latent directly to a
denoised VAE latent. Spatial residual blocks stay explicit so a later optical
branch can run in parallel with the electronic residual branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DistilledDecoderConfig:
    base_channels: int = 64
    condition_dim: int = 256
    blocks_per_level: int = 2
    middle_blocks: int = 3
    dropout: float = 0.0
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    low_frequency_weight: float = 0.25
    gradient_weight: float = 0.0
    num_workers: int = 4
    train_seeds_per_prompt: int = 8
    val_seeds_per_prompt: int = 4

    def validate(self) -> None:
        if min(
            self.base_channels,
            self.condition_dim,
            self.blocks_per_level,
            self.middle_blocks,
            self.batch_size,
            self.epochs,
            self.train_seeds_per_prompt,
            self.val_seeds_per_prompt,
        ) <= 0:
            raise ValueError("Distilled decoder dimensions and training lengths must be positive")
        if self.base_channels % 8:
            raise ValueError("base_channels must be divisible by 8")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if min(self.weight_decay, self.low_frequency_weight, self.gradient_weight) < 0 or self.learning_rate <= 0:
            raise ValueError("Invalid distilled decoder optimizer or loss settings")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_distilled_decoder_config(path: str | Path) -> DistilledDecoderConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    model = raw.get("model", {})
    training = raw.get("training", {})
    config = DistilledDecoderConfig(
        base_channels=int(model.get("base_channels", 64)),
        condition_dim=int(model.get("condition_dim", 256)),
        blocks_per_level=int(model.get("blocks_per_level", 2)),
        middle_blocks=int(model.get("middle_blocks", 3)),
        dropout=float(model.get("dropout", 0.0)),
        batch_size=int(training.get("batch_size", 32)),
        epochs=int(training.get("epochs", 40)),
        learning_rate=float(training.get("learning_rate", 2.0e-4)),
        weight_decay=float(training.get("weight_decay", 1.0e-4)),
        low_frequency_weight=float(training.get("low_frequency_weight", 0.25)),
        gradient_weight=float(training.get("gradient_weight", 0.0)),
        num_workers=int(training.get("num_workers", 4)),
        train_seeds_per_prompt=int(training.get("train_seeds_per_prompt", 8)),
        val_seeds_per_prompt=int(training.get("val_seeds_per_prompt", 4)),
    )
    config.validate()
    return config


def _groups(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class ConditionedResidualBlock(nn.Module):
    """Electronic branch prepared for a future parallel optical branch."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * out_channels))
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.film(condition).chunk(2, dim=-1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return self.skip(value) + hidden


class ResidualStage(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, condition_dim: int, depth: int, dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ConditionedResidualBlock(
                in_channels if index == 0 else out_channels,
                out_channels,
                condition_dim,
                dropout,
            )
            for index in range(depth)
        ])

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value, condition)
        return value


class QwenOneStepLatentStudent(nn.Module):
    """One-pass 64x64 latent U-Net with no iterative denoising loop."""

    def __init__(
        self,
        text_dim: int,
        config: DistilledDecoderConfig,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        config.validate()
        c0, c1, c2 = config.base_channels, 2 * config.base_channels, 4 * config.base_channels
        self.config = config
        self.text_conditioner = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, config.condition_dim),
            nn.SiLU(),
            nn.Linear(config.condition_dim, config.condition_dim),
        )
        self.input_projection = nn.Conv2d(latent_channels, c0, 3, padding=1)
        self.encoder0 = ResidualStage(
            c0, c0, config.condition_dim, config.blocks_per_level, config.dropout
        )
        self.down1 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.encoder1 = ResidualStage(
            c1, c1, config.condition_dim, config.blocks_per_level, config.dropout
        )
        self.down2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.middle = ResidualStage(
            c2, c2, config.condition_dim, config.middle_blocks, config.dropout
        )
        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.decoder1 = ResidualStage(
            2 * c1, c1, config.condition_dim, config.blocks_per_level, config.dropout
        )
        self.up0 = nn.Conv2d(c1, c0, 3, padding=1)
        self.decoder0 = ResidualStage(
            2 * c0, c0, config.condition_dim, config.blocks_per_level, config.dropout
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(c0), c0),
            nn.SiLU(),
            nn.Conv2d(c0, latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, noisy_latent: torch.Tensor, qwen_text: torch.Tensor) -> torch.Tensor:
        condition = self.text_conditioner(qwen_text.float()).to(noisy_latent.dtype)
        level0 = self.encoder0(self.input_projection(noisy_latent), condition)
        level1 = self.encoder1(self.down1(level0), condition)
        middle = self.middle(self.down2(level1), condition)
        up1 = F.interpolate(middle, size=level1.shape[-2:], mode="nearest")
        up1 = self.up1(up1)
        up1 = self.decoder1(torch.cat((up1, level1), dim=1), condition)
        up0 = F.interpolate(up1, size=level0.shape[-2:], mode="nearest")
        up0 = self.up0(up0)
        up0 = self.decoder0(torch.cat((up0, level0), dim=1), condition)
        return noisy_latent + self.output(up0)

    def architecture_report(self) -> dict[str, Any]:
        return {
            "variant": "qwen_one_step_latent_student",
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "inference_iterations": 1,
            "uses_pca_condition": False,
            "uses_condition_tokens": False,
            "uses_sd_turbo_unet_at_inference": False,
            "spatial_residual_blocks": 4 * self.config.blocks_per_level + self.config.middle_blocks,
            "optical_replacement_contract": "parallel branch at each ConditionedResidualBlock input",
        }


__all__ = [
    "ConditionedResidualBlock",
    "DistilledDecoderConfig",
    "QwenOneStepLatentStudent",
    "ResidualStage",
    "load_distilled_decoder_config",
]
