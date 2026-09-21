"""Compact one-pass conditional VAE-GAN for the chair-only CleanRender task."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .cleanrender_gan import (
    CleanRenderDiscriminator,
    CleanRenderGANConfig,
    CleanRenderGenerator,
    load_cleanrender_config,
)


@dataclass(frozen=True)
class CleanRenderVAEConfig:
    base: CleanRenderGANConfig
    encoder_channels: tuple[int, ...]
    kl_weight: float
    kl_warmup_epochs: int
    reconstruction_weight: float
    edge_weight: float
    latent_consistency_weight: float
    prior_adversarial_weight: float
    prior_reconstruction_weight: float
    prior_edge_weight: float
    reconstruction_adversarial_weight: float
    mismatch_weight: float
    reference_variation_strength: float

    def validate(self) -> None:
        self.base.validate()
        if self.base.noise_dim <= 0 or len(self.encoder_channels) != 6:
            raise ValueError("The chair VAE encoder requires six positive channel widths")
        if any(value <= 0 or value % 8 for value in self.encoder_channels):
            raise ValueError("Encoder widths must be positive multiples of eight")
        if self.kl_warmup_epochs < 1:
            raise ValueError("kl_warmup_epochs must be positive")
        if min(
            self.kl_weight, self.reconstruction_weight, self.edge_weight,
            self.latent_consistency_weight,
            self.prior_adversarial_weight, self.reconstruction_adversarial_weight,
            self.prior_reconstruction_weight, self.prior_edge_weight,
            self.mismatch_weight, self.reference_variation_strength,
        ) < 0:
            raise ValueError("VAE loss weights and variation strength must be non-negative")


def load_cleanrender_vae_config(path: str | Path) -> CleanRenderVAEConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    vae, loss = raw["vae"], raw["loss"]
    config = CleanRenderVAEConfig(
        base=load_cleanrender_config(path),
        encoder_channels=tuple(int(value) for value in vae["encoder_channels"]),
        kl_weight=float(loss["kl_weight"]),
        kl_warmup_epochs=int(loss["kl_warmup_epochs"]),
        reconstruction_weight=float(loss["reconstruction_weight"]),
        edge_weight=float(loss["edge_weight"]),
        latent_consistency_weight=float(loss["latent_consistency_weight"]),
        prior_adversarial_weight=float(loss["prior_adversarial_weight"]),
        prior_reconstruction_weight=float(loss["prior_reconstruction_weight"]),
        prior_edge_weight=float(loss["prior_edge_weight"]),
        reconstruction_adversarial_weight=float(loss["reconstruction_adversarial_weight"]),
        mismatch_weight=float(loss["mismatch_weight"]),
        reference_variation_strength=float(vae["reference_variation_strength"]),
    )
    config.validate()
    return config


class EncoderResidualDown(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(16, output_channels)
        while output_channels % groups:
            groups -= 1
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )
        self.skip = nn.Conv2d(input_channels, output_channels, 1, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (self.main(value) + self.skip(value)) / math.sqrt(2.0)


class CleanRenderImageEncoder(nn.Module):
    """128x128 RGB -> diagonal Gaussian posterior; used for training and reference edits."""

    def __init__(self, config: CleanRenderVAEConfig) -> None:
        super().__init__()
        widths = config.encoder_channels
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1), nn.SiLU())
        self.blocks = nn.ModuleList([
            EncoderResidualDown(first, second) for first, second in zip(widths, widths[1:])
        ])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1] * 4 * 4, 2 * config.base.noise_dim),
        )
        self.latent_dim = config.base.noise_dim

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.stem(image)
        for block in self.blocks:
            value = block(value)
        parameters = self.head(value)
        mean, log_variance = parameters.chunk(2, dim=1)
        return mean, log_variance.clamp(-8.0, 5.0)

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    @staticmethod
    def seeded_variation(
        mean: torch.Tensor,
        strength: float,
        seed: int,
        log_variance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        generator = torch.Generator(device=mean.device).manual_seed(int(seed))
        noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        del log_variance  # Kept in the API for checkpoint/inference compatibility.
        amount = float(strength)
        if not 0 <= amount <= 1:
            raise ValueError("reference variation strength must be in [0,1]")
        # A convex path has an interpretable endpoint: 0 reconstructs the
        # reference posterior mean, while 1 is a pure seeded prior sample.
        return torch.lerp(mean, noise, amount)


def vae_architecture_report(
    encoder: CleanRenderImageEncoder,
    decoder: CleanRenderGenerator,
    discriminator: CleanRenderDiscriminator,
) -> dict[str, Any]:
    counts = {
        "trainable_encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "trainable_decoder_parameters": sum(p.numel() for p in decoder.parameters()),
        "trainable_discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
    }
    total = sum(counts.values())
    return {
        "variant": "qwen_cleanrender_chair_vae_gan",
        "frozen_qwen_excluded": True,
        **counts,
        "total_trainable_training_parameters": total,
        "total_trainable_inference_parameters_without_reference": counts["trainable_decoder_parameters"],
        "total_trainable_inference_parameters_with_reference": (
            counts["trainable_encoder_parameters"] + counts["trainable_decoder_parameters"]
        ),
        "under_10m_training_budget": total < 10_000_000,
        "inference_decoder_calls": 1,
        "unet": False,
        "vae": True,
        "adversarial_training": True,
    }


def config_payload(config: CleanRenderVAEConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["base"]["generator_channels"] = list(config.base.generator_channels)
    payload["encoder_channels"] = list(config.encoder_channels)
    return payload


__all__ = [
    "CleanRenderImageEncoder", "CleanRenderVAEConfig", "config_payload",
    "load_cleanrender_vae_config", "vae_architecture_report",
]
