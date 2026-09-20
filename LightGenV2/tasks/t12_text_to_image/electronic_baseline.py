"""Pure-electronic, prior-only Qwen + VAE GAN baseline.

Unlike the earlier conditional VAE, this generator is trained through the exact
same path used at inference: cached Qwen text features plus an independent
Gaussian style vector produce one VAE latent in one forward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ElectronicGANConfig:
    noise_dim: int
    condition_dim: int
    mapping_depth: int
    generator_channels: tuple[int, ...]
    noise_spatial_channels: int
    discriminator_width: int
    batch_size: int
    epochs: int
    generator_learning_rate: float
    discriminator_learning_rate: float
    num_workers: int
    amp: bool
    augmentation_probability: float
    adversarial_weight: float
    category_weight: float
    feature_matching_weight: float
    latent_moment_weight: float
    latent_discriminator_weight: float
    text_category_weight: float
    diversity_weight: float
    diversity_target: float
    ema_halflife_kimg: float
    sample_every_epochs: int
    checkpoint_every_epochs: int

    def validate(self) -> None:
        if self.noise_dim <= 0 or self.condition_dim <= 0 or self.mapping_depth < 1:
            raise ValueError("Invalid electronic GAN latent/mapping dimensions")
        if len(self.generator_channels) < 3 or any(value <= 0 for value in self.generator_channels):
            raise ValueError("generator_channels must contain at least three positive widths")
        if self.noise_spatial_channels <= 0:
            raise ValueError("noise_spatial_channels must be positive")
        if self.generator_channels[0] % 32 or any(value % 32 for value in self.generator_channels):
            raise ValueError("All generator channels must be divisible by 32")
        if not 0 <= self.augmentation_probability <= 1:
            raise ValueError("augmentation_probability must be in [0,1]")
        if min(
            self.adversarial_weight,
            self.category_weight,
            self.feature_matching_weight,
            self.latent_moment_weight,
            self.latent_discriminator_weight,
            self.text_category_weight,
            self.diversity_weight,
            self.diversity_target,
        ) < 0:
            raise ValueError("Loss weights must be non-negative")


def load_electronic_gan_config(path: str | Path) -> ElectronicGANConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model, training, loss = raw["model"], raw["training"], raw["loss"]
    config = ElectronicGANConfig(
        noise_dim=int(model["noise_dim"]),
        condition_dim=int(model["condition_dim"]),
        mapping_depth=int(model["mapping_depth"]),
        generator_channels=tuple(int(value) for value in model["generator_channels"]),
        noise_spatial_channels=int(model["noise_spatial_channels"]),
        discriminator_width=int(model["discriminator_width"]),
        batch_size=int(training["batch_size"]),
        epochs=int(training["epochs"]),
        generator_learning_rate=float(training["generator_learning_rate"]),
        discriminator_learning_rate=float(training["discriminator_learning_rate"]),
        num_workers=int(training["num_workers"]),
        amp=bool(training["amp"]),
        augmentation_probability=float(training["augmentation_probability"]),
        adversarial_weight=float(loss["adversarial_weight"]),
        category_weight=float(loss["category_weight"]),
        feature_matching_weight=float(loss["feature_matching_weight"]),
        latent_moment_weight=float(loss["latent_moment_weight"]),
        latent_discriminator_weight=float(loss["latent_discriminator_weight"]),
        text_category_weight=float(loss["text_category_weight"]),
        diversity_weight=float(loss["diversity_weight"]),
        diversity_target=float(loss["diversity_target"]),
        ema_halflife_kimg=float(training["ema_halflife_kimg"]),
        sample_every_epochs=int(training["sample_every_epochs"]),
        checkpoint_every_epochs=int(training["checkpoint_every_epochs"]),
    )
    config.validate()
    return config


class PixelNorm(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.float().square().mean(1, keepdim=True) + 1e-8).to(value.dtype)


class StyledGroupNorm(nn.Module):
    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, affine=False)
        self.affine = nn.Linear(style_dim, 2 * channels)
        nn.init.normal_(self.affine.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.affine.bias)

    def forward(self, value: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(style).chunk(2, dim=1)
        return self.norm(value) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class StyledResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, style_dim: int, *, upsample: bool) -> None:
        super().__init__()
        self.upsample = bool(upsample)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = nn.Conv2d(input_channels, output_channels, 1)
        self.norm1 = StyledGroupNorm(output_channels, style_dim)
        self.norm2 = StyledGroupNorm(output_channels, style_dim)

    def forward(self, value: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        residual = self.skip(value)
        value = F.silu(self.norm1(self.conv1(value), style))
        value = F.silu(self.norm2(self.conv2(value), style))
        return (value + residual) / math.sqrt(2.0)


class QwenElectronicGenerator(nn.Module):
    """Single-pass electronic generator: Qwen feature + z -> 4x28x28 latent."""

    def __init__(
        self, text_dim: int, config: ElectronicGANConfig, latent_channels: int = 4, categories: int = 4
    ) -> None:
        super().__init__()
        self.noise_dim = config.noise_dim
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, config.condition_dim),
            nn.SiLU(),
        )
        self.text_category = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, categories))
        self.category_embedding = nn.Embedding(categories, config.condition_dim)
        mapping: list[nn.Module] = [PixelNorm()]
        for index in range(config.mapping_depth):
            mapping.extend((nn.Linear(config.noise_dim if index == 0 else config.condition_dim, config.condition_dim), nn.SiLU()))
        self.noise_mapping = nn.Sequential(*mapping)
        self.style_fusion = nn.Sequential(nn.Linear(2 * config.condition_dim, config.condition_dim), nn.SiLU())
        channels = config.generator_channels
        self.constant = nn.Parameter(torch.randn(1, channels[0], 7, 7) / math.sqrt(channels[0]))
        self.noise_spatial = nn.Sequential(
            nn.Linear(config.noise_dim, config.noise_spatial_channels * 7 * 7),
            nn.Unflatten(1, (config.noise_spatial_channels, 7, 7)),
            nn.Conv2d(config.noise_spatial_channels, channels[0], 3, padding=1),
        )
        blocks = []
        for index, (input_channels, output_channels) in enumerate(zip(channels, channels[1:])):
            blocks.append(StyledResidualBlock(
                input_channels, output_channels, config.condition_dim, upsample=index < 2
            ))
        self.blocks = nn.ModuleList(blocks)
        self.to_latent = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], latent_channels, 1),
        )
        nn.init.normal_(self.to_latent[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.to_latent[-1].bias)

    def forward_with_aux(self, text: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if noise.shape != (text.shape[0], self.noise_dim):
            raise ValueError(f"Expected noise [B,{self.noise_dim}], got {tuple(noise.shape)}")
        text32 = text.float()
        category_logits = self.text_category(text32)
        category_context = category_logits.softmax(1) @ self.category_embedding.weight
        condition = self.text_projection(text32) + category_context
        noise_style = self.noise_mapping(noise.float())
        style = self.style_fusion(torch.cat((noise_style, condition), dim=1))
        value = self.constant.expand(text.shape[0], -1, -1, -1) + self.noise_spatial(noise.float())
        for block in self.blocks:
            value = block(value, style)
        latent = self.to_latent(value)
        if latent.shape[-2:] != (28, 28):
            raise RuntimeError(f"Electronic generator emitted {tuple(latent.shape)}, expected 28x28")
        return latent, category_logits

    def forward(self, text: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(text, noise)[0]

    def sample_noise(self, batch: int, device: torch.device, *, seed: int | None = None) -> torch.Tensor:
        generator = None if seed is None else torch.Generator(device=device).manual_seed(int(seed))
        return torch.randn(batch, self.noise_dim, device=device, generator=generator)

    def generate(self, text: torch.Tensor, *, seed: int | None = None) -> torch.Tensor:
        return self(text, self.sample_noise(len(text), text.device, seed=seed))

    def architecture_report(self) -> dict[str, Any]:
        return {
            "variant": "qwen_vae_prior_gan",
            "training_path_equals_inference_path": True,
            "image_conditioned_posterior": False,
            "inference_iterations": 1,
            "vae_decoder_calls": 1,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


def _spectral_conv(input_channels: int, output_channels: int, kernel: int, stride: int, padding: int) -> nn.Module:
    return nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, kernel, stride, padding))


class ConditionalImageLatentDiscriminator(nn.Module):
    """Projection discriminator over a decoded image and its VAE latent."""

    def __init__(
        self, text_dim: int, categories: int, width: int = 48, latent_score_weight: float = 0.25
    ) -> None:
        super().__init__()
        self.latent_score_weight = float(latent_score_weight)
        self.image_blocks = nn.ModuleList([
            nn.Sequential(_spectral_conv(3, width, 4, 2, 1), nn.LeakyReLU(0.2)),
            nn.Sequential(_spectral_conv(width, 2 * width, 4, 2, 1), nn.LeakyReLU(0.2)),
            nn.Sequential(_spectral_conv(2 * width, 4 * width, 4, 2, 1), nn.LeakyReLU(0.2)),
        ])
        self.image_tail = nn.ModuleList([
            nn.Sequential(_spectral_conv(4 * width, 8 * width, 4, 2, 1), nn.LeakyReLU(0.2)),
            nn.Sequential(_spectral_conv(8 * width, 8 * width, 4, 2, 1), nn.LeakyReLU(0.2)),
        ])
        self.latent_blocks = nn.ModuleList([
            nn.Sequential(_spectral_conv(4, 2 * width, 3, 1, 1), nn.LeakyReLU(0.2)),
            nn.Sequential(_spectral_conv(2 * width, 4 * width, 4, 2, 1), nn.LeakyReLU(0.2)),
            nn.Sequential(_spectral_conv(4 * width, 8 * width, 4, 2, 1), nn.LeakyReLU(0.2)),
        ])
        feature_dim = 8 * width
        self.image_unconditional = nn.utils.spectral_norm(nn.Linear(feature_dim, 1))
        self.latent_unconditional = nn.utils.spectral_norm(nn.Linear(feature_dim, 1))
        self.text_projection = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, feature_dim))
        self.category = nn.utils.spectral_norm(nn.Linear(feature_dim, categories))

    def forward(
        self, image: torch.Tensor, latent: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        features = []
        image_value = image
        for block in self.image_blocks:
            image_value = block(image_value)
            features.append(image_value)
        for block in self.image_tail:
            image_value = block(image_value)
            features.append(image_value)
        latent_value = latent
        for block in self.latent_blocks:
            latent_value = block(latent_value)
            features.append(latent_value)
        image_pooled = image_value.mean(dim=(-2, -1))
        latent_pooled = latent_value.mean(dim=(-2, -1))
        condition = F.normalize(self.text_projection(text.float()), dim=1)
        projection = (image_pooled * condition).sum(1, keepdim=True) / math.sqrt(image_pooled.shape[1])
        score = (
            self.image_unconditional(image_pooled)
            + projection
            + self.latent_score_weight * self.latent_unconditional(latent_pooled)
        )
        return score, self.category(image_pooled), features


def augment_image_latent_pair(
    image: torch.Tensor,
    latent: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-data spatial ADA that preserves image/latent correspondence."""

    if probability <= 0:
        return image, latent
    batch = len(image)
    flip = (torch.rand(batch, 1, 1, 1, device=image.device) < probability * 0.5)
    image = torch.where(flip, image.flip(-1), image)
    latent = torch.where(flip, latent.flip(-1), latent)
    active = torch.rand(batch, device=image.device) < probability
    shift_x = torch.randint(-2, 3, (batch,), device=image.device).float() * active
    shift_y = torch.randint(-2, 3, (batch,), device=image.device).float() * active
    theta = torch.eye(2, 3, device=image.device).unsqueeze(0).repeat(batch, 1, 1)
    theta[:, 0, 2] = 2 * shift_x / latent.shape[-1]
    theta[:, 1, 2] = 2 * shift_y / latent.shape[-2]
    image_grid = F.affine_grid(theta, image.shape, align_corners=False)
    latent_grid = F.affine_grid(theta, latent.shape, align_corners=False)
    return (
        F.grid_sample(image, image_grid, mode="bilinear", padding_mode="reflection", align_corners=False),
        F.grid_sample(latent, latent_grid, mode="bilinear", padding_mode="reflection", align_corners=False),
    )


__all__ = [
    "ConditionalImageLatentDiscriminator", "ElectronicGANConfig", "QwenElectronicGenerator",
    "augment_image_latent_pair", "load_electronic_gan_config",
]
