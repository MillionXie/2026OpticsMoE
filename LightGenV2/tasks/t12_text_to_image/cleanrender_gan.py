"""Sub-10M single-pass electronic generator for ABO CleanRender."""

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
class CleanRenderGANConfig:
    image_size: int
    text_dim: int
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
    category_weight: float
    text_category_weight: float
    feature_statistics_weight: float
    white_border_weight: float
    ema_halflife_kimg: float
    sample_every_epochs: int
    checkpoint_every_epochs: int

    def validate(self) -> None:
        if self.image_size < 32 or self.image_size & (self.image_size - 1):
            raise ValueError("image_size must be a power of two >= 32")
        if len(self.generator_channels) < 4:
            raise ValueError("At least four generator widths are required")
        initial = self.image_size // (2 ** (len(self.generator_channels) - 1))
        if initial < 4 or initial * 2 ** (len(self.generator_channels) - 1) != self.image_size:
            raise ValueError("generator_channels do not map an integer initial grid to image_size")
        if any(value <= 0 or value % 32 for value in self.generator_channels):
            raise ValueError("All generator channels must be positive multiples of 32")
        if min(self.text_dim, self.noise_dim, self.condition_dim, self.noise_spatial_channels) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.mapping_depth < 1 or self.discriminator_width <= 0:
            raise ValueError("Invalid mapping depth or discriminator width")
        if not 0 <= self.augmentation_probability <= 1:
            raise ValueError("augmentation_probability must be in [0,1]")
        if min(
            self.category_weight, self.text_category_weight,
            self.feature_statistics_weight, self.white_border_weight,
        ) < 0:
            raise ValueError("Loss weights must be non-negative")


def load_cleanrender_config(path: str | Path) -> CleanRenderGANConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model, training, loss = raw["model"], raw["training"], raw["loss"]
    config = CleanRenderGANConfig(
        image_size=int(model["image_size"]),
        text_dim=int(model["text_dim"]),
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
        category_weight=float(loss["category_weight"]),
        text_category_weight=float(loss["text_category_weight"]),
        feature_statistics_weight=float(loss["feature_statistics_weight"]),
        white_border_weight=float(loss["white_border_weight"]),
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
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, value: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(style).chunk(2, dim=1)
        return self.norm(value) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class StyledUpsampleResidual(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, style_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = nn.Conv2d(input_channels, output_channels, 1)
        self.norm1 = StyledGroupNorm(output_channels, style_dim)
        self.norm2 = StyledGroupNorm(output_channels, style_dim)

    def forward(self, value: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        residual = self.skip(value)
        value = F.silu(self.norm1(self.conv1(value), style))
        value = F.silu(self.norm2(self.conv2(value), style))
        return (value + residual) / math.sqrt(2.0)


class CleanRenderGenerator(nn.Module):
    """Frozen-Qwen feature plus Gaussian seed to RGB in one decoder call."""

    def __init__(self, config: CleanRenderGANConfig, categories: int) -> None:
        super().__init__()
        self.config = config
        self.categories = int(categories)
        self.initial_size = config.image_size // 2 ** (len(config.generator_channels) - 1)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(config.text_dim),
            nn.Linear(config.text_dim, config.condition_dim),
            nn.SiLU(),
        )
        self.text_category = nn.Sequential(
            nn.LayerNorm(config.text_dim),
            nn.Linear(config.text_dim, categories),
        )
        self.category_embedding = nn.Embedding(categories, config.condition_dim)
        mapping: list[nn.Module] = [PixelNorm()]
        for index in range(config.mapping_depth):
            input_dim = config.noise_dim if index == 0 else config.condition_dim
            mapping.extend((nn.Linear(input_dim, config.condition_dim), nn.SiLU()))
        self.noise_mapping = nn.Sequential(*mapping)
        self.style_fusion = nn.Sequential(
            nn.Linear(2 * config.condition_dim, config.condition_dim),
            nn.SiLU(),
        )
        channels = config.generator_channels
        self.constant = nn.Parameter(
            torch.randn(1, channels[0], self.initial_size, self.initial_size) / math.sqrt(channels[0])
        )
        self.noise_spatial = nn.Sequential(
            nn.Linear(config.noise_dim, config.noise_spatial_channels * self.initial_size**2),
            nn.Unflatten(1, (config.noise_spatial_channels, self.initial_size, self.initial_size)),
            nn.Conv2d(config.noise_spatial_channels, channels[0], 3, padding=1),
        )
        self.blocks = nn.ModuleList([
            StyledUpsampleResidual(input_channels, output_channels, config.condition_dim)
            for input_channels, output_channels in zip(channels, channels[1:])
        ])
        self.to_rgb = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], 3, 3, padding=1),
            nn.Tanh(),
        )

    def sample_noise(self, batch: int, device: torch.device, seed: int | None = None) -> torch.Tensor:
        generator = None if seed is None else torch.Generator(device=device).manual_seed(int(seed))
        return torch.randn(batch, self.config.noise_dim, device=device, generator=generator)

    def forward_with_aux(self, text: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if noise.shape != (len(text), self.config.noise_dim):
            raise ValueError(f"Expected noise [B,{self.config.noise_dim}], got {tuple(noise.shape)}")
        text32 = text.float()
        category_logits = self.text_category(text32)
        category_context = category_logits.softmax(1) @ self.category_embedding.weight
        condition = self.text_projection(text32) + category_context
        style = self.style_fusion(torch.cat((condition, self.noise_mapping(noise.float())), dim=1))
        value = self.constant.expand(len(text), -1, -1, -1) + self.noise_spatial(noise.float())
        for block in self.blocks:
            value = block(value, style)
        image = self.to_rgb(value)
        if image.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise RuntimeError(f"Generator emitted {tuple(image.shape)}")
        return image, category_logits

    def forward(self, text: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(text, noise)[0]

    def generate(self, text: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        return self(text, self.sample_noise(len(text), text.device, seed))


def _spectral_conv(input_channels: int, output_channels: int) -> nn.Module:
    return nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, 4, 2, 1))


class CleanRenderDiscriminator(nn.Module):
    def __init__(self, config: CleanRenderGANConfig, categories: int) -> None:
        super().__init__()
        widths = (
            config.discriminator_width,
            2 * config.discriminator_width,
            4 * config.discriminator_width,
            6 * config.discriminator_width,
            8 * config.discriminator_width,
        )
        inputs = (3, *widths[:-1])
        self.blocks = nn.ModuleList([
            nn.Sequential(_spectral_conv(input_channels, output_channels), nn.LeakyReLU(0.2))
            for input_channels, output_channels in zip(inputs, widths)
        ])
        feature_dim = widths[-1]
        self.unconditional = nn.utils.spectral_norm(nn.Linear(feature_dim, 1))
        self.text_projection = nn.Sequential(
            nn.LayerNorm(config.text_dim),
            nn.Linear(config.text_dim, feature_dim),
        )
        self.category = nn.utils.spectral_norm(nn.Linear(feature_dim, categories))

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        value = image
        for block in self.blocks:
            value = block(value)
            features.append(value)
        pooled = value.mean((-2, -1))
        condition = F.normalize(self.text_projection(text.float()), dim=1)
        projection = (pooled * condition).sum(1, keepdim=True) / math.sqrt(pooled.shape[1])
        return self.unconditional(pooled) + projection, self.category(pooled), features


def architecture_report(
    generator: CleanRenderGenerator,
    discriminator: CleanRenderDiscriminator | None = None,
) -> dict[str, Any]:
    generator_parameters = sum(parameter.numel() for parameter in generator.parameters())
    discriminator_parameters = 0 if discriminator is None else sum(
        parameter.numel() for parameter in discriminator.parameters()
    )
    total = generator_parameters + discriminator_parameters
    return {
        "variant": "qwen_cleanrender_direct_gan",
        "frozen_qwen_excluded": True,
        "trainable_generator_parameters": generator_parameters,
        "trainable_discriminator_parameters": discriminator_parameters,
        "total_trainable_training_parameters": total,
        "under_10m_training_budget": total < 10_000_000,
        "inference_iterations": 1,
        "inference_decoder_calls": 1,
        "image_encoder_at_inference": False,
        "unet": False,
        "vae": False,
    }


__all__ = [
    "CleanRenderDiscriminator", "CleanRenderGANConfig", "CleanRenderGenerator",
    "architecture_report", "load_cleanrender_config",
]
