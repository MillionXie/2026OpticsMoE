"""Single-pass chair style transfer with an optional parallel optical bottleneck."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .modeling import CompactFourierOptics, ConditionedElectronicResidual, ScaleMatchedFusion


STYLE_PROMPTS = (
    "a product photo of a chair made from warm walnut wood on a clean white background",
    "a product photo of a matte charcoal black chair on a clean white background",
    "a product photo of an ivory white chair on a clean white background",
    "a product photo of a cobalt blue chair on a clean white background",
)
STYLE_NAMES = ("warm_walnut", "matte_charcoal", "ivory_white", "cobalt_blue")


@dataclass(frozen=True)
class ChairStyleConfig:
    image_size: int
    text_dim: int
    condition_dim: int
    widths: tuple[int, ...]
    optical: bool
    optical_experts: int
    optical_top_k: int
    alpha_initial: float
    alpha_minimum: float
    alpha_maximum: float
    rms_epsilon: float
    residual_limit: float
    batch_size: int
    epochs: int
    learning_rate: float
    phase_learning_rate: float
    weight_decay: float
    num_workers: int
    amp: bool
    adversarial_weight: float
    reconstruction_weight: float
    edge_weight: float
    identity_edge_weight: float
    background_weight: float
    residual_weight: float
    adversarial_warmup_epochs: int
    sample_every_epochs: int

    def validate(self) -> None:
        if self.image_size != 128:
            raise ValueError("The first style-transfer release is fixed to 128x128")
        if len(self.widths) != 4 or any(value <= 0 for value in self.widths):
            raise ValueError("widths must contain four positive values")
        if self.optical_top_k > self.optical_experts:
            raise ValueError("optical_top_k cannot exceed optical_experts")
        if not 0 <= self.alpha_minimum < self.alpha_initial < self.alpha_maximum <= 1:
            raise ValueError("Invalid fusion-alpha interval")
        if not 0 < self.residual_limit <= 1:
            raise ValueError("residual_limit must be in (0,1]")
        if min(self.batch_size, self.epochs, self.learning_rate) <= 0:
            raise ValueError("Training dimensions must be positive")


def load_chair_style_config(path: str | Path) -> ChairStyleConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model, fusion, training, loss = raw["model"], raw["fusion"], raw["training"], raw["loss"]
    config = ChairStyleConfig(
        image_size=int(model["image_size"]), text_dim=int(model["text_dim"]),
        condition_dim=int(model["condition_dim"]), widths=tuple(int(x) for x in model["widths"]),
        optical=bool(model["optical"]), optical_experts=int(model["optical_experts"]),
        optical_top_k=int(model["optical_top_k"]),
        alpha_initial=float(fusion["alpha_initial"]), alpha_minimum=float(fusion["alpha_minimum"]),
        alpha_maximum=float(fusion["alpha_maximum"]), rms_epsilon=float(fusion["rms_epsilon"]),
        residual_limit=float(model["residual_limit"]), batch_size=int(training["batch_size"]),
        epochs=int(training["epochs"]), learning_rate=float(training["learning_rate"]),
        phase_learning_rate=float(training["phase_learning_rate"]),
        weight_decay=float(training["weight_decay"]), num_workers=int(training["num_workers"]),
        amp=bool(training["amp"]), adversarial_weight=float(loss["adversarial_weight"]),
        reconstruction_weight=float(loss["reconstruction_weight"]), edge_weight=float(loss["edge_weight"]),
        identity_edge_weight=float(loss["identity_edge_weight"]),
        background_weight=float(loss["background_weight"]), residual_weight=float(loss["residual_weight"]),
        adversarial_warmup_epochs=int(training["adversarial_warmup_epochs"]),
        sample_every_epochs=int(training["sample_every_epochs"]),
    )
    config.validate()
    return config


class ResidualDown(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(16, output_channels)
        while output_channels % groups:
            groups -= 1
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1),
            nn.GroupNorm(groups, output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels), nn.SiLU(),
        )
        self.skip = nn.Conv2d(input_channels, output_channels, 1, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (self.main(value) + self.skip(value)) * (2.0**-0.5)


class ConditionedUp(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        groups = min(16, output_channels)
        while output_channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, output_channels, affine=False)
        self.norm2 = nn.GroupNorm(groups, output_channels, affine=False)
        self.affine = nn.Linear(condition_dim, 4 * output_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = torch.cat((value, skip), dim=1)
        scale1, shift1, scale2, shift2 = self.affine(condition).chunk(4, dim=1)
        value = self.norm1(self.conv1(value)) * (1 + scale1[:, :, None, None]) + shift1[:, :, None, None]
        value = F.silu(value)
        value = self.norm2(self.conv2(value)) * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        return F.silu(value)


class ParallelStyleBottleneck(nn.Module):
    """Electronic and optical branches consume the same 16x16 feature map."""

    def __init__(self, width: int, condition_dim: int, config: ChairStyleConfig) -> None:
        super().__init__()
        self.width = int(width)
        self.grid = config.image_size // 8
        self.electronic = ConditionedElectronicResidual(width, condition_dim, self.grid)
        self.optical_enabled = config.optical
        if config.optical:
            self.optical_condition = nn.Linear(condition_dim, 2 * width)
            self.optical = CompactFourierOptics(
                width, self.grid, experts=config.optical_experts, top_k=config.optical_top_k,
            )
            self.expert_gate = nn.Parameter(torch.tensor(-2.0))
            self.global_gate = nn.Parameter(torch.tensor(-2.0))
            self.fusion = ScaleMatchedFusion(
                config.alpha_initial, config.alpha_minimum, config.alpha_maximum, config.rms_epsilon,
            )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        tokens = value.flatten(2).transpose(1, 2)
        electronic = self.electronic(tokens, condition)
        if not self.optical_enabled:
            return electronic.transpose(1, 2).reshape_as(value)
        scale, shift = self.optical_condition(condition).chunk(2, dim=1)
        optical_input = tokens * (1 + scale[:, None]) + shift[:, None]
        expert = self.optical.expert(optical_input)
        stage1 = tokens + torch.sigmoid(self.expert_gate) * expert
        optical = stage1 + torch.sigmoid(self.global_gate) * self.optical.global_block(stage1)
        fused = self.fusion(electronic, optical)
        return fused.transpose(1, 2).reshape_as(value)


class ChairStyleTransfer(nn.Module):
    """Geometry-preserving image-to-image model; decoder predicts only a bounded RGB residual."""

    def __init__(self, config: ChairStyleConfig) -> None:
        super().__init__()
        self.config = config
        widths = config.widths
        self.condition = nn.Sequential(
            nn.LayerNorm(config.text_dim), nn.Linear(config.text_dim, config.condition_dim), nn.SiLU(),
            nn.Linear(config.condition_dim, config.condition_dim), nn.SiLU(),
        )
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1), nn.SiLU())
        self.down1 = ResidualDown(widths[0], widths[1])
        self.down2 = ResidualDown(widths[1], widths[2])
        self.down3 = ResidualDown(widths[2], widths[3])
        self.bottleneck = ParallelStyleBottleneck(widths[3], config.condition_dim, config)
        self.up3 = ConditionedUp(widths[3], widths[2], widths[2], config.condition_dim)
        self.up2 = ConditionedUp(widths[2], widths[1], widths[1], config.condition_dim)
        self.up1 = ConditionedUp(widths[1], widths[0], widths[0], config.condition_dim)
        self.to_delta = nn.Sequential(nn.Conv2d(widths[0], widths[0], 3, padding=1), nn.SiLU(), nn.Conv2d(widths[0], 3, 3, padding=1))
        nn.init.zeros_(self.to_delta[-1].weight)
        nn.init.zeros_(self.to_delta[-1].bias)

    @staticmethod
    def foreground_mask(reference: torch.Tensor) -> torch.Tensor:
        rgb = reference.float().add(1).mul(0.5)
        distance_from_white = (1 - rgb).amax(dim=1, keepdim=True)
        return ((distance_from_white - 0.025) / 0.18).clamp(0, 1)

    def forward_with_aux(self, reference: torch.Tensor, style_text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = self.condition(style_text.float())
        s0 = self.stem(reference)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        value = self.bottleneck(self.down3(s2), condition)
        value = self.up3(value, s2, condition)
        value = self.up2(value, s1, condition)
        value = self.up1(value, s0, condition)
        delta = torch.tanh(self.to_delta(value)) * self.config.residual_limit
        mask = self.foreground_mask(reference).to(delta.dtype)
        output = (reference + mask * delta).clamp(-1, 1)
        return output, delta, mask

    def forward(self, reference: torch.Tensor, style_text: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(reference, style_text)[0]


class ChairStyleDiscriminator(nn.Module):
    """Small text-projection discriminator used only while training."""

    def __init__(self, text_dim: int, width: int = 32) -> None:
        super().__init__()
        channels = (width, 2 * width, 4 * width, 6 * width)
        inputs = (3, *channels[:-1])
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(first, second, 4, 2, 1)),
                nn.LeakyReLU(0.2),
            )
            for first, second in zip(inputs, channels)
        ])
        self.score = nn.utils.spectral_norm(nn.Linear(channels[-1], 1))
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, channels[-1]))

    def forward(self, image: torch.Tensor, style_text: torch.Tensor) -> torch.Tensor:
        value = image
        for block in self.blocks:
            value = block(value)
        value = value.mean((-2, -1))
        condition = F.normalize(self.text(style_text.float()), dim=1)
        return self.score(value) + (value * condition).sum(1, keepdim=True) / (value.shape[1] ** 0.5)


def architecture_report(model: ChairStyleTransfer, discriminator: nn.Module | None = None) -> dict[str, Any]:
    generator = sum(p.numel() for p in model.parameters())
    discriminator_count = 0 if discriminator is None else sum(p.numel() for p in discriminator.parameters())
    optical = sum(p.numel() for name, p in model.named_parameters() if "bottleneck.optical" in name)
    return {
        "variant": "chair_style_transfer_parallel_optical" if model.config.optical else "chair_style_transfer_electronic",
        "trainable_generator_parameters": generator,
        "trainable_discriminator_parameters": discriminator_count,
        "total_training_parameters": generator + discriminator_count,
        "optical_path_parameters": optical,
        "single_pass": True,
        "decoder_calls": 1,
        "unet_diffusion": False,
        "pixel_identity_skip": True,
        "electronic_and_optical_parallel": bool(model.config.optical),
        "optical_backend": "compact_fft_simulation" if model.config.optical else None,
    }


def config_payload(config: ChairStyleConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["widths"] = list(config.widths)
    return payload


__all__ = [
    "ChairStyleConfig", "ChairStyleDiscriminator", "ChairStyleTransfer", "STYLE_NAMES", "STYLE_PROMPTS",
    "architecture_report", "config_payload", "load_chair_style_config",
]
