from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .optics import FeedbackMode, OpticalOEOStage
from .settings import OpticalConfig


Ablation = Literal[
    "normal",
    "optical_off",
    "phase_random",
    "phase_shuffle",
    "electronic_skip_off",
    "long_skip_off",
]


class DualPoolReadout(nn.Module):
    """Combine mean energy and salient diffraction peaks without a deep CNN."""

    def __init__(self, config: OpticalConfig, num_classes: int) -> None:
        super().__init__()
        features = 2 * config.input_channels * config.pool_size * config.pool_size
        self.pool_size = int(config.pool_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, int(num_classes)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        size = (self.pool_size, self.pool_size)
        average = F.adaptive_avg_pool2d(value, size)
        maximum = F.adaptive_max_pool2d(value, size)
        return self.classifier(torch.cat((average, maximum), dim=1).flatten(1))


def _build_head(config: OpticalConfig, num_classes: int) -> nn.Module:
    if config.readout_mode == "mlp":
        features = config.input_channels * config.pool_size * config.pool_size
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((config.pool_size, config.pool_size)),
            nn.Flatten(),
            nn.LayerNorm(features),
            nn.Linear(features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, int(num_classes)),
        )
    if config.readout_mode == "dual_pool":
        return DualPoolReadout(config, num_classes)
    channels = int(config.conv_channels)
    groups = 8 if channels % 8 == 0 else 1
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((config.pool_size, config.pool_size)),
        nn.Conv2d(config.input_channels, channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(groups, channels),
        nn.GELU(),
        nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.GroupNorm(groups, 2 * channels),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(2),
        nn.Flatten(),
        nn.LayerNorm(8 * channels),
        nn.Linear(8 * channels, config.hidden_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, int(num_classes)),
    )


class OpticalClassifier(nn.Module):
    def __init__(self, config: OpticalConfig, num_classes: int) -> None:
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList(
            [
                OpticalOEOStage(
                    size=config.canvas_size,
                    channels=config.input_channels,
                    wavelength_m=config.wavelength_m,
                    pixel_size_m=config.pixel_size_m,
                    distance_m=config.propagation_distance_m,
                    phase_init_std=config.phase_init_std,
                    layernorm_eps=config.layernorm_eps,
                    residual_mode=config.residual_mode,
                    residual_main_init=config.residual_main_init,
                    residual_main_min=config.residual_main_min,
                    normalize_branch_rms=config.normalize_branch_rms,
                    random_seed=1729 + index,
                    electronic_skip_mode=config.electronic_skip_mode,
                    electronic_skip_hidden_channels=config.electronic_skip_hidden_channels,
                    electronic_skip_downsample_factor=config.electronic_skip_downsample_factor,
                    electronic_skip_scale_init=config.electronic_skip_scale_init,
                    electronic_skip_scale_max=config.electronic_skip_scale_max,
                    long_skip_enabled=(
                        config.long_skip_enabled and index > config.num_stages // 2
                    ),
                    long_skip_weight_init=config.long_skip_weight_init,
                    long_skip_weight_max=config.long_skip_weight_max,
                )
                for index in range(config.num_stages)
            ]
        )
        self.head = _build_head(config, num_classes)
        self.num_classes = int(num_classes)

    def _input_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        value = images.float()
        if self.config.input_channels == 1 and value.shape[1] == 3:
            value = (0.299 * value[:, 0] + 0.587 * value[:, 1] + 0.114 * value[:, 2]).unsqueeze(1)
        if value.shape[1] != self.config.input_channels:
            raise ValueError(f"Expected {self.config.input_channels} input channels, got {value.shape[1]}")
        value = F.interpolate(
            value,
            size=(self.config.canvas_size, self.config.canvas_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_min(0.0)
        # Dataset pixels describe intensity; the propagated quantity is amplitude.
        return value.sqrt()

    def forward(
        self,
        images: torch.Tensor,
        *,
        ablation: Ablation = "normal",
        return_diagnostics: bool = False,
    ):
        if ablation not in {
            "normal",
            "optical_off",
            "phase_random",
            "phase_shuffle",
            "electronic_skip_off",
            "long_skip_off",
        }:
            raise ValueError(f"Unsupported ablation: {ablation}")
        amplitude = self._input_amplitude(images)
        diagnostics: list[dict[str, torch.Tensor]] = []
        phases = [stage.phase() for stage in self.stages]
        stage_outputs: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            override = None
            if ablation == "phase_random":
                override = stage.random_phase
            elif ablation == "phase_shuffle":
                override = phases[(index + 1) % len(phases)]
            source_index = len(self.stages) - 1 - index
            long_skip = None
            if self.config.long_skip_enabled and 0 <= source_index < index - 1:
                long_skip = stage_outputs[source_index]
            result = stage(
                amplitude,
                phase_override=override,
                optical_off=ablation == "optical_off",
                long_skip=long_skip,
                disable_electronic_skip=ablation == "electronic_skip_off",
                disable_long_skip=ablation == "long_skip_off",
                return_details=return_diagnostics,
            )
            if return_diagnostics:
                amplitude, details = result
                diagnostics.append(details)
            else:
                amplitude = result
            stage_outputs.append(amplitude)
        logits = self.head(amplitude)
        return (logits, diagnostics) if return_diagnostics else logits

    def phase_parameters(self):
        for stage in self.stages:
            yield stage.raw_phase

    def residual_parameters(self):
        phase_ids = {id(parameter) for parameter in self.phase_parameters()}
        head_ids = {id(parameter) for parameter in self.head.parameters()}
        for parameter in self.parameters():
            if id(parameter) not in phase_ids and id(parameter) not in head_ids:
                yield parameter

    def electronic_parameters(self):
        yield from self.head.parameters()

    def optical_weights(self) -> list[float]:
        return [float(stage.residual.main_weight().detach().cpu()) for stage in self.stages]

    def electronic_skip_scales(self) -> list[float]:
        return [float(stage.electronic_skip.transform_scale().detach().cpu()) for stage in self.stages]

    def long_skip_weights(self) -> list[float]:
        return [float(stage.electronic_skip.long_skip_weight().detach().cpu()) for stage in self.stages]

    def estimated_electronic_macs(self) -> int:
        """Count convolution/linear MACs per sample; norms, activations and resampling are excluded."""

        config = self.config
        channels = config.input_channels
        hidden = config.electronic_skip_hidden_channels
        size = config.canvas_size
        per_stage = 0
        if config.electronic_skip_mode == "pointwise":
            per_stage = size * size * (channels * hidden + hidden * channels)
        elif config.electronic_skip_mode == "depthwise":
            per_stage = size * size * (
                9 * channels + channels * hidden + hidden * channels
            )
        elif config.electronic_skip_mode == "lowres":
            low = size // config.electronic_skip_downsample_factor
            per_stage = low * low * (
                9 * channels * hidden + 9 * hidden * hidden + hidden * channels
            )
        residual = config.num_stages * per_stage
        pool = config.pool_size
        if config.readout_mode == "mlp":
            features = channels * pool * pool
            head = features * config.hidden_dim + config.hidden_dim * self.num_classes
        elif config.readout_mode == "dual_pool":
            features = 2 * channels * pool * pool
            head = features * config.hidden_dim + config.hidden_dim * self.num_classes
        else:
            conv = config.conv_channels
            second = (pool + 1) // 2
            head = (
                pool * pool * 9 * channels * conv
                + second * second * 9 * conv * (2 * conv)
                + 8 * conv * config.hidden_dim
                + config.hidden_dim * self.num_classes
            )
        return int(residual + head)

    def snapshot_phases(self) -> torch.Tensor:
        return torch.stack([stage.phase().detach().cpu() for stage in self.stages], dim=0)

    def configure_feedback(
        self,
        mode: FeedbackMode,
        *,
        pretrained_phases: torch.Tensor | None = None,
        random_seed: int = 0,
    ) -> None:
        expected = (
            len(self.stages),
            self.config.input_channels,
            self.config.canvas_size,
            self.config.canvas_size,
        )
        if mode == "fa_pretrained" and (
            pretrained_phases is None or tuple(pretrained_phases.shape) != expected
        ):
            raise ValueError(f"pretrained_phases must have shape {expected}")
        generator = torch.Generator().manual_seed(int(random_seed))
        for index, stage in enumerate(self.stages):
            phase = None
            if mode == "fa_pretrained":
                phase = pretrained_phases[index]
            elif mode == "fa_random":
                phase = 2.0 * torch.pi * torch.rand(
                    stage.channels,
                    stage.size,
                    stage.size,
                    generator=generator,
                )
            stage.set_feedback(mode, phase)

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in self.state_dict().items() if name.startswith("stages.")}
