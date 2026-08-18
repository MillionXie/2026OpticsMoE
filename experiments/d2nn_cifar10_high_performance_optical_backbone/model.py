from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .optics import OpticalOEOStage
from .settings import OpticalConfig


Ablation = Literal["normal", "optical_off", "phase_random", "phase_shuffle"]


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
                )
                for index in range(config.num_stages)
            ]
        )
        features = config.input_channels * config.pool_size * config.pool_size
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((config.pool_size, config.pool_size)),
            nn.Flatten(),
            nn.LayerNorm(features),
            nn.Linear(features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, int(num_classes)),
        )

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
        if ablation not in {"normal", "optical_off", "phase_random", "phase_shuffle"}:
            raise ValueError(f"Unsupported ablation: {ablation}")
        amplitude = self._input_amplitude(images)
        diagnostics: list[dict[str, torch.Tensor]] = []
        phases = [stage.phase() for stage in self.stages]
        for index, stage in enumerate(self.stages):
            override = None
            if ablation == "phase_random":
                override = stage.random_phase
            elif ablation == "phase_shuffle":
                override = phases[(index + 1) % len(phases)]
            result = stage(
                amplitude,
                phase_override=override,
                optical_off=ablation == "optical_off",
                return_details=return_diagnostics,
            )
            if return_diagnostics:
                amplitude, details = result
                diagnostics.append(details)
            else:
                amplitude = result
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

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in self.state_dict().items() if name.startswith("stages.")}
