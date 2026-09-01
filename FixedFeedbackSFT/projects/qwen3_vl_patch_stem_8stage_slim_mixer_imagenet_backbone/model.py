from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import (
    OpticalOEOStage,
    rms_normalize,
)
from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.model import (
    CLIP_MEAN,
    CLIP_STD,
    TokenAdapter,
    TokenClassificationReadout,
)
from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.stem import (
    StaticQwenPatchStem,
)


Ablation = Literal["normal", "optical_off", "phase_random", "electronic_skip_off"]


def _bounded_logit(initial: float, maximum: float) -> torch.Tensor:
    if maximum <= 0.0:
        raise ValueError("The maximum gate value must be positive")
    probability = min(max(float(initial) / float(maximum), 1.0e-5), 1.0 - 1.0e-5)
    return torch.tensor(math.log(probability / (1.0 - probability)))


def qwen_tokens_to_grid(tokens: torch.Tensor, *, grid_size: int = 14, merge_size: int = 2) -> torch.Tensor:
    """Restore Qwen block-major tokens to their true 2-D patch grid."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,T,C] tokens, got {tuple(tokens.shape)}")
    batch, token_count, channels = tokens.shape
    if token_count != int(grid_size) ** 2 or int(grid_size) % int(merge_size):
        raise ValueError("Token count does not match the configured square Qwen grid")
    blocks = int(grid_size) // int(merge_size)
    return (
        tokens.view(batch, blocks, blocks, merge_size, merge_size, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, grid_size, grid_size)
    )


def grid_to_qwen_tokens(grid: torch.Tensor, *, merge_size: int = 2) -> torch.Tensor:
    """Return a true 2-D patch grid to Qwen's block-major token order."""

    if grid.ndim != 4 or grid.shape[-1] != grid.shape[-2]:
        raise ValueError(f"Expected a square [B,C,H,W] grid, got {tuple(grid.shape)}")
    batch, channels, grid_size, _ = grid.shape
    if grid_size % int(merge_size):
        raise ValueError("Grid size must be divisible by the Qwen merge size")
    blocks = grid_size // int(merge_size)
    return (
        grid.view(batch, channels, blocks, merge_size, blocks, merge_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, grid_size * grid_size, channels)
    )


class SlimSpatialTokenMixerSkip(nn.Module):
    """Width-96 attention-free Qwen token mixer used on one stage bypass.

    The three optical banks share one mixer.  Two separately gated residuals
    reproduce the useful part of the Caltech design: a 2-D depthwise spatial
    update followed by a channel MLP update.  No Transformer or attention is
    introduced.
    """

    def __init__(
        self,
        *,
        field_size: int,
        token_count: int,
        token_dim: int,
        optical_banks: int,
        width: int,
        expansion: float,
        kernel_size: int,
        dropout: float,
        spatial_gate_init: float,
        channel_gate_init: float,
        output_scale_init: float,
        output_scale_max: float,
        eps: float,
    ) -> None:
        super().__init__()
        if int(token_count) != 196 or int(field_size) != 224 or int(token_dim) != 224:
            raise ValueError("P09 locks the Qwen token geometry to 196x224 in a 224x224 field")
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError("Spatial mixer kernel size must be a positive odd number")
        hidden_width = int(round(int(width) * float(expansion)))
        self.field_size = int(field_size)
        self.token_count = int(token_count)
        self.token_dim = int(token_dim)
        self.optical_banks = int(optical_banks)
        self.width = int(width)
        self.eps = float(eps)
        self.output_scale_max = float(output_scale_max)

        self.input_norm = nn.LayerNorm(self.token_dim)
        self.input_adapter = nn.Linear(self.token_dim, self.width)

        self.spatial_norm = nn.LayerNorm(self.width)
        self.spatial_depthwise = nn.Conv2d(
            self.width,
            self.width,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=self.width,
            bias=False,
        )
        self.spatial_pointwise = nn.Linear(self.width, self.width)
        self.spatial_dropout = nn.Dropout(float(dropout))
        self.spatial_residual_logit = nn.Parameter(torch.logit(torch.tensor(float(spatial_gate_init))))

        self.channel_norm = nn.LayerNorm(self.width)
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.width, hidden_width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_width, self.width),
            nn.Dropout(float(dropout)),
        )
        self.channel_residual_logit = nn.Parameter(torch.logit(torch.tensor(float(channel_gate_init))))

        self.output_norm = nn.LayerNorm(self.width)
        self.output_adapter = nn.Linear(self.width, self.token_dim)
        # Preserve the identity bypass at initialization while allowing all
        # inner mixer parameters to receive gradients after the first update.
        nn.init.zeros_(self.output_adapter.weight)
        nn.init.zeros_(self.output_adapter.bias)
        self.output_scale_logit = nn.Parameter(
            _bounded_logit(float(output_scale_init), self.output_scale_max)
        )

    def spatial_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.spatial_residual_logit)

    def channel_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.channel_residual_logit)

    def transform_scale(self) -> torch.Tensor:
        return self.output_scale_max * torch.sigmoid(self.output_scale_logit)

    def long_skip_weight(self) -> torch.Tensor:
        return torch.zeros((), device=self.output_scale_logit.device)

    def forward(
        self,
        value: torch.Tensor,
        *,
        long_skip: torch.Tensor | None = None,
        disable_transform: bool = False,
        disable_long_skip: bool = False,
    ) -> torch.Tensor:
        del long_skip, disable_long_skip
        expected = (self.optical_banks, self.field_size, self.field_size)
        if value.ndim != 4 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(value.shape)}")
        base = rms_normalize(value.float(), self.eps)
        if disable_transform:
            return base

        batch = base.shape[0]
        tokens = base[:, :, : self.token_count, :].reshape(
            batch * self.optical_banks, self.token_count, self.token_dim
        )
        hidden = self.input_adapter(self.input_norm(tokens))

        spatial_input = self.spatial_norm(hidden)
        spatial_grid = qwen_tokens_to_grid(spatial_input)
        spatial_update = self.spatial_depthwise(spatial_grid)
        spatial_update = grid_to_qwen_tokens(spatial_update)
        spatial_update = self.spatial_dropout(self.spatial_pointwise(F.gelu(spatial_update)))
        hidden = hidden + self.spatial_gate().to(hidden) * spatial_update

        channel_update = self.channel_mlp(self.channel_norm(hidden))
        hidden = hidden + self.channel_gate().to(hidden) * channel_update
        delta_tokens = self.output_adapter(self.output_norm(hidden)).reshape(
            batch, self.optical_banks, self.token_count, self.token_dim
        )
        delta = F.pad(delta_tokens, (0, 0, 0, self.field_size - self.token_count))
        scale = self.transform_scale().to(base)
        return rms_normalize(F.relu(base + scale * delta), self.eps)


class QwenStemSlimMixerOpticalImageNetBackbone(nn.Module):
    """P09: frozen Qwen stem + eight optical stages + width-96 slim mixers."""

    def __init__(self, stem_checkpoint: str | Path, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.stem = StaticQwenPatchStem(stem_checkpoint)
        self.stem.requires_grad_(False)
        self.canvas_size = int(config.get("canvas_size", 224))
        self.optical_channels = int(config.get("optical_channels", 3))
        self.num_stages = int(config.get("num_stages", 8))
        self.token_dim = int(config.get("token_dim", 224))
        self.num_classes = int(config.get("num_classes", 1000))
        self.mixer_width = int(config.get("mixer_width", 96))
        if self.canvas_size != 224 or self.optical_channels != 3 or self.token_dim != 224:
            raise ValueError("P09 locks the optical tensor to three 224x224 banks")
        if self.num_stages != 8 or self.mixer_width != 96:
            raise ValueError("The comparable P09 experiment requires eight stages and mixer width 96")

        self.adapter = TokenAdapter(self.stem.hidden_size, self.token_dim)
        stages: list[nn.Module] = []
        for index in range(self.num_stages):
            stage = OpticalOEOStage(
                size=self.canvas_size,
                channels=self.optical_channels,
                wavelength_m=float(config.get("wavelength_m", 5.32e-7)),
                pixel_size_m=float(config.get("pixel_size_m", 1.6e-5)),
                distance_m=float(config.get("propagation_distance_m", 0.05)),
                phase_init_std=float(config.get("phase_init_std", 0.10)),
                layernorm_eps=float(config.get("layernorm_eps", 1.0e-5)),
                residual_mode="constrained",
                residual_main_init=float(config.get("optical_gate_init", 0.60)),
                residual_main_min=float(config.get("optical_gate_min", 0.50)),
                normalize_branch_rms=True,
                random_seed=int(config.get("seed", 2026)) + 1009 * index,
                electronic_skip_mode="identity",
                long_skip_enabled=False,
                long_skip_weight_init=0.0,
                long_skip_weight_max=0.0,
            )
            stage.electronic_skip = SlimSpatialTokenMixerSkip(
                field_size=self.canvas_size,
                token_count=self.stem.token_count,
                token_dim=self.token_dim,
                optical_banks=self.optical_channels,
                width=self.mixer_width,
                expansion=float(config.get("mixer_expansion", 2.0)),
                kernel_size=int(config.get("mixer_kernel_size", 3)),
                dropout=float(config.get("mixer_dropout", 0.10)),
                spatial_gate_init=float(config.get("mixer_spatial_gate_init", 0.10)),
                channel_gate_init=float(config.get("mixer_channel_gate_init", 0.10)),
                output_scale_init=float(config.get("residual_scale_init", 0.10)),
                output_scale_max=float(config.get("residual_scale_max", 0.25)),
                eps=float(config.get("layernorm_eps", 1.0e-5)),
            )
            stages.append(stage)
        self.stages = nn.ModuleList(stages)
        self.readout = TokenClassificationReadout(
            token_dim=self.token_dim,
            hidden_dim=int(config.get("head_hidden_dim", 448)),
            num_classes=self.num_classes,
        )
        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.stem.eval()
        return self

    def _images_01(self, images: torch.Tensor) -> torch.Tensor:
        return (images.float() * self.clip_std + self.clip_mean).clamp(0.0, 1.0)

    def optical_input(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            qwen_tokens = self.stem(self._images_01(images))
        tokens = self.adapter(qwen_tokens.detach())
        field = F.pad(tokens, (0, 0, 0, self.canvas_size - tokens.shape[1]))
        field = rms_normalize(field.unsqueeze(1).expand(-1, self.optical_channels, -1, -1), 1.0e-5)
        return field, qwen_tokens

    def forward_features(
        self, images: torch.Tensor, *, ablation: Ablation = "normal"
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if ablation not in {"normal", "optical_off", "phase_random", "electronic_skip_off"}:
            raise ValueError(f"Unsupported ablation: {ablation}")
        amplitude, _ = self.optical_input(images)
        outputs: list[torch.Tensor] = []
        for stage in self.stages:
            amplitude = stage(
                amplitude,
                phase_override=stage.random_phase if ablation == "phase_random" else None,
                optical_off=ablation == "optical_off",
                disable_electronic_skip=ablation == "electronic_skip_off",
            )
            outputs.append(amplitude)
        return amplitude, tuple(outputs)

    def forward(self, images: torch.Tensor, *, ablation: Ablation = "normal") -> torch.Tensor:
        final, _ = self.forward_features(images, ablation=ablation)
        return self.readout(final, self.stem.token_count)

    def phase_parameters(self):
        for stage in self.stages:
            yield stage.raw_phase

    def residual_parameters(self):
        phase_ids = {id(parameter) for parameter in self.phase_parameters()}
        for stage in self.stages:
            for parameter in stage.parameters():
                if id(parameter) not in phase_ids:
                    yield parameter

    def adapter_parameters(self):
        yield from self.adapter.parameters()

    def head_parameters(self):
        yield from self.readout.parameters()

    def phase_snapshot(self) -> torch.Tensor:
        return torch.stack([stage.phase().detach().cpu() for stage in self.stages])

    def phase_motion(self, initial: torch.Tensor) -> dict[str, Any]:
        current = self.phase_snapshot()
        if tuple(current.shape) != tuple(initial.shape):
            raise ValueError("Initial phase snapshot shape mismatch")
        displacement = torch.atan2(torch.sin(current - initial), torch.cos(current - initial)).abs()
        per_stage = displacement.flatten(1)
        return {
            "mean_absolute_rad": float(displacement.mean()),
            "median_absolute_rad": float(displacement.median()),
            "fraction_over_0p1_rad": float((displacement > 0.1).float().mean()),
            "per_stage_mean_absolute_rad": [float(value) for value in per_stage.mean(dim=1)],
            "per_stage_rms_rad": [float(value) for value in per_stage.square().mean(dim=1).sqrt()],
        }

    def optical_gates(self) -> list[float]:
        return [float(stage.residual.main_weight().detach().cpu()) for stage in self.stages]

    def electronic_skip_gates(self) -> list[dict[str, float]]:
        return [
            {
                "spatial": float(stage.electronic_skip.spatial_gate().detach().cpu()),
                "channel": float(stage.electronic_skip.channel_gate().detach().cpu()),
                "output_scale": float(stage.electronic_skip.transform_scale().detach().cpu()),
            }
            for stage in self.stages
        ]

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in self.state_dict().items() if not name.startswith("readout.")}

    def parameter_report(self) -> dict[str, Any]:
        optical = sum(parameter.numel() for parameter in self.phase_parameters())
        adapter = sum(parameter.numel() for parameter in self.adapter_parameters())
        residual = sum(parameter.numel() for parameter in self.residual_parameters())
        head = sum(parameter.numel() for parameter in self.head_parameters())
        backbone_trainable = optical + adapter + residual
        all_trainable = backbone_trainable + head
        frozen_stem = self.stem.parameter_report()["frozen_parameters"]
        return {
            "frozen_qwen_stem": self.stem.parameter_report(),
            "optical_phase_parameters": optical,
            "adapter_electronic_parameters": adapter,
            "residual_electronic_parameters": residual,
            "head_electronic_parameters": head,
            "backbone_trainable_parameters_excluding_task_head": backbone_trainable,
            "trainable_electronic_parameters": adapter + residual + head,
            "total_trainable_parameters": all_trainable,
            "optical_fraction_of_backbone_trainable": optical / backbone_trainable,
            "optical_fraction_of_all_trainable": optical / all_trainable,
            "optical_fraction_of_trainable": optical / all_trainable,
            "optical_fraction_including_frozen_stem": optical / (all_trainable + frozen_stem),
            "num_stages": self.num_stages,
            "latent_optical_banks": self.optical_channels,
            "token_count": self.stem.token_count,
            "token_dim": self.token_dim,
            "mixer_width": self.mixer_width,
            "mixer_instances": self.num_stages,
            "mixer_shared_across_banks_within_stage": True,
            "contains_electronic_transformer": False,
            "minimum_optical_gate": min(self.optical_gates()),
            "electronic_skip_gates": self.electronic_skip_gates(),
        }
