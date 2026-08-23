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

from .stem import StaticQwenPatchStem


Ablation = Literal["normal", "optical_off", "phase_random", "electronic_skip_off"]

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class TokenAdapter(nn.Module):
    """One small shared projection; no attention or electronic token mixer."""

    def __init__(self, input_dim: int = 1024, optical_dim: int = 224) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(input_dim))
        self.projection = nn.Linear(int(input_dim), int(optical_dim))
        self.activation = nn.Softplus()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.activation(self.projection(self.norm(tokens)))
        return value / value.square().mean(dim=-1, keepdim=True).add(1.0e-6).sqrt()


class TokenClassificationReadout(nn.Module):
    """Convex bank fusion, token pooling and a bounded ImageNet MLP head."""

    def __init__(self, *, token_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.bank_logits = nn.Parameter(torch.zeros(3))
        self.token_norm = nn.LayerNorm(int(token_dim))
        self.classifier = nn.Sequential(
            nn.Linear(2 * int(token_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, fields: torch.Tensor, token_count: int) -> torch.Tensor:
        if fields.ndim != 4 or fields.shape[1] != 3:
            raise ValueError(f"Expected three latent optical banks, got {tuple(fields.shape)}")
        weights = torch.softmax(self.bank_logits, dim=0).to(fields)
        tokens = (fields[:, :, : int(token_count), :] * weights.view(1, 3, 1, 1)).sum(dim=1)
        tokens = self.token_norm(tokens)
        descriptor = torch.cat((tokens.mean(dim=1), tokens.amax(dim=1)), dim=-1)
        return self.classifier(descriptor)


class QwenStemOpticalImageNetBackbone(nn.Module):
    """Frozen Qwen patch tokens -> adapter -> three-bank eight-stage optics."""

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
        if self.canvas_size != 224 or self.optical_channels != 3:
            raise ValueError("The locked P08 architecture requires a 224 canvas and three latent banks")
        if self.token_dim != 224 or self.stem.token_count > self.canvas_size:
            raise ValueError("Stem token count/token dimension does not fit the 224x224 optical field")
        self.adapter = TokenAdapter(self.stem.hidden_size, self.token_dim)
        self.stages = nn.ModuleList(
            [
                OpticalOEOStage(
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
                    electronic_skip_mode="lowres",
                    electronic_skip_hidden_channels=int(config.get("residual_hidden_channels", 64)),
                    electronic_skip_downsample_factor=int(config.get("residual_downsample_factor", 7)),
                    electronic_skip_scale_init=float(config.get("residual_scale_init", 0.10)),
                    electronic_skip_scale_max=float(config.get("residual_scale_max", 0.25)),
                    long_skip_enabled=False,
                    long_skip_weight_init=0.0,
                    long_skip_weight_max=0.0,
                )
                for index in range(self.num_stages)
            ]
        )
        self.readout = TokenClassificationReadout(
            token_dim=self.token_dim,
            hidden_dim=int(config.get("head_hidden_dim", 448)),
            num_classes=self.num_classes,
        )
        self.register_buffer(
            "clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.stem.eval()
        return self

    def _images_01(self, clip_normalized_images: torch.Tensor) -> torch.Tensor:
        value = clip_normalized_images.float() * self.clip_std + self.clip_mean
        return value.clamp(0.0, 1.0)

    def optical_input(self, clip_normalized_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The stem is frozen and its output is the only Qwen signal. Gradients
        # start at the small 1024->224 adapter and never enter a Qwen Transformer.
        with torch.no_grad():
            qwen_tokens = self.stem(self._images_01(clip_normalized_images))
        tokens = self.adapter(qwen_tokens.detach())
        pad_rows = self.canvas_size - tokens.shape[1]
        if pad_rows < 0:
            raise RuntimeError("Qwen token count exceeds the optical canvas")
        field = F.pad(tokens, (0, 0, 0, pad_rows))
        field = rms_normalize(field.unsqueeze(1).expand(-1, self.optical_channels, -1, -1), 1.0e-5)
        return field, qwen_tokens

    def forward_features(
        self,
        images: torch.Tensor,
        *,
        ablation: Ablation = "normal",
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

    def forward(
        self,
        images: torch.Tensor,
        *,
        ablation: Ablation = "normal",
    ) -> torch.Tensor:
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

    def parameter_report(self) -> dict[str, Any]:
        optical = sum(parameter.numel() for parameter in self.phase_parameters())
        adapter = sum(parameter.numel() for parameter in self.adapter_parameters())
        residual = sum(parameter.numel() for parameter in self.residual_parameters())
        head = sum(parameter.numel() for parameter in self.head_parameters())
        trainable = optical + adapter + residual + head
        frozen_stem = self.stem.parameter_report()["frozen_parameters"]
        return {
            "frozen_qwen_stem": self.stem.parameter_report(),
            "optical_phase_parameters": optical,
            "adapter_electronic_parameters": adapter,
            "residual_electronic_parameters": residual,
            "head_electronic_parameters": head,
            "trainable_electronic_parameters": adapter + residual + head,
            "total_trainable_parameters": trainable,
            "optical_fraction_of_trainable": optical / trainable,
            "optical_fraction_including_frozen_stem": optical / (trainable + frozen_stem),
            "num_stages": self.num_stages,
            "latent_optical_banks": self.optical_channels,
            "token_count": self.stem.token_count,
            "token_dim": self.token_dim,
            "contains_electronic_transformer": False,
            "minimum_optical_gate": min(self.optical_gates()),
        }
