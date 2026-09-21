"""Parallel optical mid-block for the compact one-step Turbo baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .modeling import CompactFourierOptics, ScaleMatchedFusion


@dataclass(frozen=True)
class OpticalTurboConfig:
    optical_width: int
    grid: int
    experts: int
    top_k: int
    alpha_initial: float
    alpha_minimum: float
    alpha_maximum: float
    rms_epsilon: float
    batch_size: int
    epochs: int
    learning_rate: float
    phase_learning_rate: float
    weight_decay: float
    detail_weight: float
    num_workers: int

    def validate(self) -> None:
        if min(self.optical_width, self.grid, self.experts, self.top_k) <= 0:
            raise ValueError("Optical dimensions must be positive")
        if self.top_k > self.experts:
            raise ValueError("top_k cannot exceed experts")
        if not 0 <= self.alpha_minimum < self.alpha_initial < self.alpha_maximum <= 1:
            raise ValueError("Fusion alpha must start strictly inside its range")
        if min(self.batch_size, self.epochs, self.learning_rate, self.phase_learning_rate) <= 0:
            raise ValueError("Training settings must be positive")
        if min(self.weight_decay, self.detail_weight, self.num_workers) < 0:
            raise ValueError("Regularization settings must be non-negative")


def load_optical_turbo_config(path: str | Path) -> OpticalTurboConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model, fusion, training, loss = (
        raw["model"], raw["fusion"], raw["training"], raw.get("loss", {})
    )
    config = OpticalTurboConfig(
        optical_width=int(model["optical_width"]),
        grid=int(model["grid"]),
        experts=int(model["experts"]),
        top_k=int(model["top_k"]),
        alpha_initial=float(fusion["alpha_initial"]),
        alpha_minimum=float(fusion["alpha_minimum"]),
        alpha_maximum=float(fusion["alpha_maximum"]),
        rms_epsilon=float(fusion["rms_epsilon"]),
        batch_size=int(training["batch_size"]),
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        phase_learning_rate=float(training["phase_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        detail_weight=float(loss.get("detail_weight", 0.0)),
        num_workers=int(training["num_workers"]),
    )
    config.validate()
    return config


class ParallelOpticalMidBlock(nn.Module):
    """Keep the complete electronic mid block and add a parallel optical path.

    Both branches consume ``hidden_states`` directly. The optical path never
    consumes the electronic branch output; only their final outputs are fused.
    """

    def __init__(
        self,
        electronic: nn.Module,
        *,
        channels: int,
        timestep_dim: int,
        condition_dim: int,
        config: OpticalTurboConfig,
    ) -> None:
        super().__init__()
        self.electronic = electronic
        self.channels = int(channels)
        self.grid = int(config.grid)
        width = int(config.optical_width)
        self.input_norm = nn.LayerNorm(channels)
        self.input_projection = nn.Linear(channels, width)
        self.timestep_projection = nn.Linear(timestep_dim, width)
        self.condition_projection = nn.Linear(condition_dim, width)
        self.optical = CompactFourierOptics(
            width, config.grid, experts=config.experts, top_k=config.top_k
        )
        self.expert_gate = nn.Parameter(torch.tensor(-1.5))
        self.global_gate = nn.Parameter(torch.tensor(-1.5))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, channels)
        self.output_gate = nn.Parameter(torch.tensor(-2.0))
        self.fusion = ScaleMatchedFusion(
            config.alpha_initial,
            config.alpha_minimum,
            config.alpha_maximum,
            config.rms_epsilon,
        )
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

    def optical_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("electronic."):
                yield name, parameter

    def freeze_electronic(self) -> None:
        self.electronic.requires_grad_(False)
        for _, parameter in self.optical_parameters():
            parameter.requires_grad_(True)

    def optical_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("electronic.")
        }

    def load_optical_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        current.update(state)
        self.load_state_dict(current)

    def _optical_branch(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = hidden_states.shape
        if channels != self.channels or (height, width) != (self.grid, self.grid):
            raise ValueError(
                f"Expected [B,{self.channels},{self.grid},{self.grid}], got {tuple(hidden_states.shape)}"
            )
        tokens = hidden_states.flatten(2).transpose(1, 2)
        tokens = self.input_projection(self.input_norm(tokens.float()))
        tokens = tokens + self.timestep_projection(temb.float())[:, None]
        pooled_condition = encoder_hidden_states.float().mean(dim=1)
        tokens = F.silu(tokens + self.condition_projection(pooled_condition)[:, None])
        expert = self.optical.expert(tokens)
        stage1 = tokens + torch.sigmoid(self.expert_gate) * expert
        global_value = self.optical.global_block(stage1)
        stage2 = stage1 + torch.sigmoid(self.global_gate) * global_value
        delta = self.output_projection(self.output_norm(stage2))
        delta = delta.transpose(1, 2).reshape(batch, channels, height, width)
        return hidden_states.float() + torch.sigmoid(self.output_gate) * delta

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if temb is None or encoder_hidden_states is None:
            raise ValueError("Parallel optical mid block requires timestep and text conditioning")
        electronic = self.electronic(
            hidden_states,
            temb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            encoder_attention_mask=encoder_attention_mask,
        )
        optical = self._optical_branch(hidden_states, temb, encoder_hidden_states)
        return self.fusion(electronic, optical.to(electronic.dtype))

    def architecture_report(self) -> dict[str, Any]:
        optical_parameters = sum(
            parameter.numel() for _, parameter in self.optical_parameters()
        )
        return {
            "variant": "qwen_bksdm_v2_tiny_parallel_optical_mid_v0",
            "electronic_mid_is_retained": True,
            "electronic_and_optical_are_parallel": True,
            "optical_backend": "compact_fft_simulation",
            "spatial_grid": [self.grid, self.grid],
            "valid_optical_tokens": self.grid**2,
            "optical_trainable_parameters": optical_parameters,
            "fusion": {
                "alpha": float(self.fusion.alpha.detach()),
                "minimum": self.fusion.minimum,
                "maximum": self.fusion.maximum,
            },
        }

def attach_parallel_optical_mid(unet: nn.Module, config: OpticalTurboConfig) -> ParallelOpticalMidBlock:
    electronic = unet.mid_block
    channels = int(unet.config.block_out_channels[-1])
    condition_dim = int(unet.config.cross_attention_dim)
    timestep_dim = int(unet.time_embedding.linear_2.out_features)
    wrapper = ParallelOpticalMidBlock(
        electronic,
        channels=channels,
        timestep_dim=timestep_dim,
        condition_dim=condition_dim,
        config=config,
    )
    unet.mid_block = wrapper
    return wrapper


__all__ = [
    "OpticalTurboConfig",
    "ParallelOpticalMidBlock",
    "attach_parallel_optical_mid",
    "load_optical_turbo_config",
]
