"""Qwen-conditioned adapter for a frozen one-step SD-Turbo decoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn


@dataclass(frozen=True)
class TurboAdapterConfig:
    pca_rank: int
    hidden_dim: int
    depth: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    teacher_batch_size: int
    sample_every_epochs: int
    early_stopping_patience: int

    def validate(self) -> None:
        if min(self.pca_rank, self.hidden_dim, self.depth, self.batch_size, self.epochs) <= 0:
            raise ValueError("Turbo adapter dimensions and training lengths must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")


def load_turbo_adapter_config(path: str | Path) -> TurboAdapterConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model, training = raw["model"], raw["training"]
    config = TurboAdapterConfig(
        pca_rank=int(model["pca_rank"]),
        hidden_dim=int(model["hidden_dim"]),
        depth=int(model["depth"]),
        dropout=float(model["dropout"]),
        batch_size=int(training["batch_size"]),
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        num_workers=int(training["num_workers"]),
        teacher_batch_size=int(training["teacher_batch_size"]),
        sample_every_epochs=int(training["sample_every_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
    )
    config.validate()
    return config


class ResidualMLP(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * width, width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class QwenTurboConditionAdapter(nn.Module):
    """Map one pooled Qwen vector onto a PCA manifold of CLIP token states."""

    def __init__(
        self,
        text_dim: int,
        config: TurboAdapterConfig,
        teacher_mean: torch.Tensor,
        basis: torch.Tensor,
        coefficient_mean: torch.Tensor,
        coefficient_std: torch.Tensor,
        token_count: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        if basis.shape != (config.pca_rank, token_count * condition_dim):
            raise ValueError("PCA basis shape does not match adapter contract")
        self.token_count = int(token_count)
        self.condition_dim = int(condition_dim)
        self.register_buffer("teacher_mean", teacher_mean.float())
        self.register_buffer("basis", basis.float())
        self.register_buffer("coefficient_mean", coefficient_mean.float())
        self.register_buffer("coefficient_std", coefficient_std.float().clamp_min(1e-6))
        layers: list[nn.Module] = [nn.LayerNorm(text_dim), nn.Linear(text_dim, config.hidden_dim), nn.SiLU()]
        layers.extend(ResidualMLP(config.hidden_dim, config.dropout) for _ in range(config.depth))
        layers.extend((nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.pca_rank)))
        self.predictor = nn.Sequential(*layers)

    def forward(self, qwen_text: torch.Tensor) -> torch.Tensor:
        """Return standardized PCA coefficients for a Qwen text feature."""

        return self.predictor(qwen_text.float())

    def condition_from_coefficients(self, standardized: torch.Tensor) -> torch.Tensor:
        coefficients = standardized.float() * self.coefficient_std + self.coefficient_mean
        flattened = self.teacher_mean + coefficients @ self.basis
        return flattened.view(-1, self.token_count, self.condition_dim)

    def condition(self, qwen_text: torch.Tensor) -> torch.Tensor:
        return self.condition_from_coefficients(self(qwen_text))

    def architecture_report(self) -> dict[str, Any]:
        return {
            "variant": "qwen_sd_turbo_one_step",
            "qwen_is_inference_conditioner": True,
            "decoder": "frozen SD-Turbo UNet + frozen VAE",
            "inference_iterations": 1,
            "unet_calls": 1,
            "vae_decoder_calls": 1,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "condition_tokens": self.token_count,
            "condition_dim": self.condition_dim,
            "pca_rank": self.basis.shape[0],
        }


__all__ = ["QwenTurboConditionAdapter", "TurboAdapterConfig", "load_turbo_adapter_config"]
