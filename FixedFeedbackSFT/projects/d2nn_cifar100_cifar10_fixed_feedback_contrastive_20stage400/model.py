from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .optics import FeedbackMode, OpticalOEOStage
from .settings import OpticalConfig


class EmbeddingReadout(nn.Module):
    """Signed electronic embedding readout; dropout is active only during training."""

    def __init__(self, pool_size: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.pool_size = int(pool_size)
        features = self.pool_size * self.pool_size
        self.feature_norm = nn.LayerNorm(features, elementwise_affine=True)
        self.dropout = nn.Dropout(float(dropout))
        self.projection = nn.Linear(features, int(embedding_dim))

    def forward(self, amplitude: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(amplitude.unsqueeze(1), (self.pool_size, self.pool_size)).flatten(1)
        embedding = self.projection(self.dropout(self.feature_norm(pooled)))
        norm = embedding.norm(dim=-1, keepdim=True)
        if torch.any(~torch.isfinite(norm)) or torch.any(norm <= 1e-12):
            raise FloatingPointError("Embedding readout produced a non-finite or zero norm")
        return F.normalize(embedding, dim=-1, eps=1e-12)


class OpticalEmbeddingNetwork(nn.Module):
    def __init__(self, config: OpticalConfig) -> None:
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList(
            [
                OpticalOEOStage(
                    size=config.canvas_size,
                    wavelength_m=config.wavelength_m,
                    pixel_size_m=config.pixel_size_m,
                    distance_m=config.propagation_distance_m,
                    layernorm_eps=config.layernorm_eps,
                    residual_main_init=config.residual_main_init,
                    residual_skip_init=config.residual_skip_init,
                )
                for _ in range(config.num_stages)
            ]
        )
        self.readout = EmbeddingReadout(
            config.readout_pool_size,
            config.embedding_dim,
            config.embedding_dropout,
        )
        self.feedback_mode: FeedbackMode = "bp"

    def prepare_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"Expected grayscale [B,1,H,W], got {tuple(images.shape)}")
        resized = F.interpolate(
            images.float(),
            size=(self.config.canvas_size, self.config.canvas_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return resized[:, 0].clamp(0.0, 1.0)

    def forward(self, images: torch.Tensor, *, return_intermediates: bool = False):
        amplitude = self.prepare_amplitude(images)
        input_amplitude = amplitude
        details: list[dict[str, torch.Tensor]] = []
        for stage in self.stages:
            if return_intermediates:
                amplitude, stage_details = stage(amplitude, return_details=True)
                details.append(stage_details)
            else:
                amplitude = stage(amplitude)
        embedding = self.readout(amplitude)
        if not return_intermediates:
            return embedding
        return embedding, {
            "input_amplitude": input_amplitude,
            "stages": details,
            "final_amplitude": amplitude,
            "embedding": embedding,
        }

    def phase_stack(self) -> torch.Tensor:
        return torch.stack([stage.phase() for stage in self.stages], dim=0)

    def residual_weights(self) -> torch.Tensor:
        return torch.stack([stage.residual.weights() for stage in self.stages], dim=0)

    def snapshot_feedback_phases(self) -> torch.Tensor:
        return self.phase_stack().detach().cpu()

    def configure_feedback(
        self,
        mode: FeedbackMode,
        *,
        pretrained_phases: torch.Tensor | None = None,
        random_seed: int = 0,
    ) -> None:
        if mode == "bp":
            for stage in self.stages:
                stage.set_feedback("bp")
        elif mode == "fa_pretrained":
            expected = (len(self.stages), self.config.canvas_size, self.config.canvas_size)
            if pretrained_phases is None or tuple(pretrained_phases.shape) != expected:
                raise ValueError(f"pretrained_phases must have shape {expected}")
            for index, stage in enumerate(self.stages):
                stage.set_feedback(mode, pretrained_phases[index])
        elif mode == "fa_random":
            generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
            for stage in self.stages:
                phase = torch.rand(self.config.canvas_size, self.config.canvas_size, generator=generator) * (
                    2.0 * math.pi
                )
                stage.set_feedback(mode, phase)
        else:
            raise ValueError(f"Unsupported feedback mode: {mode}")
        self.feedback_mode = mode

    def phase_parameters(self) -> Iterable[nn.Parameter]:
        for stage in self.stages:
            yield stage.raw_phase

    def electronic_parameters(self) -> Iterable[nn.Parameter]:
        for stage in self.stages:
            yield stage.residual.logits
        yield from self.readout.parameters()

    def parameter_report(self) -> dict[str, int]:
        phase = sum(parameter.numel() for parameter in self.phase_parameters())
        electronic = sum(parameter.numel() for parameter in self.electronic_parameters())
        return {"phase": phase, "electronic": electronic, "total": phase + electronic}
