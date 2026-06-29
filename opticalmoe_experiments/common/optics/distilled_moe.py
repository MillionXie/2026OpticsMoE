from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_layout import ExpertLayout
from .optical_models import ASGlobalRouterMoEClassifier


class GridPoolFeatureDetector(nn.Module):
    """Convert detector-plane intensity into a fixed optical feature vector."""

    def __init__(self, grid_size: int = 16, pooling: str = "sum", normalize_total_energy: bool = True) -> None:
        super().__init__()
        if pooling not in {"sum", "mean"}:
            raise ValueError("feature detector pooling must be 'sum' or 'mean'.")
        self.grid_size = int(grid_size)
        self.pooling = str(pooling)
        self.normalize_total_energy = bool(normalize_total_energy)
        self.feature_dim = self.grid_size * self.grid_size

    def forward(self, detector_intensity: torch.Tensor) -> torch.Tensor:
        intensity = torch.as_tensor(detector_intensity).float()
        if intensity.ndim == 3:
            intensity = intensity.unsqueeze(1)
        if intensity.ndim != 4 or intensity.shape[1] != 1:
            raise ValueError("detector_intensity must have shape [B,H,W] or [B,1,H,W].")
        height, width = intensity.shape[-2:]
        y_edges = torch.div(
            torch.arange(self.grid_size + 1, device=intensity.device) * height,
            self.grid_size,
            rounding_mode="floor",
        ).long()
        x_edges = torch.div(
            torch.arange(self.grid_size + 1, device=intensity.device) * width,
            self.grid_size,
            rounding_mode="floor",
        ).long()
        integral = F.pad(intensity.cumsum(dim=-2).cumsum(dim=-1), (1, 0, 1, 0))
        y0, y1 = y_edges[:-1], y_edges[1:]
        x0, x1 = x_edges[:-1], x_edges[1:]
        pooled = (
            integral[:, :, y1[:, None], x1[None, :]]
            - integral[:, :, y0[:, None], x1[None, :]]
            - integral[:, :, y1[:, None], x0[None, :]]
            + integral[:, :, y0[:, None], x0[None, :]]
        )
        if self.pooling == "mean":
            areas = (y1 - y0).float()[:, None] * (x1 - x0).float()[None, :]
            pooled = pooled / areas.clamp_min(1.0).view(1, 1, self.grid_size, self.grid_size)
        feature = pooled.flatten(1)
        if self.normalize_total_energy:
            feature = feature / feature.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return feature


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(input_dim: int, output_dim: int, cfg: Dict) -> nn.Module:
    hidden_layers = int(cfg.get("hidden_layers", 1))
    hidden_dim = int(cfg.get("hidden_dim", input_dim))
    dropout = float(cfg.get("dropout", 0.0))
    if hidden_layers <= 0:
        return nn.Linear(int(input_dim), int(output_dim))
    layers = []
    current = int(input_dim)
    for _ in range(hidden_layers):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(_activation(cfg.get("activation", "gelu")))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current = hidden_dim
    layers.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*layers)


class DetectorFeatureASGlobalRouterMoEClassifier(nn.Module):
    """AS global-router OpticalMoE with a detector-intensity feature classifier."""

    def __init__(
        self,
        num_classes: int,
        layout: ExpertLayout,
        feature_detector_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        **optical_backbone_kwargs,
    ) -> None:
        super().__init__()
        feature_cfg = dict(feature_detector_config or {})
        classifier_cfg = dict(classifier_config or {})
        self.num_classes = int(num_classes)
        self.layout = layout
        self.optical_backbone = ASGlobalRouterMoEClassifier(
            num_classes=self.num_classes,
            layout=layout,
            detector_size=1,
            detector_layout="grid",
            normalize_detector_energy=True,
            readout_type="optical_only",
            readout_input_norm="none",
            readout_norm_affine=False,
            **optical_backbone_kwargs,
        )
        self.feature_detector = GridPoolFeatureDetector(
            grid_size=int(feature_cfg.get("grid_size", 16)),
            pooling=str(feature_cfg.get("pooling", "sum")),
            normalize_total_energy=bool(feature_cfg.get("normalize_total_energy", True)),
        )
        configured_dim = int(feature_cfg.get("feature_dim", self.feature_detector.feature_dim))
        if configured_dim != self.feature_detector.feature_dim:
            raise ValueError(
                f"feature_detector.feature_dim={configured_dim} must equal grid_size^2={self.feature_detector.feature_dim}."
            )
        self.optical_feature_dim = configured_dim
        self.classifier = _mlp(self.optical_feature_dim, self.num_classes, classifier_cfg)
        self.feature_detector_config = {
            "type": "grid_pool",
            "grid_size": self.feature_detector.grid_size,
            "pooling": self.feature_detector.pooling,
            "normalize_total_energy": self.feature_detector.normalize_total_energy,
            "feature_dim": self.optical_feature_dim,
        }

    @property
    def expert_layers(self):
        return self.optical_backbone.expert_layers

    @property
    def global_fc(self):
        return self.optical_backbone.global_fc

    @property
    def prompt(self):
        return self.optical_backbone.prompt

    @property
    def expert_masks(self):
        return self.optical_backbone.expert_masks

    def _forward_features(self, images: torch.Tensor, return_intermediates: bool = False):
        optical_output = self.optical_backbone.forward_to_detector(images, return_intermediates=return_intermediates)
        if return_intermediates:
            detector_field, intermediates = optical_output
        else:
            detector_field = optical_output
            intermediates = None
        detector_intensity = torch.abs(detector_field).square()
        optical_feature = self.feature_detector(detector_intensity)
        logits = self.classifier(optical_feature)
        if not return_intermediates:
            return logits, optical_feature
        intermediates["detector_intensity"] = detector_intensity
        intermediates["optical_feature"] = optical_feature
        intermediates["prompt_weights"] = intermediates.get("normalized_prompt_powers")
        intermediates["logits"] = logits
        return logits, optical_feature, intermediates

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        return self._forward_features(images, return_intermediates=return_intermediates)

    def optical_parameter_count(self) -> int:
        return int(self.optical_backbone.optical_parameter_count())

    def prompt_parameter_count(self) -> int:
        return int(self.optical_backbone.prompt_parameter_count())

    def classifier_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.classifier.parameters())

    def electronic_parameter_count(self) -> int:
        return int(self.classifier_parameter_count())

    def total_parameter_count(self) -> int:
        return int(self.optical_parameter_count() + self.prompt_parameter_count() + self.classifier_parameter_count())

    def set_phase_dropout_active(self, active: bool) -> None:
        self.optical_backbone.set_phase_dropout_active(active)


class FeatureDistilledASGlobalRouterMoEClassifier(DetectorFeatureASGlobalRouterMoEClassifier):
    """Detector-feature OpticalMoE with a training-time teacher projector."""

    def __init__(
        self,
        num_classes: int,
        teacher_feature_dim: int,
        layout: ExpertLayout,
        feature_detector_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        projector_config: Optional[Dict] = None,
        **optical_backbone_kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            layout=layout,
            feature_detector_config=feature_detector_config,
            classifier_config=classifier_config,
            **optical_backbone_kwargs,
        )
        self.teacher_feature_dim = int(teacher_feature_dim)
        self.projector = _mlp(self.optical_feature_dim, self.teacher_feature_dim, dict(projector_config or {}))

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        feature_output = self._forward_features(images, return_intermediates=return_intermediates)
        if return_intermediates:
            logits, optical_feature, intermediates = feature_output
        else:
            logits, optical_feature = feature_output
        projected_feature = F.normalize(self.projector(optical_feature), dim=-1)
        if not return_intermediates:
            return logits, optical_feature, projected_feature
        intermediates["projected_feature"] = projected_feature
        return logits, optical_feature, projected_feature, intermediates

    def projector_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.projector.parameters())

    def total_parameter_count(self) -> int:
        return int(super().total_parameter_count() + self.projector_parameter_count())
