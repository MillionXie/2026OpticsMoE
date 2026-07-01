from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_layout import ExpertLayout
from .optical_models import ASGlobalRouterMoEClassifier


class GridPoolFeatureDetector(nn.Module):
    """Pool a physical camera intensity crop into a fixed grid feature."""

    def __init__(self, grid_size: int = 30, pooling: str = "sum", normalize_total_energy: bool = False) -> None:
        super().__init__()
        if pooling not in {"sum", "mean"}:
            raise ValueError("feature detector pooling must be 'sum' or 'mean'.")
        self.grid_size = int(grid_size)
        self.pooling = str(pooling)
        self.normalize_total_energy = bool(normalize_total_energy)
        self.feature_dim = self.grid_size * self.grid_size

    def forward(self, camera_intensity: torch.Tensor) -> torch.Tensor:
        intensity = torch.as_tensor(camera_intensity).float()
        if intensity.ndim == 3:
            intensity = intensity.unsqueeze(1)
        if intensity.ndim != 4 or intensity.shape[1] != 1:
            raise ValueError("camera_intensity must have shape [B,H,W] or [B,1,H,W].")
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


class FeaturePreprocess(nn.Module):
    def __init__(self, feature_dim: int, config: Optional[Dict] = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.norm_name = str(cfg.get("norm", "layernorm")).lower()
        self.activation_name = str(cfg.get("activation", "gelu")).lower()
        self.norm_affine = bool(cfg.get("norm_affine", True))
        if self.norm_name == "none":
            self.norm = nn.Identity()
        elif self.norm_name == "layernorm":
            self.norm = nn.LayerNorm(int(feature_dim), elementwise_affine=self.norm_affine)
        else:
            raise ValueError("feature_preprocess.norm must be none or layernorm.")
        if self.activation_name == "none":
            self.activation = nn.Identity()
        elif self.activation_name in {"gelu", "relu", "silu"}:
            self.activation = _activation(self.activation_name)
        else:
            raise ValueError("feature_preprocess.activation must be none, gelu, relu, or silu.")
        self.config = {
            "norm": self.norm_name,
            "norm_affine": self.norm_affine,
            "activation": self.activation_name,
        }

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(feature))


def _resolve_auto_dim(value, automatic: int, field: str) -> int:
    if value in {None, "auto_feature_dim", "auto_teacher_dim"}:
        return int(automatic)
    resolved = int(value)
    if resolved != int(automatic):
        raise ValueError(f"{field}={resolved} does not match resolved dimension {automatic}.")
    return resolved


def build_projector(input_dim: int, output_dim: int, config: Optional[Dict] = None):
    cfg = dict(config or {})
    projector_type = str(cfg.get("type", "mlp")).lower()
    resolved_input = _resolve_auto_dim(cfg.get("input_dim", "auto_feature_dim"), input_dim, "projector.input_dim")
    resolved_output = _resolve_auto_dim(cfg.get("output_dim", "auto_teacher_dim"), output_dim, "projector.output_dim")
    output_l2_normalize = bool(cfg.get("output_l2_normalize", True))
    if projector_type == "linear":
        module = nn.Linear(resolved_input, resolved_output)
    elif projector_type == "mlp":
        hidden_layers = int(cfg.get("hidden_layers", 1))
        if hidden_layers < 1:
            raise ValueError("MLP projector.hidden_layers must be at least 1; use projector.type=linear otherwise.")
        hidden_dim = int(cfg.get("hidden_dim", 512))
        hidden_norm = str(cfg.get("hidden_norm", "none")).lower()
        if hidden_norm not in {"none", "layernorm"}:
            raise ValueError("projector.hidden_norm must be none or layernorm.")
        dropout = float(cfg.get("dropout", 0.1))
        layers = []
        current = resolved_input
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current, hidden_dim))
            if hidden_norm == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(_activation(cfg.get("activation", "gelu")))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        layers.append(nn.Linear(current, resolved_output))
        module = nn.Sequential(*layers)
    else:
        raise ValueError("projector.type must be linear or mlp.")
    resolved = {
        "type": projector_type,
        "input_dim": resolved_input,
        "output_dim": resolved_output,
        "hidden_layers": int(cfg.get("hidden_layers", 1)) if projector_type == "mlp" else 0,
        "hidden_dim": int(cfg.get("hidden_dim", 512)) if projector_type == "mlp" else None,
        "hidden_norm": str(cfg.get("hidden_norm", "none")) if projector_type == "mlp" else "none",
        "activation": str(cfg.get("activation", "gelu")) if projector_type == "mlp" else "none",
        "dropout": float(cfg.get("dropout", 0.1)) if projector_type == "mlp" else 0.0,
        "output_l2_normalize": output_l2_normalize,
    }
    return module, resolved


class DetectorFeatureASGlobalRouterMoEClassifier(nn.Module):
    """AS OpticalMoE whose features come only from the physical camera crop."""

    def __init__(
        self,
        num_classes: int,
        layout: ExpertLayout,
        feature_detector_config: Optional[Dict] = None,
        feature_preprocess_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        **optical_backbone_kwargs,
    ) -> None:
        super().__init__()
        feature_cfg = dict(feature_detector_config or {})
        classifier_cfg = dict(classifier_config or {})
        self.num_classes = int(num_classes)
        self.layout = layout
        self.camera_region = [
            int(layout.active_window_aperture.y0),
            int(layout.active_window_aperture.y1),
            int(layout.active_window_aperture.x0),
            int(layout.active_window_aperture.x1),
        ]
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
        source_region = str(feature_cfg.get("source_region", "camera_active_window"))
        if source_region != "camera_active_window":
            raise ValueError("feature_detector.source_region must be camera_active_window.")
        grid_size = int(feature_cfg.get("grid_size", 30))
        if grid_size != 30:
            raise ValueError("Camera-aware feature_detector.grid_size must be 30.")
        self.feature_detector = GridPoolFeatureDetector(
            grid_size=grid_size,
            pooling=str(feature_cfg.get("pooling", "sum")),
            normalize_total_energy=bool(feature_cfg.get("normalize_total_energy", False)),
        )
        configured_dim = int(feature_cfg.get("feature_dim", self.feature_detector.feature_dim))
        if configured_dim != 900 or configured_dim != self.feature_detector.feature_dim:
            raise ValueError(
                f"feature_detector.feature_dim={configured_dim} must equal 30^2=900."
            )
        self.optical_feature_dim = configured_dim
        self.camera_feature_dim = configured_dim
        self.feature_preprocess = FeaturePreprocess(self.camera_feature_dim, feature_preprocess_config)
        self.classifier = _mlp(self.camera_feature_dim, self.num_classes, classifier_cfg)
        self.feature_detector_config = {
            "type": "grid_pool",
            "source_region": source_region,
            "grid_size": self.feature_detector.grid_size,
            "pooling": self.feature_detector.pooling,
            "normalize_total_energy": self.feature_detector.normalize_total_energy,
            "feature_dim": self.camera_feature_dim,
        }
        self.feature_preprocess_config = dict(self.feature_preprocess.config)

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

    def _camera_features(self, images: torch.Tensor, return_intermediates: bool = False):
        optical_output = self.optical_backbone.forward_to_detector(images, return_intermediates=return_intermediates)
        if return_intermediates:
            detector_field, intermediates = optical_output
        else:
            detector_field = optical_output
            intermediates = None
        detector_intensity = torch.abs(detector_field).square()
        y0, y1, x0, x1 = self.camera_region
        camera_intensity = detector_intensity[..., y0:y1, x0:x1]
        total_energy = detector_intensity.sum(dim=(-2, -1))
        camera_energy = camera_intensity.sum(dim=(-2, -1))
        outside_ratio = (total_energy - camera_energy).clamp_min(0.0) / total_energy.clamp_min(1e-12)
        camera_feature_raw = self.feature_detector(camera_intensity)
        camera_feature_processed = self.feature_preprocess(camera_feature_raw)
        if not return_intermediates:
            return camera_feature_raw, camera_feature_processed, outside_ratio
        sparsity = (camera_feature_raw.abs() <= 1e-12).float().mean(dim=1)
        intermediates.update(
            {
                "detector_intensity": detector_intensity,
                "camera_intensity": camera_intensity,
                "camera_region": list(self.camera_region),
                "outside_camera_energy_ratio": outside_ratio,
                "camera_feature_raw": camera_feature_raw,
                "camera_feature_processed": camera_feature_processed,
                "camera_feature_mean": camera_feature_raw.mean(dim=1),
                "camera_feature_std": camera_feature_raw.std(dim=1, unbiased=False),
                "camera_feature_min": camera_feature_raw.min(dim=1).values,
                "camera_feature_max": camera_feature_raw.max(dim=1).values,
                "camera_feature_sparsity": sparsity,
                "optical_feature": camera_feature_processed,
                "prompt_weights": intermediates.get("normalized_prompt_powers"),
            }
        )
        return camera_feature_raw, camera_feature_processed, outside_ratio, intermediates

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        outputs = self._camera_features(images, return_intermediates=return_intermediates)
        if return_intermediates:
            camera_raw, camera_processed, _outside_ratio, intermediates = outputs
        else:
            camera_raw, camera_processed, _outside_ratio = outputs
        logits = self.classifier(camera_processed)
        if not return_intermediates:
            return logits, camera_processed
        intermediates["logits"] = logits
        return logits, camera_processed, intermediates

    def optical_parameter_count(self) -> int:
        return int(self.optical_backbone.optical_parameter_count())

    def expert_phase_parameter_count(self) -> int:
        return int(self.optical_backbone.expert_phase_parameter_count())

    def global_fc_parameter_count(self) -> int:
        return int(self.optical_backbone.global_fc_parameter_count())

    def prompt_parameter_count(self) -> int:
        return int(self.optical_backbone.prompt_parameter_count())

    def feature_preprocess_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.feature_preprocess.parameters())

    def classifier_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.classifier.parameters())

    def electronic_parameter_count(self) -> int:
        return int(self.classifier_parameter_count())

    def total_parameter_count(self) -> int:
        return int(
            self.optical_parameter_count()
            + self.prompt_parameter_count()
            + self.feature_preprocess_parameter_count()
            + self.electronic_parameter_count()
        )

    def set_phase_dropout_active(self, active: bool) -> None:
        self.optical_backbone.set_phase_dropout_active(active)


class FeatureDistilledASGlobalRouterMoEClassifier(DetectorFeatureASGlobalRouterMoEClassifier):
    """Camera-feature OpticalMoE with one shared semantic inference path."""

    def __init__(
        self,
        num_classes: int,
        teacher_feature_dim: int,
        layout: ExpertLayout,
        feature_detector_config: Optional[Dict] = None,
        feature_preprocess_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        projector_config: Optional[Dict] = None,
        **optical_backbone_kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            layout=layout,
            feature_detector_config=feature_detector_config,
            feature_preprocess_config=feature_preprocess_config,
            classifier_config=classifier_config,
            **optical_backbone_kwargs,
        )
        self.teacher_feature_dim = int(teacher_feature_dim)
        self.projector, self.projector_config = build_projector(
            self.camera_feature_dim,
            self.teacher_feature_dim,
            projector_config,
        )
        classifier_cfg = dict(classifier_config or {})
        if str(classifier_cfg.get("input", "semantic_feature")) != "semantic_feature":
            raise ValueError("Distillation classifier.input must be semantic_feature.")
        _resolve_auto_dim(
            classifier_cfg.get("input_dim", "auto_teacher_dim"),
            self.teacher_feature_dim,
            "classifier.input_dim",
        )
        self.classifier = _mlp(self.teacher_feature_dim, self.num_classes, classifier_cfg)
        self.classifier_config = {
            **classifier_cfg,
            "input": "semantic_feature",
            "input_dim": self.teacher_feature_dim,
        }

    def _semantic_outputs(self, camera_processed: torch.Tensor):
        semantic_feature = self.projector(camera_processed)
        semantic_feature_normalized = F.normalize(semantic_feature, dim=-1)
        classifier_feature = (
            semantic_feature_normalized
            if self.projector_config["output_l2_normalize"]
            else semantic_feature
        )
        logits = self.classifier(classifier_feature)
        return logits, semantic_feature, semantic_feature_normalized, classifier_feature

    def forward_with_aux(self, images: torch.Tensor):
        camera_raw, camera_processed, outside_ratio = self._camera_features(
            images, return_intermediates=False
        )
        logits, semantic, semantic_normalized, _classifier_feature = self._semantic_outputs(
            camera_processed
        )
        return (
            logits,
            camera_raw,
            camera_processed,
            semantic,
            semantic_normalized,
            {"outside_camera_energy_ratio": outside_ratio},
        )

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        outputs = self._camera_features(images, return_intermediates=return_intermediates)
        if return_intermediates:
            camera_raw, camera_processed, _outside_ratio, intermediates = outputs
        else:
            camera_raw, camera_processed, _outside_ratio = outputs
        logits, semantic_feature, semantic_feature_normalized, classifier_feature = self._semantic_outputs(
            camera_processed
        )
        if not return_intermediates:
            return logits, camera_raw, camera_processed, semantic_feature, semantic_feature_normalized
        intermediates.update(
            {
                "semantic_feature": semantic_feature,
                "semantic_feature_normalized": semantic_feature_normalized,
                "classifier_feature": classifier_feature,
                "projected_feature": semantic_feature_normalized,
                "logits": logits,
            }
        )
        return logits, camera_raw, camera_processed, semantic_feature, semantic_feature_normalized, intermediates

    def projector_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.projector.parameters())

    def electronic_parameter_count(self) -> int:
        return int(self.projector_parameter_count() + self.classifier_parameter_count())

    def total_parameter_count(self) -> int:
        return int(
            self.optical_parameter_count()
            + self.prompt_parameter_count()
            + self.feature_preprocess_parameter_count()
            + self.electronic_parameter_count()
        )
