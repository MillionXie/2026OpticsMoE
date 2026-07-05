from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.optics.distilled_moe import FeaturePreprocess, _mlp, build_projector


def _activation(name: str) -> nn.Module:
    choices = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
    key = str(name).lower()
    if key not in choices:
        raise ValueError(f"Unsupported LeNet activation: {name!r}.")
    return choices[key]()


class FeatureDistilledLeNetClassifier(nn.Module):
    """Electronic diagnostic baseline using the shared semantic distillation path."""

    def __init__(
        self,
        num_classes: int,
        teacher_feature_dim: int,
        lenet_config: Optional[Dict] = None,
        feature_preprocess_config: Optional[Dict] = None,
        projector_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        lenet_cfg = dict(lenet_config or {})
        channels: Sequence[int] = lenet_cfg.get("channels", [32, 64, 128])
        if len(channels) != 3:
            raise ValueError("lenet.channels must contain exactly three channel sizes.")
        input_channels = int(lenet_cfg.get("input_channels", 1))
        if input_channels != 1:
            raise ValueError("FeatureDistilledLeNetClassifier requires one-channel grayscale input.")
        if str(lenet_cfg.get("pooling", "avg")).lower() != "avg":
            raise ValueError("The LeNet distillation baseline currently supports pooling=avg only.")
        activation = str(lenet_cfg.get("activation", "gelu"))
        pool_size = int(lenet_cfg.get("adaptive_pool_size", 5))
        self.student_feature_dim = int(lenet_cfg.get("output_feature_dim", 900))
        if self.student_feature_dim != 900:
            raise ValueError("lenet.output_feature_dim must be 900 for the matched distillation backend.")
        self.num_classes = int(num_classes)
        self.teacher_feature_dim = int(teacher_feature_dim)
        self.lenet_config = {
            "input_channels": input_channels,
            "channels": [int(value) for value in channels],
            "activation": activation,
            "pooling": "avg",
            "adaptive_pool_size": pool_size,
            "output_feature_dim": self.student_feature_dim,
        }
        c1, c2, c3 = self.lenet_config["channels"]
        self.lenet_backbone = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=5, padding=2),
            _activation(activation),
            nn.AvgPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=5, padding=2),
            _activation(activation),
            nn.AvgPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            _activation(activation),
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
        )
        self.lenet_feature_projection = nn.Linear(c3 * pool_size * pool_size, self.student_feature_dim)
        self.feature_preprocess = FeaturePreprocess(self.student_feature_dim, feature_preprocess_config)
        self.feature_preprocess_config = dict(self.feature_preprocess.config)
        self.projector, self.projector_config = build_projector(
            self.student_feature_dim,
            self.teacher_feature_dim,
            projector_config,
        )
        classifier_cfg = dict(classifier_config or {})
        if str(classifier_cfg.get("input", "semantic_feature")) != "semantic_feature":
            raise ValueError("LeNet distillation classifier.input must be semantic_feature.")
        configured_input = classifier_cfg.get("input_dim", "auto_teacher_dim")
        if configured_input not in {None, "auto_teacher_dim"} and int(configured_input) != self.teacher_feature_dim:
            raise ValueError("classifier.input_dim must resolve to teacher_feature_dim.")
        self.classifier = _mlp(self.teacher_feature_dim, self.num_classes, classifier_cfg)
        self.classifier_config = {
            **classifier_cfg,
            "input": "semantic_feature",
            "input_dim": self.teacher_feature_dim,
        }

    def _features(self, images: torch.Tensor):
        images = torch.as_tensor(images).float()
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("LeNet distillation input must have shape [B,1,H,W].")
        encoded = self.lenet_backbone(images).flatten(1)
        raw = self.lenet_feature_projection(encoded)
        processed = self.feature_preprocess(raw)
        semantic = self.projector(processed)
        semantic_normalized = F.normalize(semantic, dim=-1)
        classifier_input = semantic_normalized if self.projector_config["output_l2_normalize"] else semantic
        logits = self.classifier(classifier_input)
        return logits, raw, processed, semantic, semantic_normalized, classifier_input

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        logits, raw, processed, semantic, semantic_normalized, classifier_input = self._features(images)
        if not return_intermediates:
            return logits, raw, processed, semantic, semantic_normalized
        intermediates = {
            "lenet_feature_raw": raw,
            "lenet_feature_processed": processed,
            "semantic_feature": semantic,
            "semantic_feature_normalized": semantic_normalized,
            "classifier_feature": classifier_input,
            "logits": logits,
            "outside_camera_energy_ratio": None,
        }
        return logits, raw, processed, semantic, semantic_normalized, intermediates

    def forward_with_aux(self, images: torch.Tensor):
        logits, raw, processed, semantic, semantic_normalized, _classifier_input = self._features(images)
        return logits, raw, processed, semantic, semantic_normalized, {
            "outside_camera_energy_ratio": logits.new_zeros(logits.shape[0])
        }

    def lenet_parameter_count(self) -> int:
        return int(
            sum(parameter.numel() for parameter in self.lenet_backbone.parameters())
            + sum(parameter.numel() for parameter in self.lenet_feature_projection.parameters())
        )

    def feature_preprocess_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.feature_preprocess.parameters()))

    def projector_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.projector.parameters()))

    def classifier_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.classifier.parameters()))

    def optical_parameter_count(self) -> int:
        return 0

    def prompt_parameter_count(self) -> int:
        return 0

    def electronic_parameter_count(self) -> int:
        return int(
            self.lenet_parameter_count()
            + self.feature_preprocess_parameter_count()
            + self.projector_parameter_count()
            + self.classifier_parameter_count()
        )

    def total_parameter_count(self) -> int:
        return self.electronic_parameter_count()
