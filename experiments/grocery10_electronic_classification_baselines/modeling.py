from __future__ import annotations

from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)
from torchvision.transforms import functional as TF

from .settings import Settings


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_backbone(model_name: str, weights: str) -> tuple[nn.Module, int]:
    pretrained = weights == "imagenet1k"
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier = nn.Identity()
        return model, feature_dim
    if model_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        feature_dim = int(model.classifier[0].in_features)
        model.classifier = nn.Identity()
        return model, feature_dim
    raise ValueError(f"Unsupported electronic backbone {model_name!r}")


class ElectronicTenClassClassifier(nn.Module):
    """Pretrained CNN with a direct ten-logit classification readout."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        *,
        weights: str = "imagenet1k",
        train_backbone: bool = True,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.weights_name = str(weights)
        self.num_classes = int(num_classes)
        self.backbone, self.feature_dim = _build_backbone(model_name, weights)
        self.feature_norm = (
            nn.LayerNorm(self.feature_dim) if use_layernorm else nn.Identity()
        )
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        self.train_backbone = bool(train_backbone)
        self.use_layernorm = bool(use_layernorm)
        if not self.train_backbone:
            self.backbone.requires_grad_(False)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.shape != (images.shape[0], self.feature_dim):
            raise RuntimeError(
                f"{self.model_name} returned {tuple(features.shape)}; "
                f"expected [B,{self.feature_dim}]"
            )
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(images)
        logits = self.classifier(self.feature_norm(features.float()))
        if logits.shape != (images.shape[0], self.num_classes):
            raise RuntimeError("Electronic classification logits have invalid shape")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Electronic classification logits contain NaN/Inf")
        return logits

    def train(self, mode: bool = True) -> "ElectronicTenClassClassifier":
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        return self


def build_model(settings: Settings, device: torch.device) -> ElectronicTenClassClassifier:
    return ElectronicTenClassClassifier(
        settings.model_name,
        len(settings.selected_skus),
        weights=settings.weights,
        train_backbone=settings.train_backbone,
        use_layernorm=settings.head_use_layernorm,
    ).to(device)


def preprocess_pil_images(
    images: Sequence[Image.Image], device: torch.device
) -> torch.Tensor:
    values = torch.stack(
        [TF.pil_to_tensor(image.convert("RGB")).float().div_(255.0) for image in images]
    )
    mean = values.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = values.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((values - mean) / std).to(device, non_blocking=True)


def parameter_report(model: ElectronicTenClassClassifier) -> dict[str, Any]:
    def count(module: nn.Module, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or parameter.requires_grad
        )

    norm = count(model.feature_norm)
    classifier = count(model.classifier)
    return {
        "task": "grocery10_direct_classification",
        "similarity_matching_used": False,
        "model_name": model.model_name,
        "weights": model.weights_name,
        "feature_dim": model.feature_dim,
        "num_classes": model.num_classes,
        "head": (
            f"{'LayerNorm -> ' if model.use_layernorm else ''}"
            f"Linear({model.feature_dim},{model.num_classes})"
        ),
        "backbone_parameters": count(model.backbone),
        "backbone_trainable_parameters": count(
            model.backbone, trainable_only=True
        ),
        "feature_norm_parameters": norm,
        "classifier_parameters": classifier,
        "classification_head_parameters": norm + classifier,
        "parameters": count(model),
        "trainable_parameters": count(model, trainable_only=True),
        "output_shape": [None, model.num_classes],
    }
