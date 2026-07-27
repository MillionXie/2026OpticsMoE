from __future__ import annotations

from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
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


class ElectronicRetrievalEncoder(nn.Module):
    """ImageNet CNN backbone followed by a signed 64-D retrieval projection."""

    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 64,
        *,
        weights: str = "imagenet1k",
        train_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.weights_name = str(weights)
        self.backbone, self.feature_dim = _build_backbone(model_name, weights)
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.projection = nn.Linear(self.feature_dim, embedding_dim)
        self.embedding_dim = int(embedding_dim)
        self.train_backbone = bool(train_backbone)
        if not self.train_backbone:
            self.backbone.requires_grad_(False)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        values = self.backbone(images)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"{self.model_name} feature shape {tuple(values.shape)} is invalid; "
                f"expected [B,{self.feature_dim}]"
            )
        return values

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(images)
        signed = self.projection(self.feature_norm(features.float()))
        if not torch.isfinite(signed).all():
            raise RuntimeError("Electronic retrieval projection contains NaN/Inf")
        norms = signed.norm(dim=-1)
        if torch.any(norms <= 1e-12):
            raise RuntimeError("Electronic retrieval projection produced a zero vector")
        return F.normalize(signed, dim=-1)

    def train(self, mode: bool = True) -> "ElectronicRetrievalEncoder":
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        return self


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


def build_model(settings: Settings, device: torch.device) -> ElectronicRetrievalEncoder:
    model = ElectronicRetrievalEncoder(
        settings.model_name,
        settings.embedding_dim,
        weights=settings.weights,
        train_backbone=settings.train_backbone,
    )
    return model.to(device)


def preprocess_pil_images(
    images: Sequence[Image.Image], device: torch.device
) -> torch.Tensor:
    values = torch.stack([TF.pil_to_tensor(image).float().div_(255.0) for image in images])
    mean = values.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = values.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    values = (values - mean) / std
    return values.to(device, non_blocking=True)


def parameter_report(model: ElectronicRetrievalEncoder) -> dict[str, Any]:
    def count(module: nn.Module, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or parameter.requires_grad
        )

    backbone_parameters = count(model.backbone)
    norm_parameters = count(model.feature_norm)
    projection_parameters = count(model.projection)
    total = count(model)
    trainable = count(model, trainable_only=True)
    return {
        "model_name": model.model_name,
        "weights": model.weights_name,
        "feature_dim": model.feature_dim,
        "embedding_dim": model.embedding_dim,
        "embedding_normalization": "L2",
        "head": (
            f"LayerNorm({model.feature_dim}) -> "
            f"Linear({model.feature_dim},{model.embedding_dim}) -> L2 normalize"
        ),
        "backbone_parameters": backbone_parameters,
        "backbone_trainable_parameters": count(
            model.backbone, trainable_only=True
        ),
        "feature_norm_parameters": norm_parameters,
        "projection_parameters": projection_parameters,
        "retrieval_head_parameters": norm_parameters + projection_parameters,
        "parameters": total,
        "trainable_parameters": trainable,
    }
