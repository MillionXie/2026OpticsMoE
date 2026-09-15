from __future__ import annotations

import importlib
from typing import Any, Mapping

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    forward_base_hidden,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
)


SOURCE_PACKAGE = (
    "qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval"
)


def _source_module(name: str) -> Any:
    candidates = (
        f"experiments.{SOURCE_PACKAGE}.{name}",
        f"{SOURCE_PACKAGE}.{name}",
    )
    errors: list[BaseException] = []
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except ModuleNotFoundError as error:
            errors.append(error)
    raise ModuleNotFoundError(
        f"Could not import the router source package from {candidates}"
    ) from errors[-1]


_source = _source_module("modeling")
RouterAblationReplacement = _source.RouterAblationReplacement
load_backbone = _source.load_backbone


class CIFAR10ClassificationHead(nn.Module):
    """Unnormalised ten-class logits from the 384-D detector representation."""

    def __init__(self, detector_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        self.detector_dim = int(detector_dim)
        self.num_classes = int(num_classes)
        self.norm = nn.LayerNorm(self.detector_dim)
        self.classifier = nn.Linear(self.detector_dim, self.num_classes)

    def forward(self, detector_features: torch.Tensor) -> torch.Tensor:
        if detector_features.ndim != 2 or detector_features.shape[1] != self.detector_dim:
            raise RuntimeError(
                f"Detector features must be [B,{self.detector_dim}], got "
                f"{tuple(detector_features.shape)}"
            )
        if not torch.isfinite(detector_features).all():
            raise RuntimeError("Detector features contain NaN or Inf")
        logits = self.classifier(self.norm(detector_features.float()))
        if tuple(logits.shape) != (detector_features.shape[0], self.num_classes):
            raise RuntimeError(f"Unexpected classifier output {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Classification logits contain NaN or Inf")
        return logits

    def specification(self) -> dict[str, Any]:
        return {
            "type": "cifar10_classification_head",
            "architecture": (
                f"LayerNorm({self.detector_dim}) -> "
                f"Linear({self.detector_dim},{self.num_classes})"
            ),
            "softmax_inside_model": False,
            "l2_normalization": False,
            "detector_dim": self.detector_dim,
            "num_classes": self.num_classes,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


def build_classification_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[RouterAblationReplacement, CIFAR10ClassificationHead]:
    settings.resolve_architecture(loaded.model)
    if int(settings.detector_output_size) != 384:
        raise RuntimeError(
            "The audited CIFAR-10 architecture requires mean+max detector "
            f"features of width 384, got {settings.detector_output_size}"
        )
    vision = _source.VisionTwoBlockOpticalReplacement(
        settings.vision_hidden_size, settings
    )
    language = _source.LanguageTwoBlockOpticalReplacement(
        settings.text_hidden_size, settings
    )
    _source._install_router(vision, settings)
    _source._install_router(language, settings)
    vision = vision.to(loaded.device)
    language = language.to(loaded.device)
    replacement = RouterAblationReplacement(
        loaded.model, vision, language, settings=settings
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(settings.classification_head_seed))
        head = CIFAR10ClassificationHead(
            settings.detector_output_size, settings.num_classes
        )
    head = head.to(loaded.device)
    replacement.configure_student_trainability()
    head.requires_grad_(True)
    return replacement, head


def classification_logits(
    model: nn.Module,
    replacement: RouterAblationReplacement,
    head: CIFAR10ClassificationHead,
    inputs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    replacement.use_student()
    replacement.prepare_student_batch(
        inputs["attention_mask"], inputs.get("image_grid_thw")
    )
    forward_base_hidden(model, inputs)
    detector_features = replacement.language_surrogate.retrieval_detector_features()
    if tuple(detector_features.shape) != (
        inputs["attention_mask"].shape[0],
        head.detector_dim,
    ):
        raise RuntimeError(
            "Expected one 384-D detector vector per input, got "
            f"{tuple(detector_features.shape)}"
        )
    return head(detector_features), detector_features


def load_backbone_only_warmstart(
    settings: Any,
    replacement: RouterAblationReplacement,
) -> dict[str, Any]:
    """Strictly load the two surrogate bodies and deliberately skip the 64-D head."""

    payload, digest = _source._load_payload(
        settings.router_source_checkpoint, settings.router_source_sha256
    )
    _source._validate_metadata(
        payload,
        settings,
        expected_architecture=_source.STAGE_ARCHITECTURES["joint"],
    )
    reports: dict[str, Any] = {}
    for name, surrogate in (
        ("vision", replacement.vision_surrogate),
        ("language", replacement.language_surrogate),
    ):
        state, report = _source._copy_surrogate_state(
            surrogate.state_dict(),
            payload[f"{name}_optical"],
            reset_router=settings.router_reset_parameters,
        )
        surrogate.load_state_dict(state, strict=True)
        reports[name] = report
    return {
        "mode": "warmstart5_body_only_fresh_cifar10_head",
        "path": str(settings.router_source_checkpoint),
        "sha256": digest,
        "source_epoch": int(payload["epoch"]),
        "source_train_loss": float(payload["train_loss"]),
        "source_checkpoint_architecture": _source.STAGE_ARCHITECTURES["joint"],
        "target_checkpoint_architecture": replacement.checkpoint_architecture,
        "skipped_source_keys": ["retrieval_readout", "optimizer"],
        "classification_head_seed": int(settings.classification_head_seed),
        "router_parameters_reset": bool(settings.router_reset_parameters),
        "surrogates": reports,
    }


def architecture_report(
    replacement: RouterAblationReplacement,
    head: CIFAR10ClassificationHead,
    settings: Any,
) -> dict[str, Any]:
    report = replacement.student_architecture_report()
    report.update(
        {
            "task": "cifar10_classification",
            "initialization": (
                "warmstart5_body_only_fresh_cifar10_head"
                if settings.classification_initialization == "warmstart_body"
                else "deterministic_random_compact_student_and_cifar10_head"
            ),
            "checkpoint_architecture": (
                f"{replacement.checkpoint_architecture}_cifar10_cls10_v1"
            ),
            "detector_feature_pooling": "language latent mean+max",
            "classification_head": head.specification(),
            "class_names": list(settings.class_names),
            "task_loss": "cross_entropy_on_raw_logits",
        }
    )
    return report


__all__ = [
    "CIFAR10ClassificationHead",
    "RouterAblationReplacement",
    "architecture_report",
    "build_classification_student",
    "classification_logits",
    "load_backbone",
    "load_backbone_only_warmstart",
]
