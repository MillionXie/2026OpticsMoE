from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.modeling import (
    FourLayerOpticalReplacement as RobustFourLayerOpticalReplacement,
    load_backbone,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import (
    LanguageTwoBlockOpticalReplacement,
    VisionTwoBlockOpticalReplacement,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
)


ROBUST_SOURCE_ARCHITECTURE = (
    "vision2_language2_moe4_10cm_robust_bounded_fusion_v2"
)
STAGE_ARCHITECTURES = {
    "optical_calibration": "vision2_language2_moe4_10cm_warmstart5_stage_a_v1",
    "joint": "vision2_language2_moe4_10cm_warmstart5_stage_b_v1",
}
GATE_KEYS = {
    "core.block1_optical_fusion_logit",
    "core.block2_optical_fusion_logit",
}
OPTICAL_PREFIX = "core.optical_branch."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_payload(path: Path, expected_sha256: str | None) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint is missing: {path}")
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256.lower():
        raise RuntimeError(
            f"Warm-start checkpoint SHA-256 mismatch for {path}: "
            f"expected={expected_sha256.lower()} actual={digest}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "checkpoint_version",
        "epoch",
        "train_loss",
        "vision_optical",
        "language_optical",
        "retrieval_readout",
        "metadata",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Checkpoint {path} is missing keys {sorted(missing)}")
    if int(payload["checkpoint_version"]) != 2:
        raise RuntimeError(f"Checkpoint {path} is not version 2")
    return payload, digest


def _assert_tensor_shape(name: str, source: torch.Tensor, target: torch.Tensor) -> None:
    if tuple(source.shape) != tuple(target.shape):
        raise RuntimeError(
            f"Warm-start tensor shape mismatch for {name}: "
            f"source={tuple(source.shape)} target={tuple(target.shape)}"
        )


def merge_surrogate_states(
    target: Mapping[str, torch.Tensor],
    electronic: Mapping[str, torch.Tensor],
    optical: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Strictly merge all electronic tensors and only optical-branch tensors."""

    target_keys = set(target)
    optical_keys = {key for key in target_keys if key.startswith(OPTICAL_PREFIX)}
    gate_keys = target_keys.intersection(GATE_KEYS)
    electronic_keys = target_keys.difference(optical_keys | gate_keys)
    if set(electronic) != electronic_keys:
        raise RuntimeError(
            "Electronic checkpoint is not the exact 2D/no-DeepStack architecture: "
            f"missing={sorted(electronic_keys.difference(electronic))} "
            f"unexpected={sorted(set(electronic).difference(electronic_keys))}"
        )
    source_optical_keys = {key for key in optical if key.startswith(OPTICAL_PREFIX)}
    if source_optical_keys != optical_keys:
        raise RuntimeError(
            "Optical checkpoint does not contain the exact robust optical branch: "
            f"missing={sorted(optical_keys.difference(source_optical_keys))} "
            f"unexpected={sorted(source_optical_keys.difference(optical_keys))}"
        )
    merged = {key: value.detach().clone() for key, value in target.items()}
    for key in sorted(electronic_keys):
        _assert_tensor_shape(key, electronic[key], target[key])
        merged[key] = electronic[key].detach().clone()
    for key in sorted(optical_keys):
        _assert_tensor_shape(key, optical[key], target[key])
        merged[key] = optical[key].detach().clone()
    # Gate tensors deliberately remain at the freshly constructed 5.5% value.
    return merged, {
        "electronic_tensor_count": len(electronic_keys),
        "optical_tensor_count": len(optical_keys),
        "gate_tensor_count": len(gate_keys),
        "gate_keys_reset_not_loaded": sorted(gate_keys),
    }


def apply_stage_trainability(
    replacement: Any, readout: torch.nn.Module, stage: str
) -> dict[str, int]:
    if stage not in STAGE_ARCHITECTURES:
        raise ValueError(f"Unknown warm-start stage {stage!r}")
    if stage == "joint":
        replacement.configure_student_trainability()
        replacement.language_surrogate.core.residual_logit.requires_grad_(False)
        readout.requires_grad_(True)
    else:
        replacement.vision_surrogate.requires_grad_(False)
        replacement.language_surrogate.requires_grad_(False)
        readout.requires_grad_(False)
        replacement.vision_surrogate.core.optical_branch.requires_grad_(True)
        replacement.language_surrogate.core.optical_branch.requires_grad_(True)
    return {
        "replacement_trainable_parameters": sum(
            parameter.numel()
            for parameter in replacement.trainable_parameters()
            if parameter.requires_grad
        ),
        "readout_trainable_parameters": sum(
            parameter.numel() for parameter in readout.parameters() if parameter.requires_grad
        ),
    }


class WarmStartFiveReplacement(RobustFourLayerOpticalReplacement):
    training_architecture_label = "vision2_language2_moe4_10cm_warmstart5"

    def __init__(self, *args: Any, settings: Any, **kwargs: Any) -> None:
        self.warmstart_stage = settings.warmstart_stage
        super().__init__(*args, settings=settings, **kwargs)
        self.checkpoint_architecture = STAGE_ARCHITECTURES[self.warmstart_stage]

    def configure_student_trainability(self) -> None:
        super().configure_student_trainability()
        if self.warmstart_stage == "optical_calibration":
            self.vision_surrogate.requires_grad_(False)
            self.language_surrogate.requires_grad_(False)
            self.vision_surrogate.core.optical_branch.requires_grad_(True)
            self.language_surrogate.core.optical_branch.requires_grad_(True)

    def student_architecture_report(self) -> dict[str, Any]:
        report = super().student_architecture_report()
        report.update(
            {
                "type": self.training_architecture_label,
                "checkpoint_architecture": self.checkpoint_architecture,
                "initialization": "strict_dual_source_warm_start",
                "warmstart_stage": self.warmstart_stage,
                "sealed_test": True,
                "minimum_optical_fusion_coefficient": 0.05,
                "initial_optical_fusion_coefficient": 0.055,
                "optical_fraction_semantics": (
                    "coefficient floor; not a measured optical-energy fraction"
                ),
            }
        )
        return report


def build_hybrid_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[WarmStartFiveReplacement, ElectronicRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    vision = VisionTwoBlockOpticalReplacement(
        settings.vision_hidden_size, settings
    ).to(loaded.device)
    language = LanguageTwoBlockOpticalReplacement(
        settings.text_hidden_size, settings
    ).to(loaded.device)
    replacement = WarmStartFiveReplacement(
        loaded.model, vision, language, settings=settings
    )
    readout = ElectronicRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(loaded.device)
    apply_stage_trainability(replacement, readout, settings.warmstart_stage)
    return replacement, readout


def _validate_metadata(
    payload: Mapping[str, Any], settings: Any, *, expected_architecture: str | None
) -> None:
    metadata = dict(payload.get("metadata", {}))
    if int(metadata.get("embedding_dim", -1)) != int(settings.embedding_dim):
        raise RuntimeError("Warm-start checkpoint embedding_dim mismatch")
    if int(metadata.get("detector_dim", -1)) != int(settings.detector_output_size):
        raise RuntimeError("Warm-start checkpoint detector_dim mismatch")
    if str(metadata.get("model_id")) != str(settings.model_id):
        raise RuntimeError("Warm-start checkpoint Qwen model_id mismatch")
    # Historical checkpoint metadata records the native Qwen tap marker as one
    # auxiliary even when the replacement's actual DeepStack index list is
    # empty. Exact state keys, tensor shapes and the pinned SHA are therefore
    # authoritative; rejecting on that stale diagnostic would reject both
    # audited source checkpoints.
    if metadata.get("test_metrics_used_for_selection") is not False:
        raise RuntimeError("Warm-start source must be selected without test metrics")
    if metadata.get("weight_variant") != "ema":
        raise RuntimeError("Warm-start source must be an EMA checkpoint")
    if expected_architecture is not None and metadata.get("optical_architecture") != expected_architecture:
        raise RuntimeError(
            "Warm-start optical architecture mismatch: "
            f"expected={expected_architecture!r} "
            f"actual={metadata.get('optical_architecture')!r}"
        )


def load_dual_initialization(
    settings: Any,
    replacement: WarmStartFiveReplacement,
    readout: ElectronicRetrievalReadout,
) -> dict[str, Any]:
    if settings.warmstart_stage != "optical_calibration":
        raise RuntimeError("Dual-source initialization is only valid for Stage A")
    electronic, electronic_sha = _load_payload(
        settings.warmstart_electronic_checkpoint,
        settings.warmstart_electronic_sha256,
    )
    optical, optical_sha = _load_payload(
        settings.warmstart_optical_checkpoint,
        settings.warmstart_optical_sha256,
    )
    _validate_metadata(electronic, settings, expected_architecture=None)
    _validate_metadata(
        optical, settings, expected_architecture=ROBUST_SOURCE_ARCHITECTURE
    )
    reports: dict[str, Any] = {}
    for name, surrogate in (
        ("vision", replacement.vision_surrogate),
        ("language", replacement.language_surrogate),
    ):
        merged, report = merge_surrogate_states(
            surrogate.state_dict(),
            electronic[f"{name}_optical"],
            optical[f"{name}_optical"],
        )
        surrogate.load_state_dict(merged, strict=True)
        reports[name] = report
    target_readout = readout.state_dict()
    source_readout = electronic["retrieval_readout"]
    if set(source_readout) != set(target_readout):
        raise RuntimeError("Electronic retrieval head keys do not match")
    for key in target_readout:
        _assert_tensor_shape(f"retrieval_readout.{key}", source_readout[key], target_readout[key])
    readout.load_state_dict(source_readout, strict=True)
    return {
        "mode": "dual_source",
        "electronic": {
            "path": str(settings.warmstart_electronic_checkpoint),
            "sha256": electronic_sha,
            "epoch": int(electronic["epoch"]),
            "train_loss": float(electronic["train_loss"]),
        },
        "optical": {
            "path": str(settings.warmstart_optical_checkpoint),
            "sha256": optical_sha,
            "epoch": int(optical["epoch"]),
            "train_loss": float(optical["train_loss"]),
        },
        "surrogates": reports,
        "readout_tensor_count": len(target_readout),
        "fusion": {"minimum": 0.05, "initial": 0.055},
        "trainability": apply_stage_trainability(
            replacement, readout, settings.warmstart_stage
        ),
    }


def load_stage_a_initialization(
    settings: Any,
    replacement: WarmStartFiveReplacement,
    readout: ElectronicRetrievalReadout,
) -> dict[str, Any]:
    if settings.warmstart_stage != "joint":
        raise RuntimeError("Stage-A initialization is only valid for Stage B")
    payload, digest = _load_payload(settings.warmstart_stage_a_checkpoint, None)
    _validate_metadata(
        payload,
        settings,
        expected_architecture=STAGE_ARCHITECTURES["optical_calibration"],
    )
    replacement.vision_surrogate.load_state_dict(payload["vision_optical"], strict=True)
    replacement.language_surrogate.load_state_dict(payload["language_optical"], strict=True)
    readout.load_state_dict(payload["retrieval_readout"], strict=True)
    for name, core in (
        ("vision", replacement.vision_surrogate.core),
        ("language", replacement.language_surrogate.core),
    ):
        for gate_name, value in (
            ("block1", core.block1_optical_fusion),
            ("block2", core.block2_optical_fusion),
        ):
            if abs(float(value.detach()) - 0.055) > 1.0e-5:
                raise RuntimeError(
                    f"Stage-A {name} {gate_name} gate changed despite being frozen"
                )
    return {
        "mode": "stage_a_checkpoint",
        "path": str(settings.warmstart_stage_a_checkpoint),
        "sha256": digest,
        "epoch": int(payload["epoch"]),
        "train_loss": float(payload["train_loss"]),
        "trainability": apply_stage_trainability(replacement, readout, "joint"),
    }


__all__ = [
    "GATE_KEYS",
    "OPTICAL_PREFIX",
    "ROBUST_SOURCE_ARCHITECTURE",
    "STAGE_ARCHITECTURES",
    "WarmStartFiveReplacement",
    "apply_stage_trainability",
    "build_hybrid_student",
    "load_backbone",
    "load_dual_initialization",
    "load_stage_a_initialization",
    "merge_surrogate_states",
    "sha256_file",
]
