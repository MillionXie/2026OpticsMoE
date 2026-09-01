from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
    QwenStemSeparableOpticalImageNetBackbone,
)

from .model import (
    P11_SOURCE_STAGE_COUNT,
    P13_SUPPORTED_DEPTHS,
    QwenStemProgressiveOpticalImageNetBackbone,
    anchor_stage_indices,
)


P11_SIGNATURE_KEY = "p11_separable_architecture_signature"
P11_SIGNATURE = (11, 1, 2, 4)
P13_SIGNATURE_KEY = "p13_progressive_architecture_signature"
MIGRATION_FORMAT = "p13-progressive-p11-migration-v1"
PROGRESSIVE_MIGRATION_FORMAT = "p13-progressive-depth-migration-v1"
P13_TRAINING_CHECKPOINT_FORMAT = "p13-progressive-imagenet-training-v1"
P13_DEPTH_TRANSITIONS = ((16, 32), (32, 64), (64, 100))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_sequence_sha256(phases: Sequence[torch.Tensor]) -> str:
    """Hash an ordered phase sequence, including dtype and tensor shape."""

    digest = hashlib.sha256()
    for index, phase in enumerate(phases):
        value = phase.detach().cpu().contiguous()
        digest.update(f"{index}:{value.dtype}:{tuple(value.shape)}:".encode("utf-8"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a tensor state dict in a key-order-independent representation."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(
            f"{name}:{value.dtype}:{tuple(value.shape)}:".encode("utf-8")
        )
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_state(value: Any, *, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"{name} must be a non-empty tensor state dict")
    state: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"{name} must map string names to tensors")
        state[key] = tensor
    return state


def _signature(
    state: Mapping[str, torch.Tensor],
    key: str,
    *,
    length: int = 4,
) -> tuple[int, ...] | None:
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (length,):
        return None
    return tuple(int(item) for item in value.tolist())


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} must be a non-empty string")
    return value


def _official_backbone_payload(
    checkpoint: str | Path,
) -> tuple[Path, Mapping[str, Any], dict[str, torch.Tensor]]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing P11 backbone checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "backbone" not in payload:
        raise RuntimeError(
            "P13 migration requires the official P11 backbone.pt payload with a "
            "top-level 'backbone' state dict"
        )
    state = _tensor_state(payload["backbone"], name="P11 backbone")
    if any(name.startswith("readout.") for name in state):
        raise RuntimeError("P11 reusable backbone export must not contain its task head")
    return path, payload, state


def _official_p11_training_payload(
    checkpoint: str | Path,
) -> tuple[Path, Mapping[str, Any], dict[str, torch.Tensor]]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing P11 training checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise RuntimeError(
            "P11 continuation migration requires best.pt with a top-level "
            "complete 'model' state dict"
        )
    state = _tensor_state(payload["model"], name="P11 best model")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise RuntimeError("P11 best.pt must contain a positive integer epoch")
    best_top1 = payload.get("best_validation_top1")
    if not isinstance(best_top1, (int, float)) or not math.isfinite(float(best_top1)):
        raise RuntimeError("P11 best.pt must contain finite best_validation_top1")
    _require_nonempty_string(payload.get("config_digest"), name="P11 best config_digest")
    return path, payload, state


def _validate_p11_identity(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    payload: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
) -> None:
    if _signature(state, P11_SIGNATURE_KEY) != P11_SIGNATURE:
        raise RuntimeError(
            f"Expected {P11_SIGNATURE_KEY}={P11_SIGNATURE}; this is not the locked P11 source"
        )
    source_stem_sha = _require_nonempty_string(
        payload.get("stem_checkpoint_sha256"),
        name="P11 backbone stem_checkpoint_sha256",
    )
    if source_stem_sha != target.stem.checkpoint_sha256:
        raise RuntimeError(
            "P11 and P13 Qwen stem SHA-256 values differ; refusing a non-reproducible migration"
        )
    report = payload.get("model_report")
    if not isinstance(report, Mapping):
        raise RuntimeError("Official P11 export is missing model_report")
    if report.get("optical_mixer_variant") != "separable_token_channel_axis":
        raise RuntimeError("P11 model_report does not identify the separable optical mixer")
    if report.get("num_stages") != P11_SOURCE_STAGE_COUNT:
        raise RuntimeError("P11 model_report does not contain exactly eight source stages")
    best_epoch = payload.get("best_epoch")
    if not isinstance(best_epoch, int) or isinstance(best_epoch, bool) or best_epoch <= 0:
        raise RuntimeError("Official P11 backbone export is missing a positive best_epoch")
    _require_nonempty_string(
        payload.get("config_digest"), name="P11 backbone config_digest"
    )


def _load_p11_source(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    backbone_checkpoint: str | Path,
    training_checkpoint: str | Path | None,
) -> tuple[QwenStemSeparableOpticalImageNetBackbone, dict[str, Any]]:
    backbone_path, backbone_payload, backbone_state = _official_backbone_payload(
        backbone_checkpoint
    )
    _validate_p11_identity(target, backbone_payload, backbone_state)
    source = QwenStemSeparableOpticalImageNetBackbone(
        target.stem_checkpoint,
        target.p11_reference_config(),
    )
    identity: dict[str, Any] = {
        "backbone_checkpoint": str(backbone_path),
        "backbone_checkpoint_sha256": sha256_file(backbone_path),
        "backbone_state_sha256": state_dict_sha256(backbone_state),
        "training_checkpoint": None,
        "training_checkpoint_sha256": None,
        "source_imagenet_head_migrated": False,
        "source_best_epoch": int(backbone_payload["best_epoch"]),
        "source_best_validation_top1": None,
        "source_config_digest": str(backbone_payload["config_digest"]),
        "source_stem_checkpoint_sha256": str(
            backbone_payload["stem_checkpoint_sha256"]
        ),
    }

    if training_checkpoint is None:
        incompatible = source.load_state_dict(backbone_state, strict=False)
        expected_missing = {
            f"readout.{name}" for name in source.readout.state_dict()
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "P11 backbone export failed strict reusable-body validation: "
                f"missing={sorted(incompatible.missing_keys)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        return source, identity

    training_path, training_payload, training_state = _official_p11_training_payload(
        training_checkpoint
    )
    if _signature(training_state, P11_SIGNATURE_KEY) != P11_SIGNATURE:
        raise RuntimeError("P11 best.pt architecture signature does not match P11")
    if training_payload["config_digest"] != backbone_payload["config_digest"]:
        raise RuntimeError("P11 best.pt/backbone.pt config digests differ")
    if int(training_payload["epoch"]) != int(backbone_payload["best_epoch"]):
        raise RuntimeError("P11 best.pt epoch does not match backbone.pt best_epoch")
    training_backbone = {
        name: value
        for name, value in training_state.items()
        if not name.startswith("readout.")
    }
    if set(training_backbone) != set(backbone_state):
        missing = sorted(set(backbone_state).difference(training_backbone))
        unexpected = sorted(set(training_backbone).difference(backbone_state))
        raise RuntimeError(
            "P11 best.pt and backbone.pt non-head key sets differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatches = [
        name
        for name in sorted(backbone_state)
        if not torch.equal(training_backbone[name], backbone_state[name])
    ]
    if mismatches:
        raise RuntimeError(
            "P11 best.pt and backbone.pt non-head tensors differ: "
            f"{mismatches[:8]}"
        )
    source.load_state_dict(training_state, strict=True)
    identity.update(
        {
            "training_checkpoint": str(training_path),
            "training_checkpoint_sha256": sha256_file(training_path),
            "training_state_sha256": state_dict_sha256(training_state),
            "source_imagenet_head_migrated": True,
            "source_best_validation_top1": float(
                training_payload["best_validation_top1"]
            ),
        }
    )
    return source, identity


def _p11_growth_parent_indices(
    target: QwenStemProgressiveOpticalImageNetBackbone,
) -> tuple[int, ...]:
    values = [-1] * target.num_stages
    for source_index, target_index in enumerate(target.anchor_indices):
        values[target_index] = source_index
    return tuple(values)


def _apply_p11_source(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    source: QwenStemSeparableOpticalImageNetBackbone,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    target.stem.load_state_dict(source.stem.state_dict(), strict=True)
    target.adapter.load_state_dict(source.adapter.state_dict(), strict=True)
    if identity["source_imagenet_head_migrated"]:
        target.readout.load_state_dict(source.readout.state_dict(), strict=True)
    target.set_growth_parent_stage_indices(
        _p11_growth_parent_indices(target),
        new_stage_alpha=0.0,
    )
    mapping: list[dict[str, Any]] = []
    for source_index, target_index in enumerate(target.anchor_indices):
        source_stage = source.stages[source_index]
        target_slot = target.slots[target_index]
        if not target_slot.is_p11_mixer_anchor:
            raise RuntimeError("Internal P13 mixer-anchor schedule is inconsistent")
        if target_slot.p11_source_stage_index != source_index:
            raise RuntimeError("Internal P13 P11-source schedule is inconsistent")
        if target_slot.growth_parent_stage_index != source_index:
            raise RuntimeError("Internal P13 growth-parent schedule is inconsistent")
        if target_slot.stage.optical_axis != source_stage.optical_axis:
            raise RuntimeError("P11/P13 anchor optical axes do not match")
        target_slot.stage.load_state_dict(source_stage.state_dict(), strict=True)
        target_slot.set_alpha(1.0)
        mapping.append(
            {
                "p11_source_stage_zero_based": source_index,
                "p13_target_stage_zero_based": target_index,
                "axis": source_stage.optical_axis,
            }
        )
    source_phase_hash = phase_sequence_sha256(
        [stage.raw_phase for stage in source.stages]
    )
    target_anchor_phase_hash = phase_sequence_sha256(
        [slot.stage.raw_phase for slot in target.mixer_anchor_slots()]
    )
    if source_phase_hash != target_anchor_phase_hash:
        raise RuntimeError("Migrated P13 anchor phases do not hash-match P11")
    feedback_source = target.capture_feedback_source(
        provenance={
            "capture": "immediately_after_strict_p11_to_p13_migration",
            "p11_source_checkpoint": identity["backbone_checkpoint"],
            "p11_source_checkpoint_sha256": identity[
                "backbone_checkpoint_sha256"
            ],
            "p11_training_checkpoint": identity["training_checkpoint"],
            "p11_training_checkpoint_sha256": identity[
                "training_checkpoint_sha256"
            ],
            "new_stage_phase_initialization": "deterministic_target_seed_schedule",
            "full_target_depth_captured": target.num_stages,
        }
    )
    manifest: dict[str, Any] = {
        "format": MIGRATION_FORMAT,
        "source_checkpoint": identity["backbone_checkpoint"],
        "source_checkpoint_sha256": identity["backbone_checkpoint_sha256"],
        "source_backbone_state_sha256": identity["backbone_state_sha256"],
        "source_training_checkpoint": identity["training_checkpoint"],
        "source_training_checkpoint_sha256": identity[
            "training_checkpoint_sha256"
        ],
        "source_training_state_sha256": identity.get("training_state_sha256"),
        "source_stem_checkpoint_sha256": identity[
            "source_stem_checkpoint_sha256"
        ],
        "source_architecture_signature": list(P11_SIGNATURE),
        "target_architecture_signature": [13, 1, 2, target.num_stages],
        "source_depth": P11_SOURCE_STAGE_COUNT,
        "target_num_stages": target.num_stages,
        "target_optical_phase_parameters": sum(
            parameter.numel() for parameter in target.phase_parameters()
        ),
        "anchor_mapping": mapping,
        "growth_parent_stage_indices_zero_based": [
            int(value)
            for value in target.growth_parent_stage_indices.detach().cpu().tolist()
        ],
        "source_phase_sequence_sha256": source_phase_hash,
        "target_anchor_phase_sequence_sha256": target_anchor_phase_hash,
        "phase_hash_domain": "ordered raw_phase tensors including dtype and shape",
        "adapter_migrated": True,
        "stem_buffers_migrated": True,
        "source_imagenet_head_migrated": bool(
            identity["source_imagenet_head_migrated"]
        ),
        "new_stage_phase_initialization": "deterministic target seed schedule",
        "new_stage_identity_skip_parameters": 0,
        "new_stage_depth_alpha": target.depth_alpha_report(),
        "full_depth_feedback_source": feedback_source,
        "source_best_epoch": identity["source_best_epoch"],
        "source_best_validation_top1": identity["source_best_validation_top1"],
        "source_config_digest": identity["source_config_digest"],
    }
    target.migration_manifest = manifest
    return manifest


def migrate_strict_p11_checkpoint(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Migrate the official reusable P11 body, intentionally excluding its head."""

    source, identity = _load_p11_source(target, checkpoint, None)
    return _apply_p11_source(target, source, identity)


def migrate_strict_p11_training_checkpoint(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    backbone_checkpoint: str | Path,
    training_checkpoint: str | Path,
) -> dict[str, Any]:
    """Cross-check P11 backbone.pt/best.pt and migrate body plus ImageNet head."""

    source, identity = _load_p11_source(
        target,
        backbone_checkpoint,
        training_checkpoint,
    )
    return _apply_p11_source(target, source, identity)


def _nearest_integer_fraction(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError(
            "Progressive pair interpolation expects non-negative fractions"
        )
    return (2 * numerator + denominator) // (2 * denominator)


def progressive_pair_mapping(
    source_depth: int,
    target_depth: int,
) -> tuple[int, ...]:
    """Embed every source token/channel pair while pinning P11 mixer anchors."""

    source_depth = int(source_depth)
    target_depth = int(target_depth)
    if (source_depth, target_depth) not in P13_DEPTH_TRANSITIONS:
        raise ValueError(
            f"Supported progressive transitions are {P13_DEPTH_TRANSITIONS}, "
            f"got {(source_depth, target_depth)}"
        )
    source_anchors = tuple(
        index // 2 for index in anchor_stage_indices(source_depth)[::2]
    )
    target_anchors = tuple(
        index // 2 for index in anchor_stage_indices(target_depth)[::2]
    )
    mapped: list[int] = []
    for source_pair in range(source_depth // 2):
        interval = next(
            index
            for index in range(len(source_anchors) - 1)
            if source_anchors[index] <= source_pair <= source_anchors[index + 1]
        )
        source_left, source_right = (
            source_anchors[interval],
            source_anchors[interval + 1],
        )
        target_left, target_right = (
            target_anchors[interval],
            target_anchors[interval + 1],
        )
        offset = _nearest_integer_fraction(
            (source_pair - source_left) * (target_right - target_left),
            source_right - source_left,
        )
        mapped.append(target_left + offset)
    if mapped != sorted(set(mapped)):
        raise RuntimeError("Progressive pair mapping is not strictly increasing")
    if any(
        mapped[source_anchor] != target_anchor
        for source_anchor, target_anchor in zip(source_anchors, target_anchors)
    ):
        raise RuntimeError("Progressive pair mapping did not preserve P11 mixer anchors")
    return tuple(mapped)


def progressive_stage_mapping(
    source_depth: int,
    target_depth: int,
) -> tuple[int, ...]:
    return tuple(
        stage
        for pair in progressive_pair_mapping(source_depth, target_depth)
        for stage in (2 * pair, 2 * pair + 1)
    )


_SHARED_CONFIG_DEFAULTS: dict[str, Any] = {
    "canvas_size": 224,
    "optical_channels": 3,
    "token_dim": 224,
    "num_classes": 1000,
    "head_hidden_dim": 448,
    "wavelength_m": 5.32e-7,
    "pixel_size_m": 1.6e-5,
    "token_axis_propagation_distance_m": 0.05,
    "channel_axis_propagation_distance_m": 0.05,
    "layernorm_eps": 1.0e-5,
    "optical_gate_init": 0.60,
    "optical_gate_min": 0.50,
    "mixer_width": 96,
    "mixer_expansion": 2.0,
    "mixer_kernel_size": 3,
    "mixer_dropout": 0.10,
    "mixer_spatial_gate_init": 0.10,
    "mixer_channel_gate_init": 0.10,
    "residual_scale_init": 0.10,
    "residual_scale_max": 0.25,
}


def _validate_shared_config(
    source_config: Mapping[str, Any],
    target: QwenStemProgressiveOpticalImageNetBackbone,
) -> None:
    mismatches: list[str] = []
    for key, default in _SHARED_CONFIG_DEFAULTS.items():
        source_value = source_config.get(key, default)
        target_value = target.config.get(key, default)
        if source_value != target_value:
            mismatches.append(
                f"{key}: source={source_value!r}, target={target_value!r}"
            )
    if mismatches:
        raise RuntimeError(
            "P13 source/target reusable architecture configs differ: "
            + "; ".join(mismatches)
        )


def _official_p13_training_payload(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    checkpoint: str | Path,
) -> tuple[
    Path,
    Mapping[str, Any],
    dict[str, torch.Tensor],
    dict[str, Any],
    QwenStemProgressiveOpticalImageNetBackbone,
]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing P13 training checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("P13 training checkpoint must contain a mapping")
    if payload.get("format") != P13_TRAINING_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"Expected P13 checkpoint format {P13_TRAINING_CHECKPOINT_FORMAT!r}"
        )
    if payload.get("checkpoint_role") != "best_full_depth":
        raise RuntimeError(
            "Progressive growth requires the formal best_full_depth checkpoint"
        )
    state = _tensor_state(payload.get("model"), name="P13 complete model")
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise RuntimeError("P13 training checkpoint is missing model_config")
    source_config = dict(raw_config)
    source_depth = source_config.get("num_stages")
    if not isinstance(source_depth, int) or source_depth not in P13_SUPPORTED_DEPTHS:
        raise RuntimeError("P13 source model_config has an unsupported num_stages")
    if (source_depth, target.num_stages) not in P13_DEPTH_TRANSITIONS:
        raise RuntimeError(
            f"P13 checkpoint depth transition {(source_depth, target.num_stages)} "
            f"is not one of {P13_DEPTH_TRANSITIONS}"
        )
    if _signature(state, P13_SIGNATURE_KEY) != (13, 1, 2, source_depth):
        raise RuntimeError("P13 source state architecture signature is invalid")
    source_stem_sha = _require_nonempty_string(
        payload.get("stem_checkpoint_sha256"),
        name="P13 source stem_checkpoint_sha256",
    )
    if source_stem_sha != target.stem.checkpoint_sha256:
        raise RuntimeError("P13 source and target stem SHA-256 values differ")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise RuntimeError("P13 training checkpoint must contain a positive epoch")
    _require_nonempty_string(payload.get("config_digest"), name="P13 config_digest")
    report = payload.get("model_report")
    if not isinstance(report, Mapping):
        raise RuntimeError("P13 training checkpoint is missing model_report")
    if report.get("architecture") != "p13_progressive_p11_token_channel":
        raise RuntimeError("P13 source model_report architecture is invalid")
    if report.get("num_stages") != source_depth:
        raise RuntimeError("P13 source model_report depth is inconsistent")
    depth_alpha = payload.get("depth_alpha")
    if (
        not isinstance(depth_alpha, Mapping)
        or depth_alpha.get("all_full_depth") is not True
    ):
        raise RuntimeError(
            "P13 source checkpoint metadata is not at alpha-one full depth"
        )
    if report.get("depth_alpha") != depth_alpha:
        raise RuntimeError("P13 source alpha metadata disagrees with model_report")
    _validate_shared_config(source_config, target)

    source = QwenStemProgressiveOpticalImageNetBackbone(
        target.stem_checkpoint,
        source_config,
    )
    source.load_state_dict(state, strict=True)
    actual_depth_alpha = source.depth_alpha_report()
    if actual_depth_alpha["all_full_depth"] is not True:
        raise RuntimeError("P13 parent checkpoint is not at alpha-one full depth")
    if dict(depth_alpha) != actual_depth_alpha:
        raise RuntimeError("P13 source alpha metadata disagrees with model state")
    if source.growth_parent_depth >= source.num_stages:
        raise RuntimeError("P13 parent growth provenance is not a shallower embedding")
    reported_growth = report.get("growth_parent_stage_indices_zero_based")
    actual_growth = [
        int(value)
        for value in source.growth_parent_stage_indices.detach().cpu().tolist()
    ]
    if reported_growth != actual_growth:
        raise RuntimeError("P13 source model_report growth provenance is inconsistent")
    return path, payload, state, source_config, source


def migrate_strict_progressive_checkpoint(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Grow one trained full-depth P13 checkpoint to the next supported depth."""

    source_path, payload, state, _, source = _official_p13_training_payload(
        target,
        checkpoint,
    )
    source_depth = source.num_stages
    stage_mapping = progressive_stage_mapping(source_depth, target.num_stages)
    target.stem.load_state_dict(source.stem.state_dict(), strict=True)
    target.adapter.load_state_dict(source.adapter.state_dict(), strict=True)
    target.readout.load_state_dict(source.readout.state_dict(), strict=True)

    growth = [-1] * target.num_stages
    mapping: list[dict[str, Any]] = []
    for source_index, target_index in enumerate(stage_mapping):
        source_slot = source.slots[source_index]
        target_slot = target.slots[target_index]
        if source_slot.stage.optical_axis != target_slot.stage.optical_axis:
            raise RuntimeError("P13 progressive mapping changed a stage optical axis")
        if source_slot.is_p11_mixer_anchor != target_slot.is_p11_mixer_anchor:
            raise RuntimeError("P13 progressive mapping changed mixer-anchor structure")
        if source_slot.p11_source_stage_index != target_slot.p11_source_stage_index:
            raise RuntimeError("P13 progressive mapping changed P11 anchor provenance")
        if source_slot.alpha_value != 1.0:
            raise RuntimeError("A carried P13 source stage does not have alpha one")
        target_slot.stage.load_state_dict(source_slot.stage.state_dict(), strict=True)
        growth[target_index] = source_index
        mapping.append(
            {
                "source_stage_zero_based": source_index,
                "target_stage_zero_based": target_index,
                "axis": source_slot.stage.optical_axis,
                "is_p11_mixer_anchor": source_slot.is_p11_mixer_anchor,
                "p11_source_stage_zero_based": source_slot.p11_source_stage_index,
            }
        )
    target.set_growth_parent_stage_indices(growth, new_stage_alpha=0.0)

    source_phase_hash = phase_sequence_sha256(
        [slot.stage.raw_phase for slot in source.slots]
    )
    target_carried_phase_hash = phase_sequence_sha256(
        [slot.stage.raw_phase for slot in target.carried_slots()]
    )
    if source_phase_hash != target_carried_phase_hash:
        raise RuntimeError("P13 carried phases do not hash-match the parent depth")
    source_sha = sha256_file(source_path)
    feedback_source = target.capture_feedback_source(
        provenance={
            "capture": "immediately_after_strict_p13_progressive_migration",
            "parent_checkpoint": str(source_path),
            "parent_checkpoint_sha256": source_sha,
            "parent_depth": source_depth,
            "target_depth": target.num_stages,
            "new_stage_phase_initialization": "deterministic_target_seed_schedule",
            "full_target_depth_captured": target.num_stages,
        }
    )
    manifest: dict[str, Any] = {
        "format": PROGRESSIVE_MIGRATION_FORMAT,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "source_model_state_sha256": state_dict_sha256(state),
        "source_config_digest": payload["config_digest"],
        "source_epoch": int(payload["epoch"]),
        "source_stem_checkpoint_sha256": payload["stem_checkpoint_sha256"],
        "source_architecture_signature": [13, 1, 2, source_depth],
        "target_architecture_signature": [13, 1, 2, target.num_stages],
        "source_depth": source_depth,
        "target_num_stages": target.num_stages,
        "stage_mapping": mapping,
        "growth_parent_stage_indices_zero_based": [
            int(value)
            for value in target.growth_parent_stage_indices.detach().cpu().tolist()
        ],
        "source_phase_sequence_sha256": source_phase_hash,
        "target_carried_phase_sequence_sha256": target_carried_phase_hash,
        "adapter_migrated": True,
        "stem_buffers_migrated": True,
        "source_imagenet_head_migrated": True,
        "new_stage_phase_initialization": "deterministic target seed schedule",
        "new_stage_depth_alpha": target.depth_alpha_report(),
        "full_depth_feedback_source": feedback_source,
        "parent_migration_manifest": payload.get("migration_manifest"),
    }
    target.migration_manifest = manifest
    return manifest


def save_migrated_prototype(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    checkpoint: str | Path,
    output_directory: str | Path,
    *,
    training_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize a migrated P11 initialization and JSON audit manifest."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = (
        migrate_strict_p11_checkpoint(target, checkpoint)
        if training_checkpoint is None
        else migrate_strict_p11_training_checkpoint(
            target,
            checkpoint,
            training_checkpoint,
        )
    )
    report = target.parameter_report()
    checkpoint_path = output / "p13_migrated_initialization.pt"
    checkpoint_payload: dict[str, Any] = {
        "format": "p13-progressive-backbone-initialization-v1",
        "backbone": target.backbone_state_dict(),
        "config": target.config,
        "migration_manifest": manifest,
        "feedback_manifest": target.feedback_manifest(),
        "model_report": report,
    }
    if training_checkpoint is not None:
        checkpoint_payload["model"] = target.state_dict()
    torch.save(checkpoint_payload, checkpoint_path)
    complete_manifest = {
        "migration": manifest,
        "model": report,
        "prototype_checkpoint": str(checkpoint_path),
        "prototype_checkpoint_sha256": sha256_file(checkpoint_path),
        "formal_training_started": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return complete_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly migrate P11 into a function-preserving P13 prototype"
    )
    parser.add_argument("--stem-checkpoint", type=Path, required=True)
    parser.add_argument("--p11-checkpoint", type=Path, required=True)
    parser.add_argument("--p11-training-checkpoint", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--num-stages", type=int, choices=P13_SUPPORTED_DEPTHS, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--new-stage-alpha-init", type=float, default=0.0)
    parser.add_argument("--new-stage-alpha-epsilon", type=float, default=0.01)
    parser.add_argument("--new-stage-ramp-epochs", type=int, default=10)
    parser.add_argument("--activation-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = QwenStemProgressiveOpticalImageNetBackbone(
        args.stem_checkpoint,
        {
            "num_stages": args.num_stages,
            "seed": args.seed,
            "new_stage_alpha_init": args.new_stage_alpha_init,
            "new_stage_alpha_epsilon": args.new_stage_alpha_epsilon,
            "new_stage_ramp_epochs": args.new_stage_ramp_epochs,
            "activation_checkpointing": args.activation_checkpointing,
        },
    )
    manifest = save_migrated_prototype(
        model,
        args.p11_checkpoint,
        args.output_directory,
        training_checkpoint=args.p11_training_checkpoint,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
