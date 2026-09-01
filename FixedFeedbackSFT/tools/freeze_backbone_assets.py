#!/usr/bin/env python3
"""Freeze the official P11 (8-stage) and P13 (16-stage) backbone assets.

The source runs and frozen Qwen stem must be presented through symlinks below
``FixedFeedbackSFT/runs``.  A snapshot directory is committed atomically and is
immutable from this tool's point of view: an identical content identity is a
no-op, while any different or damaged existing snapshot is rejected.

This utility intentionally does not discover historical worktrees.  The caller
must register those locations in the central runs tree first, which keeps all
new project state below ``2026OpticsMoE/FixedFeedbackSFT``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from FixedFeedbackSFT.paths import RUNS_ROOT  # noqa: E402


SNAPSHOT_FORMAT = "fixed-feedback-backbone-asset-snapshot-v1"
ASSET_DIRECTORY_NAME = "_assets"
SHA256_HEX = frozenset("0123456789abcdef")
OFFICIAL_STEM_SHA256 = (
    "e3b12b274211d29f928eee95fdfc60b32d10751f1bbdc98cd63f0cccd0792485"
)

# This is the model-only reconstruction contract used by the P13 migration
# guard.  It is deliberately local to this new utility so freezing never
# imports the P13 trainer or mutates its implementation-manifest file set.
OFFICIAL_P11_MODEL_CONFIG: dict[str, Any] = {
    "canvas_size": 224,
    "optical_channels": 3,
    "token_dim": 224,
    "num_classes": 1000,
    "head_hidden_dim": 448,
    "wavelength_m": 5.32e-7,
    "pixel_size_m": 1.6e-5,
    "propagation_distance_m": 0.05,
    "token_axis_propagation_distance_m": 0.05,
    "channel_axis_propagation_distance_m": 0.05,
    "phase_init_std": 0.10,
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
    "seed": 2026,
}


class SnapshotError(RuntimeError):
    """Base error for a refused or invalid snapshot operation."""


class ImmutableSnapshotConflict(SnapshotError):
    """Raised when an existing snapshot is not the requested intact content."""


@dataclass(frozen=True)
class RunAsset:
    source_relative: str
    destination_relative: str
    role: str
    required: bool = True
    checkpoint: bool = False


@dataclass(frozen=True)
class StageSpec:
    label: str
    num_stages: int
    project: str
    run_name: str
    config_digest: str
    backbone_sha256: str
    training_sha256: str
    assets: tuple[RunAsset, ...]


P11_SPEC = StageSpec(
    label="8stage",
    num_stages=8,
    project="qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone",
    run_name="p11_imagenet1k_pretrain_bs96_90e",
    config_digest="c588b9ead9661b5bc513f00349681895979729d8c46b08bba72a109b6d5c74fa",
    backbone_sha256="c3ad0b780dfbb3e5f8e1f7b7850c06fcb5c6d977e106f351b4602fcaadf210d2",
    training_sha256="a30d5c06b61a635bb3dc379aeaca4c371c1d27e6b862c5ffd4977ce738b33034",
    assets=(
        RunAsset(
            "checkpoints/backbone.pt",
            "checkpoints/backbone.pt",
            "backbone_export",
            checkpoint=True,
        ),
        RunAsset(
            "checkpoints/best.pt",
            "checkpoints/best.pt",
            "best_training_checkpoint",
            checkpoint=True,
        ),
        RunAsset("manifest.json", "run_metadata/manifest.json", "run_manifest", False),
        RunAsset("result.json", "run_metadata/result.json", "terminal_result", False),
        RunAsset(
            "metrics/history.json",
            "run_metadata/metrics/history.json",
            "metric_history",
            False,
        ),
        RunAsset(
            "metrics/latest.json",
            "run_metadata/metrics/latest.json",
            "latest_metric",
            False,
        ),
        RunAsset(
            "metrics/initial_baseline.json",
            "run_metadata/metrics/initial_baseline.json",
            "initial_baseline",
            False,
        ),
    ),
)

P13_SPEC = StageSpec(
    label="16stage",
    num_stages=16,
    project="qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone",
    run_name="p13_growth16_fa_source_20e_gb192",
    config_digest="c5aa8c11d0d2ee711adaa09263ddf91cf1c81fcb56944b7e76612f1cb1775837",
    backbone_sha256="80b9b7b0f4415fd789bf46312dc23ccaf3600b5c4df9885c2972ce465dc9129d",
    training_sha256="97ae579e9b33a2fc5825debde2fbad8d4fbddcbf91df5e39584036eb125ed6ec",
    assets=(
        RunAsset(
            "checkpoints/backbone_full_depth.pt",
            "checkpoints/backbone_full_depth.pt",
            "backbone_export",
            checkpoint=True,
        ),
        RunAsset(
            "checkpoints/best_full_depth.pt",
            "checkpoints/best_full_depth.pt",
            "best_training_checkpoint",
            checkpoint=True,
        ),
        RunAsset("manifest.json", "run_metadata/manifest.json", "run_manifest"),
        RunAsset("result.json", "run_metadata/result.json", "terminal_result"),
        RunAsset(
            "metrics/history.json",
            "run_metadata/metrics/history.json",
            "metric_history",
        ),
        RunAsset(
            "metrics/latest.json",
            "run_metadata/metrics/latest.json",
            "latest_metric",
        ),
        RunAsset(
            "metrics/initial_baseline.json",
            "run_metadata/metrics/initial_baseline.json",
            "initial_baseline",
            False,
        ),
    ),
)


def _lexical_absolute(path: str | Path, *, base: Path = REPOSITORY_ROOT) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base / candidate
    # abspath normalizes '..' without following the registered symlink.
    return Path(os.path.abspath(candidate))


def _relative_to(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise SnapshotError(f"Source must stay below the central runs root: {path}") from exc


def _safe_snapshot_relative(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SnapshotError(f"Unsafe snapshot-relative path: {value!r}")
    return relative


def _require_registered_link(path: str | Path, runs_root: Path, *, kind: str) -> Path:
    lexical = _lexical_absolute(path)
    relative = _relative_to(lexical, runs_root)
    if not relative.parts or relative.parts[0] == ASSET_DIRECTORY_NAME:
        raise SnapshotError(f"{kind} cannot be the runs root or a frozen asset: {lexical}")
    if not lexical.is_symlink():
        raise SnapshotError(
            f"{kind} must be a registered symlink inside FixedFeedbackSFT/runs: {lexical}"
        )
    if not lexical.exists():
        raise SnapshotError(f"{kind} symlink is broken: {lexical}")
    return lexical


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in SHA256_HEX for char in value)
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _infer_num_stages(payload: Mapping[str, Any]) -> int | None:
    candidates = (
        _nested(payload, "model_config", "num_stages"),
        _nested(payload, "model_report", "num_stages"),
        _nested(payload, "feedback", "manifest", "depth"),
    )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    depth = payload.get("depth_alpha")
    if isinstance(depth, Mapping):
        carried = depth.get("carried_stage_count")
        new = depth.get("new_stage_count")
        if all(isinstance(value, int) and not isinstance(value, bool) for value in (carried, new)):
            return int(carried) + int(new)
    return None


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Read a trusted project checkpoint and return JSON-safe audit metadata."""

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SnapshotError(f"Checkpoint root is not a mapping: {path}")
    state_name = "model" if isinstance(payload.get("model"), Mapping) else "backbone"
    state = payload.get(state_name)
    if not isinstance(state, Mapping):
        raise SnapshotError(f"Checkpoint has neither model nor backbone state: {path}")
    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    if not tensors:
        raise SnapshotError(f"Checkpoint state contains no tensors: {path}")
    tensor_audit = {
        "tensor_count": len(tensors),
        "parameter_elements": sum(int(value.numel()) for value in tensors),
        "all_nonempty": all(value.numel() > 0 for value in tensors),
        "all_finite": all(bool(torch.isfinite(value).all()) for value in tensors),
    }
    if not tensor_audit["all_nonempty"] or not tensor_audit["all_finite"]:
        raise SnapshotError(f"Checkpoint tensor integrity failed: {path}")
    implementation_sha = _nested(
        payload, "implementation_manifest", "aggregate_sha256"
    )
    return {
        "format": payload.get("format"),
        "checkpoint_role": payload.get("checkpoint_role"),
        "epoch": payload.get("epoch", payload.get("best_epoch")),
        "config_digest": payload.get("config_digest"),
        "stem_checkpoint_sha256": payload.get("stem_checkpoint_sha256"),
        "num_stages": _infer_num_stages(payload),
        "state_scope": state_name,
        "tensor_audit": tensor_audit,
        "implementation_aggregate_sha256": implementation_sha,
        "strict_load_capability": {
            "state_dict_available": True,
            "state_scope": state_name,
            "model_config_embedded": isinstance(payload.get("model_config"), Mapping),
            "stem_digest_embedded": _valid_sha256(payload.get("stem_checkpoint_sha256")),
            "implementation_manifest_embedded": isinstance(
                payload.get("implementation_manifest"), Mapping
            ),
        },
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"Invalid JSON evidence file: {path}") from exc


def summarize_metrics(run_link: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    history_path = run_link / "metrics" / "history.json"
    if history_path.is_file():
        history = _load_json(history_path)
        if not isinstance(history, list):
            raise SnapshotError(f"Metric history must be a list: {history_path}")
        rows = [row for row in history if isinstance(row, Mapping)]
        validation_rows: list[tuple[int | None, float, float | None]] = []
        for row in rows:
            validation = row.get("validation")
            if not isinstance(validation, Mapping):
                continue
            top1 = _finite_number(validation.get("top1_accuracy"))
            top5 = _finite_number(validation.get("top5_accuracy"))
            epoch = row.get("epoch")
            epoch_value = int(epoch) if isinstance(epoch, int) and not isinstance(epoch, bool) else None
            if top1 is not None:
                validation_rows.append((epoch_value, top1, top5))
        history_summary: dict[str, Any] = {
            "row_count": len(rows),
            "first_epoch": rows[0].get("epoch") if rows else None,
            "last_epoch": rows[-1].get("epoch") if rows else None,
        }
        if validation_rows:
            best = max(validation_rows, key=lambda item: item[1])
            last = validation_rows[-1]
            history_summary.update(
                {
                    "best_validation_epoch": best[0],
                    "best_validation_top1": best[1],
                    "best_validation_top5": best[2],
                    "last_validation_epoch": last[0],
                    "last_validation_top1": last[1],
                    "last_validation_top5": last[2],
                }
            )
        summary["history"] = history_summary
    result_path = run_link / "result.json"
    if result_path.is_file():
        result = _load_json(result_path)
        if not isinstance(result, Mapping):
            raise SnapshotError(f"Terminal result must be a mapping: {result_path}")
        validation = result.get("best_full_depth_validation", result.get("best_validation"))
        summary["terminal_result"] = {
            "status": result.get("status"),
            "best_epoch": result.get("best_full_depth_epoch", result.get("best_epoch")),
            "best_top1": result.get("best_full_depth_top1"),
            "validation_top1": (
                validation.get("top1_accuracy") if isinstance(validation, Mapping) else None
            ),
            "validation_top5": (
                validation.get("top5_accuracy") if isinstance(validation, Mapping) else None
            ),
        }
    return summary


def _load_payload(path: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SnapshotError(f"Checkpoint root is not a mapping: {path}")
    return payload


def _load_state_exact(model: Any, state: Mapping[str, Any], *, backbone: bool) -> dict[str, Any]:
    incompatible = model.load_state_dict(state, strict=not backbone)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    expected_missing = (
        sorted(name for name in model.state_dict() if name.startswith("readout."))
        if backbone
        else []
    )
    if missing != expected_missing or unexpected:
        raise SnapshotError(
            "Strict model reconstruction failed: "
            f"missing={missing}, expected_missing={expected_missing}, unexpected={unexpected}"
        )
    return {
        "compatible": True,
        "strict_argument": not backbone,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "expected_missing_keys": expected_missing,
    }


def strict_load_stage(spec: StageSpec, run_link: Path, stem_link: Path) -> dict[str, Any]:
    """Reconstruct the live model and check both preserved checkpoint scopes."""

    if spec.num_stages == 8:
        from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
            QwenStemSeparableOpticalImageNetBackbone,
        )

        training_path = run_link / "checkpoints" / "best.pt"
        backbone_path = run_link / "checkpoints" / "backbone.pt"
        config = dict(OFFICIAL_P11_MODEL_CONFIG)
        model_class_name = (
            "experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone."
            "model.QwenStemSeparableOpticalImageNetBackbone"
        )
        model = QwenStemSeparableOpticalImageNetBackbone(stem_link, config)
    elif spec.num_stages == 16:
        from experiments.qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone.model import (
            QwenStemProgressiveOpticalImageNetBackbone,
        )

        training_path = run_link / "checkpoints" / "best_full_depth.pt"
        backbone_path = run_link / "checkpoints" / "backbone_full_depth.pt"
        training_probe = _load_payload(training_path)
        model_config = training_probe.get("model_config")
        if not isinstance(model_config, Mapping):
            raise SnapshotError("P13 training checkpoint has no model_config")
        config = dict(model_config)
        model_class_name = (
            "experiments.qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone."
            "model.QwenStemProgressiveOpticalImageNetBackbone"
        )
        del training_probe
        model = QwenStemProgressiveOpticalImageNetBackbone(stem_link, config)
    else:
        raise SnapshotError(f"Unsupported strict-load depth: {spec.num_stages}")

    if int(model.num_stages) != spec.num_stages:
        raise SnapshotError(
            f"Live model depth {model.num_stages} differs from {spec.num_stages}"
        )
    live_stem_sha = model.stem.checkpoint_sha256
    if live_stem_sha != OFFICIAL_STEM_SHA256:
        raise SnapshotError("Live model loaded a non-official stem checkpoint")

    training = _load_payload(training_path)
    training_state = training.get("model")
    if not isinstance(training_state, Mapping):
        raise SnapshotError("Best training checkpoint has no full model state")
    training_result = _load_state_exact(model, training_state, backbone=False)
    del training, training_state

    backbone = _load_payload(backbone_path)
    backbone_state = backbone.get("backbone")
    if not isinstance(backbone_state, Mapping):
        raise SnapshotError("Backbone export has no backbone state")
    backbone_result = _load_state_exact(model, backbone_state, backbone=True)
    del backbone, backbone_state, model
    return {
        "executed": True,
        "compatible": True,
        "live_num_stages": spec.num_stages,
        "live_stem_checkpoint_sha256": live_stem_sha,
        "model_class": model_class_name,
        "reconstruction_model_config": config,
        "reconstruction_model_config_sha256": canonical_sha256(config),
        "best_training_checkpoint": training_result,
        "backbone_export": backbone_result,
    }


def _source_link_record(link: Path, runs_root: Path) -> dict[str, Any]:
    return {
        "path_below_runs": link.relative_to(runs_root).as_posix(),
        "link_target_at_freeze": os.readlink(link),
    }


def _checkpoint_identity_guard(
    spec: StageSpec,
    file_records: Sequence[Mapping[str, Any]],
    checkpoint_metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    by_role = {str(record["role"]): record for record in file_records}
    expected_hashes = {
        "backbone_export": spec.backbone_sha256,
        "best_training_checkpoint": spec.training_sha256,
        "frozen_qwen_stem": OFFICIAL_STEM_SHA256,
    }
    for role, expected in expected_hashes.items():
        record = by_role.get(role)
        if record is None or record.get("sha256") != expected:
            actual = None if record is None else record.get("sha256")
            raise SnapshotError(
                f"{spec.label} {role} SHA-256 mismatch: expected={expected}, actual={actual}"
            )
    for role in ("backbone_export", "best_training_checkpoint"):
        metadata = checkpoint_metadata.get(role)
        if not isinstance(metadata, Mapping):
            raise SnapshotError(f"Missing checkpoint metadata for {spec.label} {role}")
        digest = metadata.get("config_digest")
        if digest != spec.config_digest:
            raise SnapshotError(
                f"{spec.label} {role} config digest mismatch: {digest!r}"
            )
        depth = metadata.get("num_stages")
        if role == "backbone_export" and depth is None:
            raise SnapshotError(f"{spec.label} backbone export does not declare its depth")
        if depth is not None and depth != spec.num_stages:
            raise SnapshotError(f"{spec.label} {role} declares depth {depth}")
        stem = metadata.get("stem_checkpoint_sha256")
        if role == "backbone_export" and stem is None:
            raise SnapshotError(f"{spec.label} backbone export does not declare its stem")
        if stem is not None and stem != OFFICIAL_STEM_SHA256:
            raise SnapshotError(f"{spec.label} {role} declares a different stem")


def prepare_snapshot(
    spec: StageSpec,
    run_link: Path,
    stem_link: Path,
    runs_root: Path,
    *,
    checkpoint_inspector: Callable[[Path], dict[str, Any]] = inspect_checkpoint,
    strict_loader: Callable[[StageSpec, Path, Path], dict[str, Any]] = strict_load_stage,
) -> tuple[dict[str, Any], list[tuple[Path, str]]]:
    file_records: list[dict[str, Any]] = []
    copy_sources: list[tuple[Path, str]] = []
    checkpoint_metadata: dict[str, dict[str, Any]] = {}
    for asset in spec.assets:
        source = run_link / Path(asset.source_relative)
        if not source.is_file():
            if asset.required:
                raise SnapshotError(f"Missing required {spec.label} asset: {source}")
            continue
        digest = sha256_file(source)
        record = {
            "role": asset.role,
            "path": asset.destination_relative,
            "sha256": digest,
            "size_bytes": source.stat().st_size,
        }
        file_records.append(record)
        copy_sources.append((source, asset.destination_relative))
        if asset.checkpoint:
            checkpoint_metadata[asset.role] = checkpoint_inspector(source)

    stem_destination = "dependencies/qwen3_vl_static_stem_224.pt"
    stem_record = {
        "role": "frozen_qwen_stem",
        "path": stem_destination,
        "sha256": sha256_file(stem_link),
        "size_bytes": stem_link.stat().st_size,
    }
    file_records.append(stem_record)
    copy_sources.append((stem_link, stem_destination))
    _checkpoint_identity_guard(spec, file_records, checkpoint_metadata)

    strict_load = strict_loader(spec, run_link, stem_link)
    if not strict_load.get("executed") or not strict_load.get("compatible"):
        raise SnapshotError(f"Strict live-model load did not pass for {spec.label}")
    if strict_load.get("live_num_stages") != spec.num_stages:
        raise SnapshotError(f"Strict live-model depth did not match for {spec.label}")
    if strict_load.get("live_stem_checkpoint_sha256") != OFFICIAL_STEM_SHA256:
        raise SnapshotError(f"Strict live-model stem did not match for {spec.label}")
    metrics = summarize_metrics(run_link)
    content_identity = {
        "format": SNAPSHOT_FORMAT,
        "label": spec.label,
        "num_stages": spec.num_stages,
        "config_digest": spec.config_digest,
        "stem_checkpoint_sha256": OFFICIAL_STEM_SHA256,
        "files": sorted(file_records, key=lambda record: str(record["path"])),
    }
    manifest = {
        **content_identity,
        "content_identity_sha256": canonical_sha256(content_identity),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "run": _source_link_record(run_link, runs_root),
            "stem": _source_link_record(stem_link, runs_root),
            "policy": "sources accessed only through registered links below FixedFeedbackSFT/runs",
        },
        "checkpoint_metadata": checkpoint_metadata,
        "metrics": metrics,
        "strict_load": strict_load,
        "immutability": {
            "same_content_identity": "idempotent no-op",
            "different_content_or_damage": "refuse overwrite",
            "commit": "sibling staging directory followed by atomic rename",
        },
    }
    return manifest, copy_sources


def _write_bytes_fsync(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=4 * 1024 * 1024)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise SnapshotError(
            f"Source changed while freezing {source}: expected={expected_sha256}, copied={actual}"
        )


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def verify_existing_snapshot(destination: Path) -> dict[str, Any]:
    manifest_path = destination / "manifest.json"
    checksum_path = destination / "manifest.sha256"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise ImmutableSnapshotConflict(
            f"Existing snapshot is incomplete and will not be overwritten: {destination}"
        )
    checksum_parts = checksum_path.read_text(encoding="ascii").strip().split()
    if len(checksum_parts) != 2 or checksum_parts[1] != "manifest.json":
        raise ImmutableSnapshotConflict(f"Invalid manifest.sha256: {checksum_path}")
    actual_manifest_sha = sha256_file(manifest_path)
    if checksum_parts[0] != actual_manifest_sha:
        raise ImmutableSnapshotConflict(f"Manifest checksum mismatch: {destination}")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("format") != SNAPSHOT_FORMAT:
        raise ImmutableSnapshotConflict(f"Unknown snapshot manifest: {destination}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ImmutableSnapshotConflict(f"Snapshot has no file inventory: {destination}")
    for record in files:
        if not isinstance(record, Mapping):
            raise ImmutableSnapshotConflict(f"Malformed file inventory: {destination}")
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not _valid_sha256(expected):
            raise ImmutableSnapshotConflict(f"Malformed file identity: {destination}")
        try:
            safe_relative = _safe_snapshot_relative(relative)
        except SnapshotError as exc:
            raise ImmutableSnapshotConflict(str(exc)) from exc
        path = destination / safe_relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ImmutableSnapshotConflict(
                f"Frozen file is missing or changed; refusing repair: {path}"
            )
    return dict(manifest)


def commit_snapshot(
    destination: Path,
    manifest: Mapping[str, Any],
    copy_sources: Sequence[tuple[Path, str]],
) -> str:
    """Atomically create one snapshot, or verify an identical existing one."""

    expected_identity = manifest.get("content_identity_sha256")
    if not _valid_sha256(expected_identity):
        raise SnapshotError("Prepared manifest has no valid content identity")
    if destination.is_symlink():
        raise ImmutableSnapshotConflict(
            f"Frozen snapshot destination must not be a symlink: {destination}"
        )
    if destination.exists():
        existing = verify_existing_snapshot(destination)
        if existing.get("content_identity_sha256") != expected_identity:
            raise ImmutableSnapshotConflict(
                f"Different content already frozen at {destination}; refusing overwrite"
            )
        return "unchanged"

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        expected_by_path = {
            str(record["path"]): str(record["sha256"])
            for record in manifest["files"]
        }
        for source, relative in copy_sources:
            safe_relative = _safe_snapshot_relative(relative)
            _copy_verified(source, staging / safe_relative, expected_by_path[relative])
        manifest_content = _manifest_bytes(manifest)
        _write_bytes_fsync(staging / "manifest.json", manifest_content)
        manifest_sha = sha256_file(staging / "manifest.json")
        _write_bytes_fsync(
            staging / "manifest.sha256",
            f"{manifest_sha}  manifest.json\n".encode("ascii"),
        )
        try:
            os.replace(staging, destination)
        except OSError:
            # A concurrent freezer may have won the atomic rename.
            if destination.exists():
                existing = verify_existing_snapshot(destination)
                if existing.get("content_identity_sha256") == expected_identity:
                    return "unchanged"
            raise
        return "created"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def freeze_official_assets(
    runs_root: Path,
    eight_run: str | Path,
    sixteen_run: str | Path,
    stem_link: str | Path,
) -> dict[str, Any]:
    runs_root = _lexical_absolute(runs_root)
    official_runs_root = _lexical_absolute(REPOSITORY_ROOT / "FixedFeedbackSFT" / "runs")
    if runs_root != official_runs_root:
        raise SnapshotError(
            "Runs root must be the current repository's exact "
            f"FixedFeedbackSFT/runs directory: expected={official_runs_root}, "
            f"actual={runs_root}"
        )
    eight_link = _require_registered_link(eight_run, runs_root, kind="P11 run")
    sixteen_link = _require_registered_link(sixteen_run, runs_root, kind="P13 run")
    stem_source = _require_registered_link(stem_link, runs_root, kind="Qwen stem")
    if eight_link.is_file() or sixteen_link.is_file() or not stem_source.is_file():
        raise SnapshotError("Run links must target directories and the stem link must target a file")

    prepared = [
        (P11_SPEC, *prepare_snapshot(P11_SPEC, eight_link, stem_source, runs_root)),
        (P13_SPEC, *prepare_snapshot(P13_SPEC, sixteen_link, stem_source, runs_root)),
    ]
    asset_root = runs_root / ASSET_DIRECTORY_NAME
    if asset_root.is_symlink():
        raise SnapshotError(f"Frozen asset root must not be a symlink: {asset_root}")
    if asset_root.exists() and not asset_root.is_dir():
        raise SnapshotError(f"Frozen asset root is not a directory: {asset_root}")
    results: dict[str, Any] = {}
    for spec, manifest, sources in prepared:
        destination = asset_root / spec.label
        status = commit_snapshot(destination, manifest, sources)
        results[spec.label] = {
            "status": status,
            "path": str(destination),
            "content_identity_sha256": manifest["content_identity_sha256"],
            "manifest_sha256": sha256_file(destination / "manifest.json"),
        }
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    runs_root = _lexical_absolute(RUNS_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=runs_root,
        help="central FixedFeedbackSFT/runs directory (default: %(default)s)",
    )
    parser.add_argument(
        "--eight-run",
        type=Path,
        default=runs_root / P11_SPEC.project / P11_SPEC.run_name,
        help="registered P11 run symlink",
    )
    parser.add_argument(
        "--sixteen-run",
        type=Path,
        default=runs_root / P13_SPEC.project / P13_SPEC.run_name,
        help="registered P13 run symlink",
    )
    parser.add_argument(
        "--stem-link",
        type=Path,
        default=runs_root / "backbone_sources" / "qwen3_vl_static_stem_224.pt",
        help="registered symlink to the frozen Qwen patch/position stem",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = freeze_official_assets(
            args.runs_root, args.eight_run, args.sixteen_run, args.stem_link
        )
    except SnapshotError as exc:
        print(f"freeze refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
