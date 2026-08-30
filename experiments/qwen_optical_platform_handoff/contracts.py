"""Fail-fast validation for a new Qwen + optical task.

The contract is deliberately small.  It does not replace the experiment YAML;
it records the cross-cutting decisions that must remain consistent between
simulation, mask export, physical capture and measured-CCD fine-tuning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when a handoff contract would make an experiment ambiguous."""


TASK_KINDS = {"retrieval", "classification", "quality_regression", "dense_prediction"}
STAGE_ORDER = (
    "vision_expert",
    "vision_global",
    "language_expert",
    "language_global",
)


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"{key} must be an object")
    return value


def _nonempty(mapping: dict[str, Any], key: str, prefix: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ContractError(f"{prefix}.{key} must be non-empty")
    return value


def _positive(mapping: dict[str, Any], key: str, prefix: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"{prefix}.{key} must be numeric") from error
    if not math.isfinite(value) or value <= 0:
        raise ContractError(f"{prefix}.{key} must be finite and positive")
    return value


def validate_contract(raw: dict[str, Any]) -> dict[str, Any]:
    if int(raw.get("schema_version", -1)) != 1:
        raise ContractError("schema_version must be 1")
    project = _mapping(raw, "project")
    task = _mapping(raw, "task")
    data = _mapping(raw, "data")
    optics = _mapping(raw, "optics")
    training = _mapping(raw, "training")
    hardware = _mapping(raw, "hardware")

    _nonempty(project, "name", "project")
    _nonempty(project, "reference_experiment", "project")
    kind = _nonempty(task, "kind", "task")
    if kind not in TASK_KINDS:
        raise ContractError(f"task.kind must be one of {sorted(TASK_KINDS)}")
    losses = task.get("losses")
    metrics = task.get("metrics")
    if not isinstance(losses, list) or not losses or not all(str(v).strip() for v in losses):
        raise ContractError("task.losses must be a non-empty list")
    if not isinstance(metrics, list) or not metrics or not all(str(v).strip() for v in metrics):
        raise ContractError("task.metrics must be a non-empty list")

    split_names = {
        name: _nonempty(data, name, "data")
        for name in ("train_split", "development_split", "test_split")
    }
    if len(set(split_names.values())) != 3:
        raise ContractError("train/development/test splits must be distinct")
    selection_split = _nonempty(training, "selection_split", "training")
    if selection_split != split_names["development_split"]:
        raise ContractError("training.selection_split must equal data.development_split")
    if selection_split == split_names["test_split"]:
        raise ContractError("the test split must never select checkpoints")
    if training.get("sealed_test_once_after_selection") is not True:
        raise ContractError("training.sealed_test_once_after_selection must be true")

    stages = optics.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ContractError("optics.stages must be a non-empty list")
    unknown = [stage for stage in stages if stage not in STAGE_ORDER]
    if unknown:
        raise ContractError(f"unknown optical stages: {unknown}")
    indexes = [STAGE_ORDER.index(stage) for stage in stages]
    if indexes != sorted(set(indexes)):
        raise ContractError("optics.stages must follow canonical order without duplicates")
    _positive(optics, "wavelength_nm", "optics")
    _positive(optics, "propagation_distance_m", "optics")
    _positive(optics, "logical_pixel_pitch_um", "optics")
    active_size = int(_positive(optics, "active_size", "optics"))
    if active_size < 32:
        raise ContractError("optics.active_size is implausibly small")
    robustness = _mapping(optics, "robustness")
    if robustness.get("shared_simulation_and_measurement_normalization") is not True:
        raise ContractError(
            "simulation and measurement must share the detector normalization"
        )

    enabled = hardware.get("enabled")
    if not isinstance(enabled, bool):
        raise ContractError("hardware.enabled must be boolean")
    if enabled:
        amplitude = _mapping(hardware, "amplitude_slm")
        phase = _mapping(hardware, "phase_slm")
        detector = _mapping(hardware, "detector")
        for mapping, prefix in (
            (amplitude, "hardware.amplitude_slm"),
            (phase, "hardware.phase_slm"),
        ):
            resolution = mapping.get("resolution_wh")
            if not (
                isinstance(resolution, list)
                and len(resolution) == 2
                and all(int(v) > 0 for v in resolution)
            ):
                raise ContractError(f"{prefix}.resolution_wh must be [width,height]")
            _positive(mapping, "pixel_pitch_um", prefix)
        if detector.get("saved_size_wh") != [active_size, active_size]:
            raise ContractError(
                "hardware.detector.saved_size_wh must equal the square optical active_size"
            )
        if int(detector.get("saved_bit_depth", -1)) not in {8, 16}:
            raise ContractError("hardware.detector.saved_bit_depth must be 8 or 16")
        if detector.get("per_frame_minmax_normalization") is not False:
            raise ContractError("per-frame detector min-max normalization is forbidden")
        if detector.get("background_subtraction") is True and not str(
            detector.get("measured_background_source", "")
        ).strip():
            raise ContractError(
                "background subtraction requires an actual measured background source"
            )
        if hardware.get("freeze_measured_upstream_stages") is not True:
            raise ContractError("measured upstream stages must be frozen after capture")
    return raw


def load_and_validate_contract(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ContractError(f"invalid JSON contract: {error}") from error
    if not isinstance(raw, dict):
        raise ContractError("contract root must be an object")
    return validate_contract(raw)
