from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_amplitude_with_metadata,
    encode_active_phase,
    reconstruct_directory,
    save_active_png,
)
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import (
    prepare_caltech101_subset,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust import (
    hardware_bridge as legacy_bridge,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    seed_everything,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optical_artifacts import (
    phase_tensors,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    load_checkpoint,
)

from .hardware_contract import (
    FEATURE_STAGES,
    ROUTER_STAGES,
    SIX_STAGES,
    expected_key_files,
    initialize_state,
    read_csv,
    read_json,
    record_event,
    require_empty_directory,
    sha256_canonical_json,
    sha256_file,
    stage_directory,
    validate_state_identity,
    write_csv,
    write_json,
)
from .modeling import architecture_label, build_hybrid_student, load_backbone
from .score_router_ccd import score_directory
from .settings import load_settings


STATE_FILENAME = "six_stage_state.json"
MANIFEST_FILENAME = "manifest.csv"
SAMPLE_MANIFEST_FIELDS = (
    "order",
    "key",
    "sample_id",
    "split",
    "sku_index",
    "sku_name",
    "image_path",
    "image_sha256",
)


def _validate_hardware_geometry(settings: Any) -> dict[str, Any]:
    if int(settings.num_experts) != 4 or (
        int(settings.expert_grid_rows), int(settings.expert_grid_cols)
    ) != (2, 2):
        raise RuntimeError("Six-stage Router bridge requires the unchanged 2x2 MoE4")
    if int(settings.expert_pitch) + int(settings.expert_size) != int(
        settings.active_size
    ):
        raise RuntimeError(
            "MoE4 geometry changed: expert_pitch + expert_size must equal active_size"
        )
    if int(settings.active_size) != 478 or int(settings.expert_size) != 224:
        raise RuntimeError("Audited hardware geometry is fixed to active478/expert224")
    if int(settings.hardware_ccd_target_size) != int(settings.active_size):
        raise RuntimeError("CCD target and model active ROI must both be 478")
    if bool(settings.hardware_ccd_flip_vertical) or bool(
        settings.hardware_ccd_flip_horizontal
    ):
        raise RuntimeError(
            "Six-stage CCD files are already canonical after detector homography; "
            "hardware.ccd downstream flips must both be false"
        )
    if bool(settings.hardware_amplitude_invert_before_export) or (
        int(settings.hardware_amplitude_bright_value_uint8),
        int(settings.hardware_amplitude_dark_value_uint8),
    ) != (255, 0):
        raise RuntimeError(
            "Hardware bridge requires corrected amplitude polarity 255=bright, 0=dark"
        )

    logical_pitch = float(settings.language_optical_pixel_pitch_um)

    def bounds(
        logical_size: int,
        *,
        native_pitch: float,
        width: int,
        height: int,
        center_x: float,
        center_y: float,
    ) -> tuple[int, int, int, int, int]:
        native_size = int(round(logical_size * logical_pitch / native_pitch))
        left = int(np.floor(center_x - native_size / 2.0 + 0.5))
        top = int(np.floor(center_y - native_size / 2.0 + 0.5))
        right, bottom = left + native_size, top + native_size
        if left < 0 or top < 0 or right > width or bottom > height:
            raise RuntimeError(
                f"Logical active{logical_size} maps outside native SLM: "
                f"bounds={(left, top, right, bottom)}, canvas={(width, height)}"
            )
        return native_size, left, top, right, bottom

    amplitude = bounds(
        int(settings.active_size),
        native_pitch=float(settings.hardware_amplitude_slm_pixel_pitch_um),
        width=int(settings.hardware_amplitude_slm_width),
        height=int(settings.hardware_amplitude_slm_height),
        center_x=float(settings.hardware_amplitude_slm_center_x),
        center_y=float(settings.hardware_amplitude_slm_center_y),
    )
    phase = bounds(
        int(settings.active_size),
        native_pitch=float(settings.hardware_phase_slm_pixel_pitch_um),
        width=int(settings.hardware_phase_slm_width),
        height=int(settings.hardware_phase_slm_height),
        center_x=float(settings.hardware_phase_slm_center_x),
        center_y=float(settings.hardware_phase_slm_center_y),
    )
    return {
        "logical_active_size": int(settings.active_size),
        "expert_size": int(settings.expert_size),
        "expert_pitch": int(settings.expert_pitch),
        "amplitude_native_active_size_and_bounds_xyxy": list(amplitude),
        "phase_native_active_size_and_bounds_xyxy": list(phase),
        "phase_flip_vertical": bool(settings.hardware_phase_flip_vertical),
        "phase_flip_horizontal": bool(settings.hardware_phase_flip_horizontal),
    }


def _resolved_hardware_contract(settings: Any) -> dict[str, Any]:
    return {
        "checkpoint_architecture": architecture_label(settings),
        "router": {
            "backend": str(settings.router_backend),
            "top_k": int(settings.top_k),
            "temperature": float(settings.router_temperature),
            "weight_normalization": str(settings.router_weight_normalization),
            "score_normalization": str(settings.optical_router_score_normalization),
            "detector_intervals_half_open": [
                list(value) for value in settings.optical_router_detector_intervals
            ],
        },
        "propagation": {
            "canvas_size": int(settings.canvas_size),
            "active_size": int(settings.active_size),
            "wavelength_nm": float(settings.language_optical_wavelength_nm),
            "logical_pixel_pitch_um": float(settings.language_optical_pixel_pitch_um),
            "distance_m": float(settings.language_optical_distance_m),
            "k_space_enabled": bool(settings.language_optical_k_space_enabled),
            "theta_max_deg": float(settings.language_optical_theta_max_deg),
        },
        # These thresholds govern acceptance of measured files only. They are
        # intentionally outside router_contract_sha256/checkpoint architecture,
        # but are sealed into the six-stage session contract.
        "measurement_quality": {
            "maximum_saturated_pixel_fraction": float(
                settings.optical_router_maximum_saturated_pixel_fraction
            ),
            "minimum_p99_uint8": float(settings.optical_router_minimum_p99_uint8),
            "minimum_dynamic_range_uint8": float(
                settings.optical_router_minimum_dynamic_range_uint8
            ),
            "minimum_topk_probability_margin": float(
                settings.optical_router_minimum_topk_probability_margin
            ),
            "background_subtraction": False,
            "uncalibrated_capture_metric": "raw_capture_fraction",
        },
        "hardware_geometry": _validate_hardware_geometry(settings),
    }


def _canonical_settings_value(value: Any, *, field: str) -> Any:
    """Convert every resolved setting to a deterministic JSON value.

    This intentionally rejects unknown runtime objects instead of silently
    omitting them from the session identity.
    """

    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.generic):
        return _canonical_settings_value(value.item(), field=field)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise RuntimeError(
                f"Resolved setting {field!r} is non-finite and cannot be sealed"
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            text_key = str(key)
            if text_key in result:
                raise RuntimeError(
                    f"Resolved setting {field!r} has colliding JSON keys"
                )
            result[text_key] = _canonical_settings_value(
                value[key], field=f"{field}.{text_key}"
            )
        return result
    if isinstance(value, (tuple, list)):
        return [
            _canonical_settings_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        items = [
            _canonical_settings_value(item, field=f"{field}[]") for item in value
        ]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    raise TypeError(
        f"Resolved setting {field!r} has unsupported type {type(value).__name__}; "
        "refusing to leave it outside the session identity"
    )


def _resolved_config_identity(settings: Any) -> dict[str, Any]:
    values = {
        key: _canonical_settings_value(value, field=key)
        for key, value in sorted(vars(settings).items())
        if not key.startswith("_")
    }
    return {
        "schema_version": 1,
        "field_count": len(values),
        "sha256": sha256_canonical_json(values),
        "values": values,
    }


def _device(settings: Any) -> torch.device:
    requested = str(settings.device)
    return torch.device(
        requested if requested != "cuda" or torch.cuda.is_available() else "cpu"
    )


def _require_checkpoint_architecture(checkpoint: Path, expected: str) -> None:
    """Fail closed when a formal hardware checkpoint has no identity evidence."""

    try:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:  # Older supported PyTorch builds do not expose mmap.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Optical-router checkpoint is not a mapping: {checkpoint}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(
            "Formal six-stage hardware checkpoint has no metadata mapping; "
            "refusing an architecture-unverifiable checkpoint"
        )
    actual = metadata.get("optical_architecture")
    if actual is None or str(actual).strip() == "":
        raise RuntimeError(
            "Formal six-stage hardware checkpoint has no optical_architecture "
            "metadata; refusing an architecture-unverifiable checkpoint"
        )
    if str(actual) != str(expected):
        raise RuntimeError(
            "Formal six-stage hardware checkpoint architecture mismatch: "
            f"saved={actual!r}, expected={expected!r}"
        )


def _load_model(settings: Any, checkpoint: Path):
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Optical-router checkpoint is missing: {checkpoint}")
    _require_checkpoint_architecture(checkpoint, architecture_label(settings))
    loaded = load_backbone(settings, _device(settings))
    settings.resolve_architecture(loaded.model)
    replacement, readout = build_hybrid_student(loaded, settings)
    load_checkpoint(checkpoint, replacement, readout)
    replacement.set_phase_dropout_active(False)
    legacy_bridge._set_replacement_eval(replacement)
    readout.eval()
    return loaded, replacement, readout


def _phase_for_stage(replacement: Any, stage: str, settings: Any) -> np.ndarray:
    modality = "vision" if stage.startswith("vision") else "language"
    branch = (
        replacement.vision_surrogate.core.optical_branch
        if modality == "vision"
        else replacement.language_surrogate.core.optical_branch
    )
    if stage.endswith("router"):
        router = branch.core.router
        if not hasattr(router, "active_phase"):
            raise RuntimeError(f"{stage} has no optical Router active phase")
        value = router.active_phase()
    else:
        values = phase_tensors(branch.core)
        value = (
            values["physical_expert_mosaic_rad"]
            if stage.endswith("expert")
            else values["physical_global_phase_rad"]
        )
    if tuple(value.shape) != (settings.active_size, settings.active_size):
        raise RuntimeError(
            f"Logical {stage} phase must be {settings.active_size}x"
            f"{settings.active_size}, got {tuple(value.shape)}"
        )
    if bool(settings.hardware_phase_flip_vertical):
        value = torch.flip(value, (-2,))
    if bool(settings.hardware_phase_flip_horizontal):
        value = torch.flip(value, (-1,))
    return encode_active_phase(value.detach().cpu().numpy())


def _reconstruct_phase(settings: Any, compact: Path, output: Path) -> dict[str, Any]:
    return reconstruct_directory(
        compact,
        output,
        slm_size_wh=(
            settings.hardware_phase_slm_width,
            settings.hardware_phase_slm_height,
        ),
        scale_factor=None,
        center_xy=(
            settings.hardware_phase_slm_center_x,
            settings.hardware_phase_slm_center_y,
        ),
        logical_pixel_pitch_um=settings.language_optical_pixel_pitch_um,
        slm_pixel_pitch_um=settings.hardware_phase_slm_pixel_pitch_um,
    )


def _reconstruct_amplitude(
    settings: Any, compact: Path, output: Path
) -> dict[str, Any]:
    return reconstruct_directory(
        compact,
        output,
        slm_size_wh=(
            settings.hardware_amplitude_slm_width,
            settings.hardware_amplitude_slm_height,
        ),
        scale_factor=None,
        center_xy=(
            settings.hardware_amplitude_slm_center_x,
            settings.hardware_amplitude_slm_center_y,
        ),
        logical_pixel_pitch_um=settings.language_optical_pixel_pitch_um,
        slm_pixel_pitch_um=settings.hardware_amplitude_slm_pixel_pitch_um,
    )


def _sample_rows(bundle: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, sample in enumerate(legacy_bridge._samples(bundle)):
        image_path = Path(sample.image_path).expanduser().resolve()
        rows.append(
            {
                "order": order,
                "key": legacy_bridge._key(sample),
                "sample_id": sample.sample_id,
                "split": sample.split,
                "sku_index": sample.sku_index,
                "sku_name": sample.sku_name,
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
            }
        )
    if not rows:
        raise RuntimeError("Caltech101 six-stage session has no samples")
    return rows


def _verify_dataset_sample_rows(
    sealed_rows: list[Mapping[str, Any]], current_rows: list[Mapping[str, Any]]
) -> None:
    """Require the current dataset to match the initialized manifest exactly."""

    if len(sealed_rows) != len(current_rows):
        raise RuntimeError(
            "Current Caltech101 sample count differs from the sealed session "
            f"manifest: sealed={len(sealed_rows)}, current={len(current_rows)}"
        )
    required = set(SAMPLE_MANIFEST_FIELDS)
    for order, (sealed, current) in enumerate(zip(sealed_rows, current_rows)):
        if set(sealed) != required:
            raise RuntimeError(
                "Sealed dataset manifest columns changed: "
                f"expected={list(SAMPLE_MANIFEST_FIELDS)}, actual={list(sealed)}"
            )
        if set(current) != required:
            raise RuntimeError(
                "Current dataset sample contract is incomplete: "
                f"expected={list(SAMPLE_MANIFEST_FIELDS)}, actual={list(current)}"
            )
        for field in SAMPLE_MANIFEST_FIELDS:
            expected = "" if sealed[field] is None else str(sealed[field])
            actual = "" if current[field] is None else str(current[field])
            if expected != actual:
                raise RuntimeError(
                    "Current Caltech101 sample differs from the sealed session "
                    f"manifest at row={order}, field={field!r}: "
                    f"sealed={expected!r}, current={actual!r}"
                )


def _write_all_phase_bundle(
    settings: Any,
    replacement: Any,
    checkpoint: Path,
    destination: Path,
) -> dict[str, Any]:
    destination = require_empty_directory(destination, label="six-phase bundle")
    compact = destination / "compact_phase"
    compact.mkdir()
    rows: list[dict[str, Any]] = []
    for stage in SIX_STAGES:
        path = compact / f"{stage}.png"
        save_active_png(_phase_for_stage(replacement, stage, settings), path)
        rows.append(
            {
                "stage": stage,
                "compact_phase": path.name,
                "compact_phase_sha256": sha256_file(path),
            }
        )
    reconstruction = _reconstruct_phase(
        settings, compact, destination / "phase_to_play"
    )
    for row in rows:
        bmp = destination / "phase_to_play" / f"{row['stage']}.bmp"
        row["native_phase_bmp"] = bmp.name
        row["native_phase_bmp_sha256"] = sha256_file(bmp)
    manifest = destination / "phase_manifest.csv"
    write_csv(manifest, rows)
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_architecture": architecture_label(settings),
        "stages": list(SIX_STAGES),
        "logical_phase_size_wh": [settings.active_size, settings.active_size],
        "router_trainable_phase_size_wh": [
            settings.expert_size,
            settings.expert_size,
        ],
        "phase_flip_vertical_before_raster": bool(
            settings.hardware_phase_flip_vertical
        ),
        "phase_flip_horizontal_before_raster": bool(
            settings.hardware_phase_flip_horizontal
        ),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "reconstruction": reconstruction,
    }
    write_json(destination / "phase_bundle_report.json", report)
    return report


def initialize_session(
    settings: Any, config: Path, checkpoint: Path, session_dir: Path
) -> dict[str, Any]:
    if settings.router_backend != "optical":
        raise ValueError("Six-stage hardware execution requires an optical Router config")
    geometry_report = _validate_hardware_geometry(settings)
    resolved_config_identity = _resolved_config_identity(settings)
    session_dir = require_empty_directory(session_dir, label="six-stage session")
    checkpoint = checkpoint.expanduser().resolve()
    bundle = prepare_caltech101_subset(settings, persist=True)
    manifest = session_dir / MANIFEST_FILENAME
    write_csv(manifest, _sample_rows(bundle))
    loaded, replacement, readout = _load_model(settings, checkpoint)
    del loaded, readout
    try:
        phase_report = _write_all_phase_bundle(
            settings, replacement, checkpoint, session_dir / "00_phase_bundle"
        )
    finally:
        replacement.close()
    state = initialize_state(
        config=config,
        checkpoint=checkpoint,
        manifest=manifest,
        architecture=architecture_label(settings),
    )
    state["resolved_hardware_contract"] = _resolved_hardware_contract(settings)
    state["resolved_config_identity"] = resolved_config_identity
    record_event(
        state,
        stage=None,
        action="initialize",
        payload={
            "phase_manifest": phase_report["manifest"],
            "phase_manifest_sha256": phase_report["manifest_sha256"],
            "hardware_geometry": geometry_report,
        },
    )
    write_json(session_dir / STATE_FILENAME, state)
    return state


def _load_session(
    *, config: Path, checkpoint: Path, session_dir: Path
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    state_path = session_dir.expanduser().resolve() / STATE_FILENAME
    manifest_path = session_dir.expanduser().resolve() / MANIFEST_FILENAME
    if not state_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Six-stage session is not initialized; run --phase initialize first"
        )
    state = read_json(state_path)
    manifest = read_csv(manifest_path)
    validate_state_identity(
        state, config=config, checkpoint=checkpoint, manifest=manifest_path
    )
    settings = load_settings(config)
    if state.get("checkpoint_architecture") != architecture_label(settings):
        raise RuntimeError("Session/config checkpoint architecture changed")
    actual_identity = _resolved_config_identity(settings)
    sealed_identity = state.get("resolved_config_identity")
    if not isinstance(sealed_identity, dict):
        raise RuntimeError(
            "Session predates the complete resolved-config identity contract; "
            "initialize a new six-stage session"
        )
    if sealed_identity != actual_identity:
        sealed_values = sealed_identity.get("values", {})
        actual_values = actual_identity["values"]
        changed = sorted(
            key
            for key in set(sealed_values).union(actual_values)
            if sealed_values.get(key) != actual_values.get(key)
        )
        raise RuntimeError(
            "Complete resolved config identity changed: "
            f"sealed_sha256={sealed_identity.get('sha256')}, "
            f"actual_sha256={actual_identity['sha256']}, "
            f"changed_fields={changed[:16]}"
        )
    if state.get("resolved_hardware_contract") != _resolved_hardware_contract(settings):
        raise RuntimeError(
            "Resolved Router/propagation/hardware contract changed, even though the "
            "leaf config file identity still matches"
        )
    return state, manifest


def _routing_csv(session_dir: Path, router_stage: str) -> Path:
    return stage_directory(session_dir, router_stage) / "routing" / "routing.csv"


def _verify_capture_artifact(session_dir: Path, stage: str) -> None:
    state = read_json(session_dir / STATE_FILENAME)
    event = state["stages"][stage].get("capture")
    if event is None:
        raise RuntimeError(f"{stage} has no sealed capture contract")
    _verify_export_artifact(session_dir, stage)
    acquisition_manifest = Path(
        str(event.get("acquisition_manifest", ""))
    ).expanduser().resolve()
    if (
        not acquisition_manifest.is_file()
        or sha256_file(acquisition_manifest)
        != event.get("acquisition_manifest_sha256")
    ):
        raise RuntimeError(f"{stage} acquisition manifest is missing or changed")
    manifest = Path(event["capture_manifest"]).expanduser().resolve()
    if not manifest.is_file() or sha256_file(manifest) != event[
        "capture_manifest_sha256"
    ]:
        raise RuntimeError(f"{stage} CCD capture manifest is missing or changed")
    rows = read_csv(manifest)
    root = stage_directory(session_dir, stage) / "ccd_captured"
    expected_names = {row["filename"] for row in rows}
    actual_names = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".bmp", ".tif", ".tiff"}
    }
    if actual_names != expected_names:
        raise RuntimeError(f"{stage} CCD files changed after capture was sealed")
    for row in rows:
        path = root / row["filename"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"{stage} CCD file changed after sealing: {path.name}")


def _safe_manifest_filename(row: Mapping[str, Any], field: str, label: str) -> str:
    value = str(row.get(field, ""))
    if not value or Path(value).name != value:
        raise RuntimeError(f"{label} has an unsafe or missing {field}: {value!r}")
    return value


def _verify_amplitude_reconstruction_chain(
    destination: Path,
    event: Mapping[str, Any],
    transport: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Verify compact PNG -> reconstructed BMP for every ordered sample."""

    compact_manifest = destination / "compact_amplitude_manifest.csv"
    compact_rows = read_csv(compact_manifest)
    compact_dir = destination / "compact_amplitude"
    compact_by_key: dict[str, dict[str, str]] = {}
    for order, row in enumerate(compact_rows):
        key = str(row.get("key", ""))
        if not key or key in compact_by_key:
            raise RuntimeError(f"Duplicate or empty compact amplitude key: {key!r}")
        if str(row.get("order", "")) != str(order):
            raise RuntimeError(f"Compact amplitude order changed for {key}")
        filename = _safe_manifest_filename(
            row, "filename", "compact amplitude manifest"
        )
        expected_sha = str(row.get("sha256", ""))
        source = compact_dir / filename
        if not source.is_file() or sha256_file(source) != expected_sha:
            raise RuntimeError(f"Compact amplitude PNG is missing or changed: {key}")
        compact_by_key[key] = {"filename": filename, "sha256": expected_sha}
    actual_compact_names = {
        path.name
        for path in compact_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".bmp", ".tif", ".tiff"}
    }
    expected_compact_names = {
        row["filename"] for row in compact_by_key.values()
    }
    if actual_compact_names != expected_compact_names:
        raise RuntimeError("Compact amplitude file set differs from its manifest")

    reconstruction_manifest = (
        destination / "amplitude_to_play" / "reconstruction_manifest.csv"
    )
    expected_manifest_sha = str(
        event.get("amplitude_reconstruction_manifest_sha256", "")
    )
    if (
        not reconstruction_manifest.is_file()
        or sha256_file(reconstruction_manifest) != expected_manifest_sha
    ):
        raise RuntimeError(
            "Amplitude reconstruction manifest is missing or changed"
        )
    if (
        str(transport.get("amplitude_reconstruction_manifest_sha256", ""))
        != expected_manifest_sha
    ):
        raise RuntimeError(
            "Transport/export amplitude reconstruction SHA contract differs"
        )

    reconstruction_rows = read_csv(reconstruction_manifest)
    reconstructed: dict[str, dict[str, str]] = {}
    for order, row in enumerate(reconstruction_rows):
        key = str(row.get("basename", ""))
        if not key or key in reconstructed:
            raise RuntimeError(f"Duplicate or empty reconstructed amplitude key: {key!r}")
        if str(row.get("order", "")) != str(order):
            raise RuntimeError(f"Amplitude reconstruction order changed for {key}")
        source_name = _safe_manifest_filename(
            row, "source_png", "amplitude reconstruction manifest"
        )
        output_name = _safe_manifest_filename(
            row, "output_bmp", "amplitude reconstruction manifest"
        )
        compact = compact_by_key.get(key)
        if compact is None:
            raise RuntimeError(
                f"Amplitude reconstruction contains an unknown sample: {key}"
            )
        if source_name != compact["filename"]:
            raise RuntimeError(
                f"Reconstruction source filename differs from compact manifest for {key}"
            )
        if str(row.get("source_sha256", "")) != compact["sha256"]:
            raise RuntimeError(
                f"Reconstruction source SHA differs from compact manifest for {key}"
            )
        if Path(output_name).stem != key:
            raise RuntimeError(f"Reconstructed amplitude output/key mismatch for {key}")
        output_sha = str(row.get("output_sha256", ""))
        output = destination / "amplitude_to_play" / output_name
        if not output.is_file() or sha256_file(output) != output_sha:
            raise RuntimeError(f"Reconstructed amplitude BMP is missing or changed: {key}")
        reconstructed[key] = {"filename": output_name, "sha256": output_sha}
    if list(reconstructed) != list(compact_by_key):
        raise RuntimeError(
            "Amplitude reconstruction sample order differs from compact manifest"
        )
    actual_bmp_names = {
        path.name
        for path in (destination / "amplitude_to_play").iterdir()
        if path.is_file() and path.suffix.lower() == ".bmp"
    }
    expected_bmp_names = {row["filename"] for row in reconstructed.values()}
    if actual_bmp_names != expected_bmp_names:
        raise RuntimeError("Reconstructed amplitude BMP set differs from its manifest")
    return reconstructed


def _verify_export_artifact(
    session_dir: Path, stage: str
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Verify every stage-export contract before accepting a physical capture."""

    state = read_json(session_dir / STATE_FILENAME)
    event = state["stages"][stage].get("export")
    if event is None:
        raise RuntimeError(f"{stage} has no sealed export contract")
    destination = stage_directory(session_dir, stage)
    transport_path = destination / "transport_spec.json"
    if not transport_path.is_file() or sha256_file(transport_path) != event[
        "transport_spec_sha256"
    ]:
        raise RuntimeError(f"{stage} transport_spec.json is missing or changed")
    transport = read_json(transport_path)
    amplitude_manifest = destination / "compact_amplitude_manifest.csv"
    if not amplitude_manifest.is_file() or sha256_file(amplitude_manifest) != event[
        "amplitude_manifest_sha256"
    ]:
        raise RuntimeError(f"{stage} compact amplitude manifest is missing or changed")
    if (
        str(transport.get("amplitude_manifest_sha256", ""))
        != event["amplitude_manifest_sha256"]
    ):
        raise RuntimeError(f"{stage} transport/export amplitude SHA contract differs")
    phase_bmp = destination / "phase_to_play" / f"{stage}.bmp"
    if not phase_bmp.is_file() or sha256_file(phase_bmp) != event["phase_bmp_sha256"]:
        raise RuntimeError(f"{stage} phase BMP is missing or changed")
    if str(transport.get("phase_bmp_sha256")) != event["phase_bmp_sha256"]:
        raise RuntimeError(f"{stage} transport/export phase SHA contract differs")
    reconstructed = _verify_amplitude_reconstruction_chain(
        destination, event, transport
    )
    return transport, reconstructed


def _verify_routing_artifact(session_dir: Path, router_stage: str) -> None:
    state = read_json(session_dir / STATE_FILENAME)
    event = state["stages"][router_stage].get("routing")
    if event is None:
        raise RuntimeError(f"{router_stage} has no sealed routing contract")
    path = _routing_csv(session_dir, router_stage)
    if not path.is_file() or sha256_file(path) != event["routing_csv_sha256"]:
        raise RuntimeError(f"{router_stage} routing.csv is missing or changed")
    report = Path(str(event.get("score_report", ""))).expanduser().resolve()
    if (
        not report.is_file()
        or sha256_file(report) != event.get("score_report_sha256")
    ):
        raise RuntimeError(
            f"{router_stage} routing score report is missing or changed"
        )
    _verify_capture_artifact(session_dir, router_stage)


def _verify_upstream_artifacts(
    session_dir: Path,
    target_stage: str,
    *,
    measured_feature_stages: tuple[str, ...] | None = None,
) -> None:
    target_index = SIX_STAGES.index(target_stage)
    for router_stage in ROUTER_STAGES:
        if SIX_STAGES.index(router_stage) < target_index:
            _verify_routing_artifact(session_dir, router_stage)
    stages = (
        tuple(
            stage
            for stage in FEATURE_STAGES
            if SIX_STAGES.index(stage) < target_index
        )
        if measured_feature_stages is None
        else measured_feature_stages
    )
    for stage in stages:
        _verify_capture_artifact(session_dir, stage)


def _routing_payload(
    session_dir: Path,
    router_stage: str,
    keys: list[str],
    *,
    device: torch.device,
    top_k: int,
) -> dict[str, torch.Tensor]:
    path = _routing_csv(session_dir, router_stage)
    if not path.is_file():
        raise FileNotFoundError(
            f"Measured {router_stage} routing is missing: {path}; capture and score it first"
        )
    rows = read_csv(path)
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        key = Path(row["filename"]).stem
        if key in by_key:
            raise RuntimeError(f"Duplicate routing row for key {key!r} in {path}")
        by_key[key] = row
    missing = sorted(set(keys).difference(by_key))
    unexpected = sorted(set(by_key).difference(keys))
    if missing or unexpected:
        raise RuntimeError(
            f"Routing CSV/sample identity mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    probabilities: list[list[float]] = []
    weights: list[list[float]] = []
    selected: list[list[bool]] = []
    indices: list[list[int]] = []
    energies: list[list[float]] = []
    energy_fractions: list[list[float]] = []
    raw_capture: list[float] = []
    for key in keys:
        row = by_key[key]
        p = [float(row[f"probability_{index}"]) for index in range(4)]
        w = [float(row[f"weight_{index}"]) for index in range(4)]
        s = [str(row[f"selected_{index}"]).lower() == "true" for index in range(4)]
        order = [int(value) for value in row["selected_experts"].split(",")]
        e = [float(row[f"energy_{index}"]) for index in range(4)]
        ef = [float(row[f"energy_fraction_{index}"]) for index in range(4)]
        values = np.asarray(
            [*p, *w, *e, *ef, float(row["raw_capture_fraction"])]
        )
        if not np.isfinite(values).all() or np.any(np.asarray([*p, *w, *e, *ef]) < 0):
            raise RuntimeError(f"Routing row for {key} is nonfinite or negative")
        if abs(sum(p) - 1.0) > 1.0e-5:
            raise RuntimeError(f"Routing probabilities for {key} do not sum to one")
        if abs(sum(ef) - 1.0) > 1.0e-5:
            raise RuntimeError(f"Routing detector fractions for {key} do not sum to one")
        if not 0.0 <= float(row["raw_capture_fraction"]) <= 1.0:
            raise RuntimeError(
                f"Routing raw capture fraction for {key} is outside [0,1]"
            )
        if len(order) != top_k or len(set(order)) != top_k:
            raise RuntimeError(f"Routing row for {key} does not select exactly top-k={top_k}")
        expected_order = torch.topk(
            torch.tensor(p, dtype=torch.float64), top_k
        ).indices.tolist()
        if order != expected_order:
            raise RuntimeError(
                f"Routing selected experts for {key} are not deterministic top-k"
            )
        if set(order) != {index for index, value in enumerate(s) if value}:
            raise RuntimeError(f"Routing selected mask/index mismatch for {key}")
        if any((not flag) and abs(w[index]) > 1.0e-7 for index, flag in enumerate(s)):
            raise RuntimeError(f"Unselected expert has nonzero amplitude weight for {key}")
        if abs(sum(value * value for value in w) - 1.0) > 2.0e-4:
            raise RuntimeError(f"power_l2 route for {key} does not satisfy sum(weight^2)=1")
        selected_power = sum(p[index] * p[index] for index in expected_order) ** 0.5
        expected_weights = [
            (p[index] / selected_power if index in expected_order else 0.0)
            for index in range(4)
        ]
        if not np.allclose(w, expected_weights, rtol=1.0e-4, atol=1.0e-6):
            raise RuntimeError(
                f"Routing weights for {key} do not match probabilities/power_l2"
            )
        probabilities.append(p)
        weights.append(w)
        selected.append(s)
        indices.append(order)
        energies.append(e)
        energy_fractions.append(ef)
        raw_capture.append(float(row["raw_capture_fraction"]))
    probability_tensor = torch.tensor(probabilities, dtype=torch.float32, device=device)
    return {
        "probabilities": probability_tensor,
        "weights": torch.tensor(weights, dtype=torch.float32, device=device),
        "selected_mask": torch.tensor(selected, dtype=torch.bool, device=device),
        "selected_indices": torch.tensor(indices, dtype=torch.long, device=device),
        "detector_energy": torch.tensor(energies, dtype=torch.float32, device=device),
        "detector_energy_fraction": torch.tensor(
            energy_fractions, dtype=torch.float32, device=device
        ),
        "raw_capture_fraction": torch.tensor(
            raw_capture, dtype=torch.float32, device=device
        ),
    }


def _router_for_stage(replacement: Any, router_stage: str) -> Any:
    branch = (
        replacement.vision_surrogate.core.optical_branch
        if router_stage.startswith("vision")
        else replacement.language_surrogate.core.optical_branch
    )
    return branch.core.router


def _set_measured_routing(router: Any, payload: Mapping[str, torch.Tensor] | None) -> None:
    setter = getattr(router, "set_measured_routing", None)
    if setter is None:
        raise RuntimeError(
            "Optical Router does not expose set_measured_routing(); the checkpoint "
            "code and six-stage hardware bridge are from different revisions"
        )
    setter(payload)


def _clear_batch_measurements(replacement: Any) -> None:
    replacement.vision_surrogate.core.optical_branch.clear_measured_ccd()
    replacement.language_surrogate.core.optical_branch.clear_measured_ccd()
    for stage in ROUTER_STAGES:
        router = _router_for_stage(replacement, stage)
        setter = getattr(router, "set_measured_routing", None)
        if setter is not None:
            setter(None)


def _load_canonical_feature_ccd(
    session_dir: Path, stage: str, key: str, *, active_size: int
) -> torch.Tensor:
    root = stage_directory(session_dir, stage) / "ccd_captured"
    matches = [
        root / f"{key}{suffix}"
        for suffix in (".png", ".bmp", ".tif", ".tiff")
        if (root / f"{key}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one canonical CCD for {key} below {root}, "
            f"found {len(matches)}"
        )
    value = legacy_bridge._load_measured_ccd_uint8(matches[0])
    if tuple(value.shape) != (active_size, active_size):
        raise RuntimeError(
            f"Canonical CCD {matches[0]} must be {active_size}x{active_size}, "
            f"got {tuple(value.shape)}"
        )
    # detector_homography already produced canonical_model_xy.  Deliberately
    # do not consult the legacy downstream flip flags here.
    return value.float().clamp_min(0.0)


def _install_batch_measurements(
    replacement: Any,
    settings: Any,
    session_dir: Path,
    keys: list[str],
    *,
    target_stage: str,
    measured_feature_stages: tuple[str, ...] | None = None,
) -> None:
    _clear_batch_measurements(replacement)
    target_index = SIX_STAGES.index(target_stage)
    device = next(replacement.vision_surrogate.parameters()).device
    for router_stage in ROUTER_STAGES:
        if SIX_STAGES.index(router_stage) < target_index:
            payload = _routing_payload(
                session_dir,
                router_stage,
                keys,
                device=device,
                top_k=int(settings.top_k),
            )
            _set_measured_routing(_router_for_stage(replacement, router_stage), payload)
    if measured_feature_stages is None:
        measured_feature_stages = tuple(
            stage
            for stage in FEATURE_STAGES
            if SIX_STAGES.index(stage) < target_index
        )
    selected = set(measured_feature_stages)
    tensors: dict[str, torch.Tensor | None] = {}
    for stage in FEATURE_STAGES:
        tensors[stage] = (
            torch.stack(
                [
                    _load_canonical_feature_ccd(
                        session_dir,
                        stage,
                        key,
                        active_size=int(settings.active_size),
                    )
                    for key in keys
                ]
            ).to(device)
            if stage in selected
            else None
        )
    replacement.vision_surrogate.core.optical_branch.set_measured_ccd(
        expert=tensors["vision_expert"], global_=tensors["vision_global"]
    )
    replacement.language_surrogate.core.optical_branch.set_measured_ccd(
        expert=tensors["language_expert"], global_=tensors["language_global"]
    )


def _capture_stage_amplitude(
    loaded: Any,
    replacement: Any,
    readout: Any,
    settings: Any,
    samples: list[Any],
    stage: str,
) -> torch.Tensor:
    branch = (
        replacement.vision_surrogate.core.optical_branch
        if stage.startswith("vision")
        else replacement.language_surrogate.core.optical_branch
    )
    branch.core.capture_intermediate_fields = True
    branch.core.capture_sample_count = len(samples)
    try:
        legacy_bridge._forward_samples(
            loaded, replacement, readout, settings, samples
        )
        if stage.endswith("router"):
            value = branch.core.last_input_fields
        elif stage.endswith("expert"):
            value = branch.last_expert_input_amplitude
        else:
            value = branch.last_global_input_amplitude
        if value is None or len(value) != len(samples):
            raise RuntimeError(f"Model did not capture a complete {stage} amplitude batch")
        expected = settings.expert_size if stage.endswith("router") else settings.active_size
        if tuple(value.shape[-2:]) != (expected, expected):
            raise RuntimeError(
                f"{stage} amplitude must be {expected}x{expected}, got {tuple(value.shape)}"
            )
        value = value.detach().cpu().float()
        if not torch.isfinite(value).all() or torch.any(value < 0):
            raise RuntimeError(f"{stage} amplitude is nonfinite or negative")
        return value
    finally:
        branch.core.capture_intermediate_fields = False


def _require_export_dependency(state: Mapping[str, Any], stage: str) -> None:
    required: tuple[str, str] | None = {
        "vision_router": None,
        "vision_expert": ("vision_router", "routing"),
        "vision_global": ("vision_expert", "finetune"),
        "language_router": ("vision_global", "finetune"),
        "language_expert": ("language_router", "routing"),
        "language_global": ("language_expert", "finetune"),
    }[stage]
    if required is not None and required[1] not in state["stages"][required[0]]:
        raise RuntimeError(
            f"Cannot export {stage}: {required[0]} has no completed {required[1]} contract"
        )


@torch.no_grad()
def export_stage(
    settings: Any,
    config: Path,
    checkpoint: Path,
    session_dir: Path,
    stage: str,
    *,
    inference_batch_size: int = 10,
) -> dict[str, Any]:
    if stage not in SIX_STAGES:
        raise ValueError(f"Unknown stage {stage!r}")
    _validate_hardware_geometry(settings)
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    state, manifest = _load_session(
        config=config, checkpoint=checkpoint, session_dir=session_dir
    )
    _require_export_dependency(state, stage)
    _verify_upstream_artifacts(session_dir, stage)
    bundle = prepare_caltech101_subset(settings, persist=True)
    current_sample_rows = _sample_rows(bundle)
    _verify_dataset_sample_rows(manifest, current_sample_rows)
    samples = legacy_bridge._samples(bundle)
    keys = [row["key"] for row in current_sample_rows]
    destination = require_empty_directory(
        stage_directory(session_dir, stage), label=f"{stage} export directory"
    )
    compact_amplitude = destination / "compact_amplitude"
    compact_phase = destination / "compact_phase"
    compact_amplitude.mkdir()
    compact_phase.mkdir()
    (destination / "ccd_captured").mkdir()

    loaded, replacement, readout = _load_model(settings, checkpoint)
    amplitude_rows: list[dict[str, Any]] = []
    try:
        phase_path = compact_phase / f"{stage}.png"
        save_active_png(_phase_for_stage(replacement, stage, settings), phase_path)
        phase_reconstruction = _reconstruct_phase(
            settings, compact_phase, destination / "phase_to_play"
        )
        for start in range(0, len(samples), inference_batch_size):
            batch = samples[start : start + inference_batch_size]
            batch_keys = keys[start : start + len(batch)]
            _install_batch_measurements(
                replacement,
                settings,
                session_dir,
                batch_keys,
                target_stage=stage,
            )
            values = _capture_stage_amplitude(
                loaded, replacement, readout, settings, batch, stage
            )
            for key, value in zip(batch_keys, values):
                encoded, encoding = encode_active_amplitude_with_metadata(
                    value.numpy()
                )
                output = compact_amplitude / f"{key}.png"
                save_active_png(encoded, output)
                amplitude_rows.append(
                    {
                        "order": len(amplitude_rows),
                        "key": key,
                        "filename": output.name,
                        "sha256": sha256_file(output),
                        "logical_height": encoded.shape[0],
                        "logical_width": encoded.shape[1],
                        "source_tensor": (
                            "router_central_input_fields"
                            if stage.endswith("router")
                            else f"measured-upstream_{stage}_amplitude_slm_canvas"
                        ),
                        **encoding,
                    }
                )
            print(
                f"[export_{stage}] {min(start + len(batch), len(samples))}/{len(samples)}",
                flush=True,
            )
    finally:
        _clear_batch_measurements(replacement)
        replacement.close()
    amplitude_manifest = destination / "compact_amplitude_manifest.csv"
    write_csv(amplitude_manifest, amplitude_rows)
    amplitude_reconstruction = _reconstruct_amplitude(
        settings, compact_amplitude, destination / "amplitude_to_play"
    )
    amplitude_reconstruction_manifest = (
        destination / "amplitude_to_play" / "reconstruction_manifest.csv"
    )
    if not amplitude_reconstruction_manifest.is_file():
        raise RuntimeError("Amplitude reconstruction produced no manifest")
    phase_bmp = destination / "phase_to_play" / f"{stage}.bmp"
    report = {
        "schema_version": 1,
        "stage": stage,
        "stage_order": SIX_STAGES.index(stage) + 1,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(config),
        "dataset_manifest_sha256": state["dataset_manifest_sha256"],
        "samples": len(amplitude_rows),
        "logical_amplitude_size_wh": (
            [settings.expert_size, settings.expert_size]
            if stage.endswith("router")
            else [settings.active_size, settings.active_size]
        ),
        "expected_ccd_contract": (
            "canonical model orientation, single-channel uint8, exactly 478x478; "
            "no additional flip inside router scoring"
            if stage.endswith("router")
            else (
                "canonical model orientation after detector homography, "
                "single-channel uint8, exactly 478x478; no downstream flip"
            )
        ),
        "amplitude_manifest": str(amplitude_manifest),
        "amplitude_manifest_sha256": sha256_file(amplitude_manifest),
        "amplitude_reconstruction_manifest": str(
            amplitude_reconstruction_manifest
        ),
        "amplitude_reconstruction_manifest_sha256": sha256_file(
            amplitude_reconstruction_manifest
        ),
        "phase_compact_sha256": sha256_file(
            destination / "compact_phase" / f"{stage}.png"
        ),
        "phase_bmp": str(phase_bmp),
        "phase_bmp_sha256": sha256_file(phase_bmp),
        "amplitude_reconstruction": amplitude_reconstruction,
        "phase_reconstruction": phase_reconstruction,
    }
    report_path = destination / "transport_spec.json"
    write_json(report_path, report)
    record_event(
        state,
        stage=stage,
        action="export",
        payload={
            "checkpoint_sha256": report["checkpoint_sha256"],
            "transport_spec": str(report_path),
            "transport_spec_sha256": sha256_file(report_path),
            "amplitude_manifest_sha256": report["amplitude_manifest_sha256"],
            "amplitude_reconstruction_manifest_sha256": report[
                "amplitude_reconstruction_manifest_sha256"
            ],
            "phase_bmp_sha256": report["phase_bmp_sha256"],
        },
    )
    write_json(session_dir / STATE_FILENAME, state)
    return report


def validate_capture(
    *, config: Path, checkpoint: Path, session_dir: Path, stage: str
) -> dict[str, Any]:
    state, _ = _load_session(
        config=config, checkpoint=checkpoint, session_dir=session_dir
    )
    if "export" not in state["stages"][stage]:
        raise RuntimeError(f"{stage} has not been exported")
    if "capture" in state["stages"][stage]:
        raise RuntimeError(
            f"{stage} capture is already sealed in the session state; do not replace CCD files"
        )
    destination = stage_directory(session_dir, stage)
    transport, reconstructed_amplitudes = _verify_export_artifact(
        session_dir, stage
    )
    amplitude_rows = read_csv(destination / "compact_amplitude_manifest.csv")
    keys = [row["key"] for row in amplitude_rows]
    rows = expected_key_files(
        destination / "ccd_captured",
        keys,
        expected_shape_hw=(478, 478),
        require_uint8=True,
    )
    acquisition_manifest = destination / "acquisition_logs" / "capture_manifest.csv"
    if not acquisition_manifest.is_file():
        raise FileNotFoundError(
            f"Formal capture manifest is missing: {acquisition_manifest}. Upload "
            "acquisition_logs together with ccd_captured."
        )
    acquisition_rows = read_csv(acquisition_manifest)
    by_ccd_stem: dict[str, dict[str, str]] = {}
    for row in acquisition_rows:
        ccd_name = _safe_manifest_filename(
            row, "ccd_capture", "acquisition manifest"
        )
        key = Path(ccd_name).stem
        if key in by_ccd_stem:
            raise RuntimeError(f"Duplicate acquisition log row for {key}")
        by_ccd_stem[key] = row
    if set(by_ccd_stem) != set(keys):
        raise RuntimeError("Acquisition log sample set differs from stage manifest")
    phase_sha = str(transport["phase_bmp_sha256"])
    for key, capture_row in zip(keys, rows):
        logged = by_ccd_stem[key]
        if str(logged.get("orientation_canonicalized", "")).lower() != "true":
            raise RuntimeError(f"CCD {key} was not canonicalized by detector homography")
        if str(logged.get("downstream_loader_flip_required", "")).lower() != "false":
            raise RuntimeError(f"CCD {key} still requests a downstream flip")
        if str(logged.get("phase_manifest_verified", "")).lower() != "true":
            raise RuntimeError(f"Phase reconstruction manifest was not verified for {key}")
        if logged.get("phase_mask_sha256") != phase_sha:
            raise RuntimeError(f"Wrong phase mask was recorded for CCD {key}")
        if logged.get("output_sha256") != capture_row["sha256"]:
            raise RuntimeError(f"CCD SHA differs from acquisition log for {key}")
        if str(logged.get("ccd_capture", "")) != capture_row["filename"]:
            raise RuntimeError(f"CCD filename differs from acquisition log for {key}")
        amplitude_name = _safe_manifest_filename(
            logged, "amplitude_bmp", "acquisition manifest"
        )
        reconstructed = reconstructed_amplitudes.get(key)
        if reconstructed is None or amplitude_name != reconstructed["filename"]:
            raise RuntimeError(f"Amplitude/CCD key mismatch in acquisition log for {key}")
        if logged.get("amplitude_bmp_sha256") != reconstructed["sha256"]:
            raise RuntimeError(f"Amplitude BMP SHA differs from export manifest for {key}")
    capture_manifest = destination / "ccd_capture_manifest.csv"
    write_csv(capture_manifest, rows)
    report = {
        "schema_version": 1,
        "stage": stage,
        "images": len(rows),
        "orientation": "canonical_model_xy_no_further_flip",
        "manifest": str(capture_manifest),
        "manifest_sha256": sha256_file(capture_manifest),
        "acquisition_manifest": str(acquisition_manifest),
        "acquisition_manifest_sha256": sha256_file(acquisition_manifest),
    }
    report_path = destination / "ccd_capture_report.json"
    write_json(report_path, report)
    record_event(
        state,
        stage=stage,
        action="capture",
        payload={
            "capture_manifest": str(capture_manifest),
            "capture_manifest_sha256": report["manifest_sha256"],
            "acquisition_manifest": str(acquisition_manifest),
            "acquisition_manifest_sha256": report[
                "acquisition_manifest_sha256"
            ],
        },
    )
    write_json(session_dir / STATE_FILENAME, state)
    return report


def score_router_stage(
    settings: Any,
    *,
    config: Path,
    checkpoint: Path,
    session_dir: Path,
    stage: str,
) -> dict[str, Any]:
    if stage not in ROUTER_STAGES:
        raise ValueError("Only vision_router or language_router has Router CCD scores")
    state, _ = _load_session(
        config=config, checkpoint=checkpoint, session_dir=session_dir
    )
    if "capture" not in state["stages"][stage]:
        validate_capture(
            config=config, checkpoint=checkpoint, session_dir=session_dir, stage=stage
        )
        state = read_json(session_dir / STATE_FILENAME)
    _verify_capture_artifact(session_dir, stage)
    destination = stage_directory(session_dir, stage)
    routing_dir = destination / "routing"
    amplitude_rows = read_csv(destination / "compact_amplitude_manifest.csv")
    expected = [row["key"] for row in amplitude_rows]
    report = score_directory(
        config,
        destination / "ccd_captured",
        routing_dir,
        expected_stems=expected,
    )
    routing_rows = read_csv(routing_dir / "routing.csv")
    actual = [Path(row["filename"]).stem for row in routing_rows]
    if actual != expected:
        raise RuntimeError("Router score order/identity differs from amplitude manifest")
    # Parse once through the same strict loader used during model injection.
    _routing_payload(
        session_dir,
        stage,
        expected,
        device=torch.device("cpu"),
        top_k=int(settings.top_k),
    )
    routing_csv = routing_dir / "routing.csv"
    record_event(
        state,
        stage=stage,
        action="routing",
        payload={
            "routing_csv": str(routing_csv),
            "routing_csv_sha256": sha256_file(routing_csv),
            "score_report": str(routing_dir / "routing_report.json"),
            "score_report_sha256": sha256_file(
                routing_dir / "routing_report.json"
            ),
        },
    )
    write_json(session_dir / STATE_FILENAME, state)
    return report


@contextlib.contextmanager
def _patched_legacy_finetune(
    *, target_stage: str
) -> Iterator[None]:
    """Reuse the audited four-layer trainer with six-stage path/route adapters."""

    originals = {
        "STAGES": legacy_bridge.STAGES,
        "_stage_dir": legacy_bridge._stage_dir,
        "_load_model": legacy_bridge._load_model,
        "_install_measurements": legacy_bridge._install_measurements,
        "_clear_measurements": legacy_bridge._clear_measurements,
    }

    def install(
        replacement: Any,
        settings: Any,
        session_dir: Path,
        keys: list[str],
        *,
        measured_stages: tuple[str, ...],
    ) -> None:
        _install_batch_measurements(
            replacement,
            settings,
            session_dir,
            keys,
            target_stage=target_stage,
            measured_feature_stages=measured_stages,
        )

    try:
        legacy_bridge.STAGES = FEATURE_STAGES
        legacy_bridge._stage_dir = stage_directory
        legacy_bridge._load_model = _load_model
        legacy_bridge._install_measurements = install
        legacy_bridge._clear_measurements = _clear_batch_measurements
        yield
    finally:
        for name, value in originals.items():
            setattr(legacy_bridge, name, value)


def finetune_feature_stage(
    settings: Any,
    *,
    config: Path,
    checkpoint: Path,
    session_dir: Path,
    stage: str,
    epochs: int,
    selection_policy: str,
    development_per_class: int,
    inference_batch_size: int,
    early_stopping_patience: int,
) -> dict[str, Any]:
    if stage not in FEATURE_STAGES:
        raise ValueError("Router exposures are scored, not passed to feature-stage fine-tune")
    state, _ = _load_session(
        config=config, checkpoint=checkpoint, session_dir=session_dir
    )
    if "capture" not in state["stages"][stage]:
        raise RuntimeError(f"Validate the complete {stage} CCD capture before fine-tuning")
    if "finetune" in state["stages"][stage]:
        raise RuntimeError(
            f"{stage} already has a completed fine-tune contract; start a new session "
            "instead of overwriting checkpoint selection"
        )
    output = session_dir / "checkpoints" / f"after_{stage}.pt"
    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite an untracked/stale fine-tune checkpoint: {output}"
        )
    current_feature_index = FEATURE_STAGES.index(stage)
    measured_feature_stages = FEATURE_STAGES[: current_feature_index + 1]
    _verify_upstream_artifacts(
        session_dir,
        stage,
        measured_feature_stages=measured_feature_stages,
    )
    with _patched_legacy_finetune(target_stage=stage):
        legacy_bridge.finetune_stage(
            settings,
            checkpoint,
            session_dir,
            stage,
            epochs,
            upstream_source="measured",
            selection_policy=selection_policy,
            development_per_class=development_per_class,
            inference_batch_size=inference_batch_size,
            early_stopping_patience=early_stopping_patience,
        )
    if not output.is_file():
        raise RuntimeError(f"Legacy measured-CCD fine-tune produced no checkpoint: {output}")
    state["current_checkpoint"] = str(output.resolve())
    state["current_checkpoint_sha256"] = sha256_file(output)
    metrics = stage_directory(session_dir, stage) / "finetune_metrics.json"
    record_event(
        state,
        stage=stage,
        action="finetune",
        payload={
            "source_checkpoint_sha256": sha256_file(checkpoint),
            "output_checkpoint": str(output.resolve()),
            "output_checkpoint_sha256": sha256_file(output),
            "metrics": str(metrics),
            "metrics_sha256": sha256_file(metrics),
            "selection_policy": selection_policy,
            "sealed_test_used_for_selection": False,
        },
    )
    write_json(session_dir / STATE_FILENAME, state)
    return state["stages"][stage]["finetune"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict six-stage optical-Router hardware state machine"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=(
            "initialize",
            "export",
            "validate_capture",
            "score_router",
            "finetune",
            "status",
        ),
        required=True,
    )
    parser.add_argument("--stage", choices=SIX_STAGES)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--selection-policy",
        choices=("development", "train_loss"),
        default="development",
    )
    parser.add_argument("--development-per-class", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    args = parser.parse_args()

    config = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    session_dir = args.session_dir.expanduser().resolve()
    settings = load_settings(config)
    seed_everything(settings.random_seed)
    if args.phase == "status":
        state = read_json(session_dir / STATE_FILENAME)
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0
    if args.phase == "initialize":
        result = initialize_session(settings, config, checkpoint, session_dir)
    else:
        if args.stage is None:
            parser.error(f"--phase {args.phase} requires --stage")
        if args.phase == "export":
            result = export_stage(
                settings,
                config,
                checkpoint,
                session_dir,
                args.stage,
                inference_batch_size=args.inference_batch_size,
            )
        elif args.phase == "validate_capture":
            result = validate_capture(
                config=config,
                checkpoint=checkpoint,
                session_dir=session_dir,
                stage=args.stage,
            )
        elif args.phase == "score_router":
            result = score_router_stage(
                settings,
                config=config,
                checkpoint=checkpoint,
                session_dir=session_dir,
                stage=args.stage,
            )
        else:
            result = finetune_feature_stage(
                settings,
                config=config,
                checkpoint=checkpoint,
                session_dir=session_dir,
                stage=args.stage,
                epochs=args.epochs,
                selection_policy=args.selection_policy,
                development_per_class=args.development_per_class,
                inference_batch_size=args.inference_batch_size,
                early_stopping_patience=args.early_stopping_patience,
            )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "export_stage",
    "finetune_feature_stage",
    "initialize_session",
    "main",
    "score_router_stage",
    "validate_capture",
]
