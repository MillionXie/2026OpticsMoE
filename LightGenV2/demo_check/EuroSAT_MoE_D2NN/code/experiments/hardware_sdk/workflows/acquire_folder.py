"""Layer-agnostic allowlisted amplitude-SLM playback and CCD acquisition.

This file intentionally has no Torch/Qwen/model imports.  Run it as the
``experiments.hardware_sdk.workflows.acquire_folder`` module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from ..devices import (
        build_camera,
        build_slm,
        convert_detector_bit_depth,
        verify_camera_roi,
    )
    from .calibration_common import load_yaml_config
    from .detector_homography import load_geometry_contract, warp_detector_intensity
except ImportError:  # direct execution from workflows/
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from devices import (
        build_camera,
        build_slm,
        convert_detector_bit_depth,
        verify_camera_roi,
    )
    from workflows.calibration_common import load_yaml_config
    from workflows.detector_homography import (
        load_geometry_contract,
        warp_detector_intensity,
    )


CAPTURE_SUFFIXES = {".npy", ".png", ".tif", ".tiff"}


def _resolve(value: str | Path, base: Path) -> Path:
    import os

    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _clear_capture_files(directory: Path) -> None:
    if not directory.exists():
        return
    unexpected = [path for path in directory.iterdir() if path.is_dir()]
    if unexpected:
        raise RuntimeError(
            f"Refusing to clear {directory}: it contains subdirectories {unexpected}"
        )
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def _resolve_stage_layout(
    stage_dir: str | Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve the four folders owned by one exported optical stage.

    Stage paths are intentionally resolved against the process working
    directory, so a path copied from PowerShell at the repository root has the
    same meaning as it does for ``reconstruct_slm``.
    """

    stage = _resolve(stage_dir, Path.cwd())
    input_dir = stage / "amplitude_to_play"
    output_dir = stage / "ccd_captured"
    log_dir = stage / "acquisition_logs"
    phase_dir = stage / "phase_to_play"
    phase_files = sorted(phase_dir.glob("*.bmp")) if phase_dir.is_dir() else []
    if len(phase_files) != 1:
        raise FileNotFoundError(
            f"--stage-dir expects exactly one phase BMP under {phase_dir}; "
            f"found {len(phase_files)}. Reconstruct/copy the stage phase mask first, "
            "or use the explicit --input-dir/--output-dir/--phase-mask mode."
        )
    phase_manifest = phase_dir / "reconstruction_manifest.csv"
    return (
        input_dir,
        output_dir,
        log_dir,
        phase_files[0],
        phase_manifest,
    )


def _reconstruct_missing_stage_amplitudes(
    raw: dict[str, Any], stage_dir: str | Path, input_dir: Path
) -> bool:
    """Rebuild compact 17 um payloads downloaded from the training server.

    Server-side stage exports intentionally transport 478x478 PNGs instead of
    1024x1024 BMPs.  A laboratory ``--stage-dir`` acquisition is otherwise
    easy to start one command too early.  Only the current, unambiguous
    Meadowlark 17 um contract is reconstructed automatically; legacy or
    differently sized devices still require an explicit ``reconstruct_slm``
    command.
    """

    if any(input_dir.glob("*.bmp")):
        return False
    stage = _resolve(stage_dir, Path.cwd())
    compact_dir = stage / "compact_amplitude"
    compact_files = sorted(compact_dir.glob("*.png")) if compact_dir.is_dir() else []
    if not compact_files:
        return False

    slm = dict(raw.get("amplitude_slm", {}))
    expected_size = tuple(
        int(value)
        for value in slm.get(
            "expected_resolution_wh",
            (slm.get("width", 0), slm.get("height", 0)),
        )
    )
    pixel_pitch_um = float(slm.get("pixel_pitch_um", 0.0))
    if expected_size != (1024, 1024) or abs(pixel_pitch_um - 17.0) > 1.0e-9:
        raise FileNotFoundError(
            f"No amplitude BMP files found in {input_dir}. Compact PNGs do exist, "
            "but automatic reconstruction is restricted to the 1024x1024, "
            "17 um Meadowlark contract. Run reconstruct_slm explicitly."
        )

    try:
        from .reconstruct_slm import reconstruct_directory
    except ImportError:  # direct execution from workflows/
        from workflows.reconstruct_slm import reconstruct_directory

    reconstruct_directory(
        compact_dir,
        input_dir,
        slm_size_wh=expected_size,
        scale_factor=1,
        center_xy=(expected_size[0] / 2.0, expected_size[1] / 2.0),
    )
    print(
        "[stage preparation] reconstructed "
        f"{len(compact_files)} compact 478x478 PNGs into {input_dir}",
        flush=True,
    )
    return True


def _files_from_manifest(input_dir: Path, manifest: Path) -> list[Path]:
    """Return the exact, sorted amplitude allowlist recorded by a CSV manifest.

    The manifest is deliberately an allowlist rather than a directory hint.  This
    lets a quick diagnostic and a formal run share one (potentially large) BMP
    directory without copying files or accidentally playing frames from another
    profile.  It accepts the export manifests' ``amplitude_file`` /
    ``amplitude_bmp`` columns and ``reconstruct_slm``'s ``output_bmp`` column;
    every selected value must still be a unique, existing plain BMP basename.
    """

    manifest = manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Amplitude selection manifest is missing: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        supported_columns = ("amplitude_file", "amplitude_bmp", "output_bmp")
        column = next(
            (candidate for candidate in supported_columns if candidate in fieldnames),
            None,
        )
        if column is None:
            raise ValueError(
                "Amplitude selection manifest must contain one of "
                f"{', '.join(supported_columns)}: {manifest}"
            )
        rows = list(reader)
        names = [str(row.get(column, "")).strip() for row in rows]
    if not names or any(not name for name in names):
        raise ValueError(f"Amplitude selection manifest is empty/incomplete: {manifest}")
    if len(names) != len({name.casefold() for name in names}):
        raise ValueError(f"Amplitude selection manifest contains duplicate files: {manifest}")
    paths: list[Path] = []
    for name in names:
        candidate_name = Path(name)
        if (
            candidate_name.name != name
            or "/" in name
            or "\\" in name
            or ":" in name
            or candidate_name.suffix.lower() != ".bmp"
        ):
            raise ValueError(
                "Manifest amplitude entries must be plain BMP basenames (no paths): "
                f"{name!r}"
            )
        candidate = input_dir / name
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Manifest-selected amplitude BMP is missing: {candidate}"
            )
        declared_sha = str(rows[len(paths)].get("output_sha256", "")).strip().lower()
        if column == "output_bmp" and declared_sha:
            observed_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if observed_sha != declared_sha:
                raise RuntimeError(
                    "Reconstructed amplitude BMP SHA-256 differs from its manifest: "
                    f"{candidate.name}"
                )
        paths.append(candidate)
    return sorted(paths, key=lambda value: value.name)


def _phase_mask_metadata(
    path: Path,
    expected_size_wh: tuple[int, int],
    expected_manifest: Path | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required phase mask is missing: {path}")
    if path.suffix.lower() != ".bmp":
        raise ValueError(f"Phase mask must be a BMP file: {path.name}")
    with Image.open(path) as image:
        if image.format != "BMP" or image.mode != "L":
            raise ValueError(
                f"Phase mask must be native 8-bit grayscale BMP; "
                f"got format={image.format} mode={image.mode}"
            )
        if tuple(image.size) != expected_size_wh:
            raise ValueError(
                f"Phase mask {path.name} size={image.size}, "
                f"expected={expected_size_wh}"
            )
    observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_metadata: dict[str, Any] | None = None
    if expected_manifest is not None:
        manifest_path = expected_manifest.expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Expected phase reconstruction manifest is missing: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        matches = [row for row in rows if str(row.get("output_bmp", "")).strip() == path.name]
        if len(matches) != 1:
            raise RuntimeError(
                "Phase reconstruction manifest must contain exactly one row for "
                f"{path.name}; found {len(matches)}"
            )
        declared_sha = str(matches[0].get("output_sha256", "")).strip().lower()
        if len(declared_sha) != 64:
            raise RuntimeError(
                "Phase reconstruction manifest has no valid output_sha256 for "
                f"{path.name}"
            )
        if observed_sha != declared_sha:
            raise RuntimeError(
                f"Phase BMP SHA-256 differs from reconstruction manifest: {path.name}"
            )
        manifest_metadata = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "declared_output_sha256": declared_sha,
            "verified": True,
        }
    return {
        "path": str(path),
        "basename": path.name,
        "size_wh": list(expected_size_wh),
        "mode": "L",
        "sha256": observed_sha,
        "reconstruction_manifest": manifest_metadata,
        "controlled_by_workflow": False,
    }


def _resolve_detector_geometry(
    camera_config: dict[str, Any], base: Path
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    """Load an optional immutable homography and return raw camera settings.

    The returned camera settings disable the legacy resize/bit-depth path.  The
    acquisition loop applies the homography to the raw device ROI first, then
    performs the configured fixed bit-depth conversion exactly once.
    """

    geometry_config = camera_config.get("detector_geometry")
    if geometry_config is None:
        return None, None, dict(camera_config)
    if not isinstance(geometry_config, dict):
        raise ValueError("camera.detector_geometry must be a mapping")
    if not bool(geometry_config.get("enabled", False)):
        return None, None, dict(camera_config)
    forbidden_flip_flags = []
    for owner, mapping in (
        ("camera", camera_config),
        ("camera.detector_geometry", geometry_config),
    ):
        for key in (
            "flip_vertical",
            "flip_horizontal",
            "flip_vertical_after_warp",
            "flip_horizontal_after_warp",
            "downstream_loader_flip_vertical",
            "downstream_loader_flip_horizontal",
        ):
            if bool(mapping.get(key, False)):
                forbidden_flip_flags.append(f"{owner}.{key}")
    if forbidden_flip_flags:
        raise ValueError(
            "canonical homography mode cannot be mixed with camera/downstream flips; "
            f"set these false and correct the logical TL/TR/BR/BL labels: "
            f"{forbidden_flip_flags}"
        )
    contract_raw = geometry_config.get("contract_file")
    if not contract_raw:
        raise ValueError(
            "camera.detector_geometry.enabled=true requires contract_file"
        )
    contract_path = _resolve(contract_raw, base)
    expected_sha = geometry_config.get("expected_file_sha256")
    if not expected_sha:
        raise ValueError(
            "formal homography acquisition requires expected_file_sha256"
        )
    contract, metadata = load_geometry_contract(
        contract_path, expected_file_sha256=str(expected_sha)
    )
    configured_roi = verify_camera_roi(camera_config)
    contract_roi = tuple(
        int(value)
        for value in contract["source"]["device_roi_xywh_full_sensor"]
    )
    if configured_roi != contract_roi:
        raise ValueError(
            "camera.device_roi_xywh does not match the homography contract: "
            f"configured={configured_roi}, contract={contract_roi}"
        )
    target_size = [int(value) for value in contract["destination"]["size_wh"]]
    configured_saved = camera_config.get("saved_frame_size_wh")
    if configured_saved is not None and [int(value) for value in configured_saved] != target_size:
        raise ValueError(
            "camera.saved_frame_size_wh must equal the homography target "
            f"{target_size}"
        )
    if camera_config.get("saved_frame_bit_depth") not in {8, 16}:
        raise ValueError(
            "homography acquisition requires a fixed saved_frame_bit_depth of 8 or 16"
        )
    raw_camera_config = dict(camera_config)
    raw_camera_config.pop("detector_geometry", None)
    raw_camera_config["saved_frame_size_wh"] = None
    raw_camera_config["saved_frame_resize_mode"] = "none"
    raw_camera_config["saved_frame_bit_depth"] = None
    raw_camera_config["saved_frame_input_range"] = None
    return contract, metadata, raw_camera_config


def _save_rectified_capture(
    path: Path,
    value: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, value)
    elif suffix in {".png", ".tif", ".tiff"}:
        Image.fromarray(value).save(
            path, format="PNG" if suffix == ".png" else "TIFF"
        )
    else:
        raise ValueError(f"unsupported rectified capture extension: {suffix}")


def _capture_with_optional_geometry(
    camera: Any,
    capture_path: Path,
    camera_config: dict[str, Any],
    geometry_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    if geometry_contract is None:
        camera.capture(capture_path)
        return dict(camera.device_info().get("last_capture") or {})

    raw_path = capture_path.parent / f".{capture_path.stem}.raw_device_roi.npy"
    if raw_path.exists():
        raise FileExistsError(f"stale temporary raw detector frame exists: {raw_path}")
    try:
        camera.capture(raw_path)
        raw_capture_info = dict(camera.device_info().get("last_capture") or {})
        raw = np.load(raw_path, allow_pickle=False)
        raw = np.asarray(raw).squeeze()
        rectified = warp_detector_intensity(raw, geometry_contract)
        converted = convert_detector_bit_depth(
            rectified,
            int(camera_config["saved_frame_bit_depth"]),
            (
                None
                if camera_config.get("saved_frame_input_range") is None
                else tuple(float(value) for value in camera_config["saved_frame_input_range"])
            ),
        )
        _save_rectified_capture(capture_path, converted)
    finally:
        raw_path.unlink(missing_ok=True)
    return {
        "source_size_wh": raw_capture_info.get("source_size_wh"),
        "saved_size_wh": [int(converted.shape[1]), int(converted.shape[0])],
        "resize_mode": "homography_bilinear_intensity",
        "resized": raw_capture_info.get("source_size_wh")
        != [int(converted.shape[1]), int(converted.shape[0])],
        "dtype": str(converted.dtype),
        "source_dtype": str(raw.dtype),
        "saved_frame_bit_depth": int(camera_config["saved_frame_bit_depth"]),
        "saved_frame_input_range": camera_config.get("saved_frame_input_range"),
        "detector_geometry_applied": True,
        "saved_frame_orientation": "canonical_model_xy",
        "downstream_loader_flip_required": False,
        "raw_capture_info": raw_capture_info,
    }


def run(
    config_path: str | Path,
    *,
    input_override: str | Path | None = None,
    output_override: str | Path | None = None,
    phase_override: str | Path | None = None,
    stage_override: str | Path | None = None,
    log_override: str | Path | None = None,
    file_manifest_override: str | Path | None = None,
    phase_manifest_override: str | Path | None = None,
    clear_output: bool = False,
    assume_yes: bool = False,
    validate_only: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    raw, config_path = load_yaml_config(config_path)
    base = config_path.parent
    if stage_override is not None:
        if any(value is not None for value in (input_override, output_override, phase_override)):
            raise ValueError(
                "Do not combine --stage-dir with --input-dir, --output-dir, or "
                "--phase-mask"
            )
        (
            input_dir,
            output_dir,
            log_dir,
            resolved_stage_phase,
            resolved_stage_phase_manifest,
        ) = _resolve_stage_layout(stage_override)
    else:
        # Explicit CLI paths follow normal shell semantics and resolve from the
        # current working directory.  Only paths written inside YAML resolve
        # relative to that YAML file.  The previous mixed behavior was a major
        # source of duplicated ``.../config/experiments/...`` paths on Windows.
        input_dir = (
            _resolve(input_override, Path.cwd())
            if input_override is not None
            else _resolve(raw["input_dir"], base)
        )
        output_dir = (
            _resolve(output_override, Path.cwd())
            if output_override is not None
            else _resolve(raw["output_dir"], base)
        )
        log_dir = _resolve(raw.get("log_dir", "../workspace/logs"), base)
        resolved_stage_phase = None
        resolved_stage_phase_manifest = None
    if log_override is not None:
        log_dir = _resolve(log_override, Path.cwd())
    selection_manifest = (
        _resolve(file_manifest_override, Path.cwd())
        if file_manifest_override is not None
        else None
    )
    if stage_override is not None:
        _reconstruct_missing_stage_amplitudes(raw, stage_override, input_dir)
    files = (
        _files_from_manifest(input_dir, selection_manifest)
        if selection_manifest is not None
        else sorted(input_dir.glob("*.bmp"))
    )
    if not files:
        raise FileNotFoundError(f"No amplitude BMP files found in {input_dir}")
    configured_limit = limit if limit is not None else raw.get("max_files")
    if configured_limit is not None:
        configured_limit = int(configured_limit)
        if configured_limit <= 0:
            raise ValueError("max_files/--limit must be positive")
        files = files[:configured_limit]
    extension = str(raw.get("output_extension", ".npy")).lower()
    if extension not in CAPTURE_SUFFIXES:
        raise ValueError(f"output_extension must be one of {sorted(CAPTURE_SUFFIXES)}")
    if clear_output and not validate_only:
        _clear_capture_files(output_dir)
    if not validate_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = [output_dir / f"{path.stem}{extension}" for path in files]
        existing = [path for path in existing if path.exists()]
        if existing:
            raise FileExistsError(
                f"{len(existing)} captures already exist (first: {existing[0]}). "
                "Clean the output folder or pass --clear-output."
            )
    phase_config = dict(raw.get("phase_slm", {}))
    expected_phase_size = tuple(
        int(value)
        for value in phase_config.get(
            "expected_resolution_wh",
            (phase_config.get("width", 1920), phase_config.get("height", 1200)),
        )
    )
    raw_phase_path = resolved_stage_phase or phase_override or raw.get("phase_mask_file")
    raw_phase_manifest = phase_manifest_override or resolved_stage_phase_manifest
    phase_required = bool(raw.get("require_phase_mask", False))
    if raw_phase_path is None and phase_required:
        raise ValueError(
            "This acquisition requires the exact phase BMP. Set phase_mask_file "
            "in YAML or pass --phase-mask."
        )
    phase_base = Path.cwd() if phase_override is not None else base
    phase_metadata = (
        _phase_mask_metadata(
            _resolve(raw_phase_path, phase_base),
            expected_phase_size,
            (
                _resolve(raw_phase_manifest, Path.cwd())
                if raw_phase_manifest is not None
                else None
            ),
        )
        if raw_phase_path is not None
        else None
    )
    # Resolve both runtimes before asking the operator to prepare a phase mask.
    # This catches an unset DVP_PYTHON without opening either device.
    camera_config = dict(raw["camera"])
    verify_camera_roi(camera_config)
    geometry_contract, geometry_metadata, raw_camera_config = _resolve_detector_geometry(
        camera_config, base
    )
    slm_driver = build_slm(dict(raw["amplitude_slm"]), base)
    camera_driver = build_camera(raw_camera_config, base)
    slm_driver.validate_runtime()
    camera_driver.validate_runtime()
    slm_driver.validate_files(files)

    readiness_report = {
        "config": str(config_path),
        "stage_dir": str(input_dir.parent) if stage_override is not None else None,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "play_count": len(files),
        "selection_manifest": (
            None
            if selection_manifest is None
            else {
                "path": str(selection_manifest),
                "sha256": hashlib.sha256(selection_manifest.read_bytes()).hexdigest(),
            }
        ),
        "phase_mask": phase_metadata,
        "detector_geometry": geometry_metadata,
        "orientation_canonicalized": geometry_contract is not None,
        "detector_processing_mode": (
            "canonical_homography" if geometry_contract is not None else "legacy_rectangle_resize"
        ),
        "amplitude_slm": slm_driver.device_info(),
        "camera": camera_driver.device_info(),
        "validate_only": bool(validate_only),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "resolved_devices.json").write_text(
        json.dumps(readiness_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if validate_only:
        print(
            f"Validated {len(files)} amplitude BMP(s), phase mask, Meadowlark "
            f"runtime/LUT and camera runtime; report: {log_dir / 'resolved_devices.json'}"
        )
        return readiness_report

    if bool(raw.get("confirm_before_start", True)) and not assume_yes:
        phase_text = (
            "No phase file was registered"
            if phase_metadata is None
            else (
                f"phase={phase_metadata['basename']}\n"
                f"phase_sha256={phase_metadata['sha256']}"
            )
        )
        answer = input(
            f"Found {len(files)} amplitude BMPs in:\n  {input_dir}\n"
            f"{phase_text}\n"
            "Confirm that this exact phase mask is visible on the phase SLM, "
            "then enter y: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            raise KeyboardInterrupt("operator cancelled acquisition")

    settle_seconds = float(raw.get("settle_delay_ms", 40.0)) / 1000.0
    rows: list[dict[str, Any]] = []
    with ExitStack() as stack:
        slm = stack.enter_context(slm_driver)
        camera = stack.enter_context(camera_driver)
        verify_camera_roi(camera_config, camera.device_info())
        slm.preload_files(files)
        device_report = {
            "config": str(config_path),
            "stage_dir": str(input_dir.parent) if stage_override is not None else None,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "play_count": len(files),
            "selection_manifest": readiness_report["selection_manifest"],
            "settle_delay_ms": settle_seconds * 1000.0,
            "phase_mask": phase_metadata,
            "detector_geometry": geometry_metadata,
            "orientation_canonicalized": geometry_contract is not None,
            "detector_processing_mode": (
                "canonical_homography"
                if geometry_contract is not None
                else "legacy_rectangle_resize"
            ),
            "amplitude_slm": slm.device_info(),
            "camera": camera.device_info(),
        }
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "resolved_devices.json").write_text(
            json.dumps(device_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for index, amplitude_path in enumerate(files):
            capture_path = output_dir / f"{amplitude_path.stem}{extension}"
            # Bind the exact frame before it is sent to the display SDK.  This
            # proves which amplitude payload was played even if the BMP is
            # accidentally modified after acquisition.
            amplitude_bmp_sha256 = hashlib.sha256(
                amplitude_path.read_bytes()
            ).hexdigest()
            slm.display_file(amplitude_path)
            time.sleep(settle_seconds)
            capture_info = _capture_with_optional_geometry(
                camera, capture_path, camera_config, geometry_contract
            )
            requested_roi = verify_camera_roi(camera_config)
            expected_size = (
                None
                if requested_roi is None
                else [int(requested_roi[2]), int(requested_roi[3])]
            )
            if expected_size is not None:
                source_size = capture_info.get("source_size_wh")
                if source_size != expected_size:
                    capture_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        "Camera source frame does not match device_roi_xywh: "
                        f"expected={expected_size}, source={source_size}"
                    )
            configured_saved_size = camera_config.get("saved_frame_size_wh")
            expected_saved_size = (
                [int(value) for value in configured_saved_size]
                if configured_saved_size is not None else expected_size
            )
            saved_size = capture_info.get("saved_size_wh")
            if expected_saved_size is not None and saved_size != expected_saved_size:
                capture_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Saved CCD frame does not match camera.saved_frame_size_wh: "
                    f"expected={expected_saved_size}, saved={saved_size}"
                )
            row = {
                "play_index": index,
                "amplitude_bmp": amplitude_path.name,
                "amplitude_bmp_sha256": amplitude_bmp_sha256,
                "ccd_capture": capture_path.name,
                "captured_utc": datetime.now(timezone.utc).isoformat(),
                "camera_exposure_us": (
                    camera.device_info().get("Exposure")
                    if camera.device_info().get("Exposure") is not None
                    else raw.get("camera", {}).get("exposure_us")
                ),
                "camera_device_roi_xywh": json.dumps(
                    camera.device_info().get("device_roi_xywh"), ensure_ascii=False
                ),
                "camera_source_size_wh": json.dumps(
                    capture_info.get("source_size_wh"), ensure_ascii=False
                ),
                "saved_frame_size_wh": json.dumps(
                    capture_info.get("saved_size_wh"), ensure_ascii=False
                ),
                "saved_frame_resize_mode": capture_info.get("resize_mode"),
                "saved_dtype": capture_info.get("dtype"),
                "output_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                "detector_geometry_file_sha256": (
                    None
                    if geometry_metadata is None
                    else geometry_metadata["file_sha256"]
                ),
                "detector_geometry_payload_sha256": (
                    None
                    if geometry_metadata is None
                    else geometry_metadata["payload_sha256"]
                ),
                "orientation_canonicalized": geometry_contract is not None,
                "saved_frame_orientation": (
                    capture_info.get("saved_frame_orientation")
                    if geometry_contract is not None
                    else "legacy_camera_native"
                ),
                "downstream_loader_flip_required": (
                    False if geometry_contract is not None else "legacy_config_defined"
                ),
                "downstream_loader_flip_vertical_required": (
                    False if geometry_contract is not None else "legacy_config_defined"
                ),
                "downstream_loader_flip_horizontal_required": (
                    False if geometry_contract is not None else "legacy_config_defined"
                ),
                "detector_processing_order": (
                    "raw_device_roi>homography_bilinear_intensity>fixed_bit_depth>save"
                    if geometry_contract is not None
                    else "device_roi>legacy_resize>fixed_bit_depth>save"
                ),
                "background_subtraction": False,
                "per_frame_minmax_normalization": False,
                "frame_number": index,
                "phase_mask": (
                    None if phase_metadata is None else phase_metadata["basename"]
                ),
                "phase_mask_sha256": (
                    None if phase_metadata is None else phase_metadata["sha256"]
                ),
                "phase_manifest_sha256": (
                    None
                    if phase_metadata is None
                    or phase_metadata["reconstruction_manifest"] is None
                    else phase_metadata["reconstruction_manifest"]["sha256"]
                ),
                "phase_manifest_verified": (
                    False
                    if phase_metadata is None
                    or phase_metadata["reconstruction_manifest"] is None
                    else bool(
                        phase_metadata["reconstruction_manifest"]["verified"]
                    )
                ),
            }
            rows.append(row)
            size_text = ""
            if capture_info.get("source_size_wh") and capture_info.get("saved_size_wh"):
                source_width, source_height = capture_info["source_size_wh"]
                saved_width, saved_height = capture_info["saved_size_wh"]
                size_text = (
                    f" [{source_width}x{source_height} -> "
                    f"{saved_width}x{saved_height}, {capture_info.get('resize_mode')}]"
                )
            print(
                f"[acquire] {index + 1}/{len(files)} "
                f"{amplitude_path.name} -> {capture_path.name}{size_text}"
            )
        device_report["camera"] = camera.device_info()
        device_report["last_processed_capture"] = capture_info
        (log_dir / "resolved_devices.json").write_text(
            json.dumps(device_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    with (log_dir / "capture_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Captured {len(rows)} frames under {output_dir}")
    return {
        "count": len(rows),
        "stage_dir": str(input_dir.parent) if stage_override is not None else None,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "selection_manifest": readiness_report["selection_manifest"],
        "detector_geometry": geometry_metadata,
        "orientation_canonicalized": geometry_contract is not None,
        "detector_processing_mode": (
            "canonical_homography" if geometry_contract is not None else "legacy_rectangle_resize"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Play selected amplitude BMPs and save same-name CCD frames; bind a "
            "manifest for formal acquisition"
        )
    )
    parser.add_argument("--config", default="configs/tucam_windows.yaml")
    parser.add_argument(
        "--stage-dir",
        default=None,
        help=(
            "Stage directory containing amplitude_to_play/ and exactly one "
            "phase_to_play/*.bmp; CCD and logs stay under this stage"
        ),
    )
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Override acquisition log directory (resolved from the working directory)",
    )
    parser.add_argument(
        "--file-manifest",
        default=None,
        help=(
            "CSV allowlist containing amplitude_file, amplitude_bmp, or "
            "reconstruct_slm output_bmp; only these plain-basename BMPs are played"
        ),
    )
    parser.add_argument(
        "--phase-mask",
        default=None,
        help="Exact 1920x1200 phase BMP already loaded by the operator",
    )
    parser.add_argument(
        "--phase-manifest",
        default=None,
        help=(
            "Optional reconstruction_manifest.csv whose output_sha256 must match "
            "the phase BMP; --stage-dir discovers phase_to_play manifest automatically"
        ),
    )
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate BMPs, DLLs, LUT, ROI and camera runtime without opening devices",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(
        args.config,
        input_override=args.input_dir,
        output_override=args.output_dir,
        phase_override=args.phase_mask,
        stage_override=args.stage_dir,
        log_override=args.log_dir,
        file_manifest_override=args.file_manifest,
        phase_manifest_override=args.phase_manifest,
        clear_output=args.clear_output,
        assume_yes=args.yes,
        validate_only=args.validate_only,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
