"""Layer-agnostic amplitude-SLM playback and CCD acquisition.

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

from PIL import Image

try:
    from ..devices import build_camera, build_slm, verify_camera_roi
    from .calibration_common import load_yaml_config
except ImportError:  # direct execution from workflows/
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from devices import build_camera, build_slm, verify_camera_roi
    from workflows.calibration_common import load_yaml_config


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


def _resolve_stage_layout(stage_dir: str | Path) -> tuple[Path, Path, Path, Path]:
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
    return input_dir, output_dir, log_dir, phase_files[0]


def _files_from_manifest(input_dir: Path, manifest: Path) -> list[Path]:
    """Return the exact, sorted amplitude allowlist recorded by a CSV manifest.

    The manifest is deliberately an allowlist rather than a directory hint.  This
    lets a quick diagnostic and a formal run share one (potentially large) BMP
    directory without copying files or accidentally playing frames from another
    profile.
    """

    manifest = manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Amplitude selection manifest is missing: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        column = (
            "amplitude_file"
            if "amplitude_file" in fieldnames
            else "amplitude_bmp" if "amplitude_bmp" in fieldnames else None
        )
        if column is None:
            raise ValueError(
                f"Amplitude selection manifest must contain amplitude_file or "
                f"amplitude_bmp: {manifest}"
            )
        names = [str(row.get(column, "")).strip() for row in reader]
    if not names or any(not name for name in names):
        raise ValueError(f"Amplitude selection manifest is empty/incomplete: {manifest}")
    if len(names) != len(set(names)):
        raise ValueError(f"Amplitude selection manifest contains duplicate files: {manifest}")
    paths: list[Path] = []
    for name in names:
        candidate_name = Path(name)
        if candidate_name.name != name or candidate_name.suffix.lower() != ".bmp":
            raise ValueError(
                "Manifest amplitude entries must be plain BMP basenames (no paths): "
                f"{name!r}"
            )
        candidate = input_dir / name
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Manifest-selected amplitude BMP is missing: {candidate}"
            )
        paths.append(candidate)
    return sorted(paths, key=lambda value: value.name)


def _phase_mask_metadata(
    path: Path,
    expected_size_wh: tuple[int, int],
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
    return {
        "path": str(path),
        "basename": path.name,
        "size_wh": list(expected_size_wh),
        "mode": "L",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "controlled_by_workflow": False,
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
        input_dir, output_dir, log_dir, resolved_stage_phase = _resolve_stage_layout(
            stage_override
        )
    else:
        input_dir = _resolve(input_override or raw["input_dir"], base)
        output_dir = _resolve(output_override or raw["output_dir"], base)
        log_dir = _resolve(raw.get("log_dir", "../workspace/logs"), base)
        resolved_stage_phase = None
    if log_override is not None:
        log_dir = _resolve(log_override, Path.cwd())
    selection_manifest = (
        _resolve(file_manifest_override, Path.cwd())
        if file_manifest_override is not None
        else None
    )
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
    phase_required = bool(raw.get("require_phase_mask", False))
    if raw_phase_path is None and phase_required:
        raise ValueError(
            "This acquisition requires the exact phase BMP. Set phase_mask_file "
            "in YAML or pass --phase-mask."
        )
    phase_metadata = (
        _phase_mask_metadata(_resolve(raw_phase_path, base), expected_phase_size)
        if raw_phase_path is not None
        else None
    )
    # Resolve both runtimes before asking the operator to prepare a phase mask.
    # This catches an unset DVP_PYTHON without opening either device.
    camera_config = dict(raw["camera"])
    verify_camera_roi(camera_config)
    slm_driver = build_slm(dict(raw["amplitude_slm"]), base)
    camera_driver = build_camera(camera_config, base)
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
            "amplitude_slm": slm.device_info(),
            "camera": camera.device_info(),
        }
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "resolved_devices.json").write_text(
            json.dumps(device_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for index, amplitude_path in enumerate(files):
            capture_path = output_dir / f"{amplitude_path.stem}{extension}"
            slm.display_file(amplitude_path)
            time.sleep(settle_seconds)
            camera.capture(capture_path)
            capture_info = camera.device_info().get("last_capture") or {}
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
                "frame_number": index,
                "phase_mask": (
                    None if phase_metadata is None else phase_metadata["basename"]
                ),
                "phase_mask_sha256": (
                    None if phase_metadata is None else phase_metadata["sha256"]
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play every amplitude BMP in a folder and save same-name CCD frames"
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
            "CSV allowlist containing amplitude_file or amplitude_bmp; only these "
            "shared-directory BMPs are played"
        ),
    )
    parser.add_argument(
        "--phase-mask",
        default=None,
        help="Exact 1920x1200 phase BMP already loaded by the operator",
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
        clear_output=args.clear_output,
        assume_yes=args.yes,
        validate_only=args.validate_only,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
