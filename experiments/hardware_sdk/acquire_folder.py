"""Layer-agnostic amplitude-SLM playback and CCD acquisition.

This file intentionally has no Torch/Qwen/model imports.  It may be executed
as a package module or copied with the hardware_sdk folder to a bench PC and
run directly as ``python acquire_folder.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .devices import build_camera, build_slm
except ImportError:  # direct execution from inside hardware_sdk/
    from devices import build_camera, build_slm


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


def run(
    config_path: str | Path,
    *,
    input_override: str | Path | None = None,
    output_override: str | Path | None = None,
    clear_output: bool = False,
    assume_yes: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    input_dir = _resolve(input_override or raw["input_dir"], base)
    output_dir = _resolve(output_override or raw["output_dir"], base)
    log_dir = _resolve(raw.get("log_dir", "../workspace/logs"), base)
    files = sorted(input_dir.glob("*.bmp"))
    if not files:
        raise FileNotFoundError(f"No amplitude BMP files found in {input_dir}")
    extension = str(raw.get("output_extension", ".npy")).lower()
    if extension not in CAPTURE_SUFFIXES:
        raise ValueError(f"output_extension must be one of {sorted(CAPTURE_SUFFIXES)}")
    if clear_output:
        _clear_capture_files(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f"{path.stem}{extension}" for path in files]
    existing = [path for path in existing if path.exists()]
    if existing:
        raise FileExistsError(
            f"{len(existing)} captures already exist (first: {existing[0]}). "
            "Clean the output folder or pass --clear-output."
        )
    # Resolve both runtimes before asking the operator to prepare a phase mask.
    # This catches an unset DVP_PYTHON without opening either device.
    slm_driver = build_slm(dict(raw["amplitude_slm"]), base)
    camera_driver = build_camera(dict(raw["camera"]), base)
    slm_driver.validate_runtime()
    camera_driver.validate_runtime()

    if bool(raw.get("confirm_before_start", True)) and not assume_yes:
        answer = input(
            f"Found {len(files)} amplitude BMPs in:\n  {input_dir}\n"
            "Confirm that the required phase mask is loaded, then enter y: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            raise KeyboardInterrupt("operator cancelled acquisition")

    settle_seconds = float(raw.get("settle_delay_ms", 40.0)) / 1000.0
    rows: list[dict[str, Any]] = []
    with ExitStack() as stack:
        slm = stack.enter_context(slm_driver)
        camera = stack.enter_context(camera_driver)
        slm.preload_files(files)
        device_report = {
            "config": str(config_path),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "play_count": len(files),
            "settle_delay_ms": settle_seconds * 1000.0,
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
            row = {
                "play_index": index,
                "amplitude_bmp": amplitude_path.name,
                "ccd_capture": capture_path.name,
                "captured_utc": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(row)
            print(
                f"[acquire] {index + 1}/{len(files)} "
                f"{amplitude_path.name} -> {capture_path.name}"
            )
    with (log_dir / "capture_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Captured {len(rows)} frames under {output_dir}")
    return {"count": len(rows), "input_dir": str(input_dir), "output_dir": str(output_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play every amplitude BMP in a folder and save same-name CCD frames"
    )
    parser.add_argument("--config", default="configs/acquisition_windows.json")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        input_override=args.input_dir,
        output_override=args.output_dir,
        clear_output=args.clear_output,
        assume_yes=args.yes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
