"""Optional, offline background subtraction for hardware CCD captures.

This workflow is deliberately separate from formal acquisition.  Raw camera
ROI frames remain untouched.  A background is measured with the current phase
mask left in place and an all-zero amplitude pattern, then subtracted into a
new directory using ``maximum(raw - background, 0)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from ..devices import build_camera, build_slm, verify_camera_roi
    from .calibration_common import (
        json_dump,
        load_frame,
        load_yaml_config,
        median_capture,
        resolve_path,
        utc_now,
    )
except ImportError:  # direct execution from workflows/
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from devices import build_camera, build_slm, verify_camera_roi
    from workflows.calibration_common import (
        json_dump,
        load_frame,
        load_yaml_config,
        median_capture,
        resolve_path,
        utc_now,
    )


INPUT_SUFFIXES = {".npy", ".png", ".tif", ".tiff"}
OUTPUT_SUFFIXES = {".npy", ".png", ".tif", ".tiff"}


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("optional_background") or {})


def _expected_shape(config: dict[str, Any]) -> tuple[int, int]:
    saved_size = config["camera"].get("saved_frame_size_wh")
    if saved_size is not None:
        return int(saved_size[1]), int(saved_size[0])
    roi = verify_camera_roi(dict(config["camera"]))
    if roi is None:
        raise RuntimeError("Optional background requires camera.device_roi_xywh")
    return int(roi[3]), int(roi[2])


def _ensure_zero_bmp(
    config: dict[str, Any], config_path: Path, configured_path: str | Path
) -> Path:
    path = resolve_path(configured_path, config_path.parent)
    width = int(config["amplitude_slm"]["width"])
    height = int(config["amplitude_slm"]["height"])
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (width, height), 0).save(path, format="BMP")
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (width, height):
            raise ValueError(
                f"Zero amplitude BMP must be 8-bit L and {width}x{height}: "
                f"mode={image.mode}, size={image.size}, path={path}"
            )
        if np.asarray(image).max() != 0:
            raise ValueError(f"Zero amplitude BMP contains nonzero pixels: {path}")
    return path


def _save_preview(path: Path, value: np.ndarray) -> None:
    finite = np.asarray(value, dtype=np.float32)
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        preview = np.zeros(finite.shape, dtype=np.uint8)
    else:
        preview = np.clip((finite - low) / (high - low), 0.0, 1.0)
        preview = np.rint(preview * 255.0).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)


def capture_background(
    config_path: str | Path,
    *,
    output_override: str | Path | None = None,
    assume_yes: bool = False,
) -> dict[str, Any]:
    config, config_path = load_yaml_config(config_path)
    settings = _settings(config)
    output_dir = resolve_path(
        output_override or settings.get("background_dir", "../artifacts/optional_background"),
        config_path.parent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    zero_bmp = _ensure_zero_bmp(
        config,
        config_path,
        settings.get(
            "zero_amplitude_bmp",
            "../artifacts/calibration/masks/amplitude/amplitude_zero.bmp",
        ),
    )
    frame_count = int(settings.get("frames", 10))
    if frame_count <= 0:
        raise ValueError("optional_background.frames must be positive")

    camera_config = dict(config["camera"])
    expected_shape = _expected_shape(config)
    slm_driver = build_slm(dict(config["amplitude_slm"]), config_path.parent)
    camera_driver = build_camera(camera_config, config_path.parent)
    slm_driver.validate_runtime()
    camera_driver.validate_runtime()
    if bool(config.get("confirm_before_start", True)) and not assume_yes:
        answer = input(
            "Keep the SAME phase mask used for the target captures loaded.\n"
            f"The amplitude SLM will display the all-zero BMP:\n  {zero_bmp}\n"
            f"Capture {frame_count} background frames now? Enter y: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            raise KeyboardInterrupt("operator cancelled background capture")

    with ExitStack() as stack:
        slm = stack.enter_context(slm_driver)
        camera = stack.enter_context(camera_driver)
        verify_camera_roi(camera_config, camera.device_info())
        slm.display_file(zero_bmp)
        time.sleep(float(config.get("settle_delay_ms", 200)) / 1000.0)
        background, frames_metadata = median_capture(
            camera, output_dir, "optional_background", frame_count
        )
        camera_info = camera.device_info()
    if background.shape != expected_shape:
        raise RuntimeError(
            f"Background shape {background.shape} does not match camera ROI {expected_shape}"
        )

    # The normal hardware workflow stores 8-bit PNG frames.  Keep the optional
    # background in the same directly inspectable representation.
    background = np.rint(background).clip(0, 255).astype(np.uint8)
    background_path = output_dir / str(
        settings.get("background_filename", "background.png")
    )
    if background_path.suffix.lower() != ".png":
        raise ValueError("optional_background.background_filename must end in .png")
    Image.fromarray(background, mode="L").save(background_path, format="PNG")
    _save_preview(output_dir / "background_preview.png", background)
    metadata = {
        "kind": "optional_system_background",
        "definition": "laser on; current phase mask retained; amplitude SLM all-zero",
        "subtraction_formula": "maximum(raw.astype(float32) - background, 0)",
        "frame_count": frame_count,
        "aggregation": "pixelwise_median",
        "shape": list(background.shape),
        "dtype": str(background.dtype),
        "background_file": str(background_path),
        "min": float(background.min()),
        "max": float(background.max()),
        "mean": float(background.mean()),
        "zero_amplitude_bmp": str(zero_bmp),
        "camera": camera_info,
        "frames": frames_metadata,
        "timestamp": utc_now(),
    }
    json_dump(output_dir / "background_metadata.json", metadata)
    print(f"Optional background saved under {output_dir}")
    return metadata


def _save_corrected(path: Path, value: np.ndarray) -> None:
    if path.suffix.lower() == ".npy":
        np.save(path, value.astype(np.float32, copy=False))
    elif path.suffix.lower() == ".png":
        converted = np.rint(value).clip(0, 255).astype(np.uint8)
        Image.fromarray(converted, mode="L").save(path, format="PNG")
    else:
        Image.fromarray(value.astype(np.float32, copy=False), mode="F").save(path)


def subtract_background_directory(
    input_dir: str | Path,
    background_path: str | Path,
    output_dir: str | Path,
    *,
    output_extension: str = ".npy",
    clear_output: bool = False,
) -> dict[str, Any]:
    input_dir = Path(input_dir).expanduser().resolve()
    background_path = Path(background_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    suffix = output_extension.lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix not in OUTPUT_SUFFIXES:
        raise ValueError(f"output_extension must be one of {sorted(OUTPUT_SUFFIXES)}")
    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in INPUT_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No CCD frames found in {input_dir}")
    background = load_frame(background_path).astype(np.float32, copy=False)
    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f"{path.stem}{suffix}" for path in files]
    if any(path.exists() for path in existing):
        raise FileExistsError(
            f"Corrected files already exist under {output_dir}; use --clear-output"
        )

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(files):
        raw = load_frame(source)
        if raw.shape != background.shape:
            raise RuntimeError(
                f"Shape mismatch: {source.name}={raw.shape}, background={background.shape}. "
                "No resize is performed by this optional workflow."
            )
        raw_float = raw.astype(np.float32, copy=False)
        difference = raw_float - background
        corrected = np.maximum(difference, 0.0)
        destination = output_dir / f"{source.stem}{suffix}"
        _save_corrected(destination, corrected)
        rows.append(
            {
                "index": index,
                "source": source.name,
                "output": destination.name,
                "shape": json.dumps(list(raw.shape)),
                "raw_mean": float(raw_float.mean()),
                "background_mean": float(background.mean()),
                "corrected_mean": float(corrected.mean()),
                "corrected_max": float(corrected.max()),
                "clipped_to_zero_fraction": float(np.mean(difference <= 0)),
            }
        )
    manifest = output_dir / "background_subtraction_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input_dir": str(input_dir),
        "background": str(background_path),
        "output_dir": str(output_dir),
        "file_count": len(rows),
        "shape": list(background.shape),
        "output_extension": suffix,
        "normalization": False,
        "resize": False,
        "formula": "maximum(raw.astype(float32) - background, 0)",
        "timestamp": utc_now(),
    }
    json_dump(output_dir / "background_subtraction_summary.json", summary)
    print(f"Background-subtracted {len(rows)} frames into {output_dir}")
    return summary


def subtract_from_config(
    config_path: str | Path,
    *,
    input_override: str | Path | None = None,
    background_override: str | Path | None = None,
    output_override: str | Path | None = None,
    output_extension: str | None = None,
    clear_output: bool = False,
) -> dict[str, Any]:
    config, config_path = load_yaml_config(config_path)
    settings = _settings(config)
    base = config_path.parent
    background_dir = resolve_path(
        settings.get("background_dir", "../artifacts/optional_background"), base
    )
    background_filename = str(settings.get("background_filename", "background.png"))
    return subtract_background_directory(
        resolve_path(
            input_override or settings.get("source_capture_dir", config["output_dir"]),
            base,
        ),
        resolve_path(
            background_override or background_dir / background_filename, base
        ),
        resolve_path(
            output_override
            or settings.get("corrected_output_dir", "../data/ccd_background_subtracted"),
            base,
        ),
        output_extension=output_extension
        or str(settings.get("output_extension", ".png")),
        clear_output=clear_output,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--config", required=True)
    capture_parser.add_argument("--output-dir", default=None)
    capture_parser.add_argument("--yes", action="store_true")
    subtract_parser = subparsers.add_parser("subtract")
    subtract_parser.add_argument("--config", required=True)
    subtract_parser.add_argument("--input-dir", default=None)
    subtract_parser.add_argument("--background", default=None)
    subtract_parser.add_argument("--output-dir", default=None)
    subtract_parser.add_argument("--output-extension", default=None)
    subtract_parser.add_argument("--clear-output", action="store_true")
    args = parser.parse_args()
    if args.phase == "capture":
        capture_background(
            args.config, output_override=args.output_dir, assume_yes=args.yes
        )
    else:
        subtract_from_config(
            args.config,
            input_override=args.input_dir,
            background_override=args.background,
            output_override=args.output_dir,
            output_extension=args.output_extension,
            clear_output=args.clear_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
