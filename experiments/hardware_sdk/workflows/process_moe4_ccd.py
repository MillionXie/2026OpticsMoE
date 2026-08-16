"""Prepare captured CCD frames for the canonical 2x2 MoE active region.

This is deliberately independent from any training experiment.  It selects one
fixed ROI (or the whole input), resizes that entire rectangle to 956x956, and
writes 8-bit grayscale PNGs.  Orientation flips belong to the downstream
experiment and are intentionally not implemented here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


TARGET_SIZE = (956, 956)
SUPPORTED_SUFFIXES = {".bmp", ".png", ".tif", ".tiff"}


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _read_frame(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].astype(np.float32).mean(axis=-1)
    if array.ndim != 2:
        raise RuntimeError(f"{path} is not a two-dimensional CCD frame")
    value = np.asarray(array, dtype=np.float32)
    if not np.isfinite(value).all():
        raise RuntimeError(f"{path} contains NaN or Inf")
    return value


def _select_roi(value: np.ndarray, roi: list[int] | None, path: Path) -> np.ndarray:
    if roi is None:
        return value
    x, y, width, height = (int(item) for item in roi)
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("roi_xywh must contain nonnegative x/y and positive width/height")
    if x + width > value.shape[1] or y + height > value.shape[0]:
        raise RuntimeError(f"ROI {roi} is outside {path} with shape {value.shape}")
    return value[y : y + height, x : x + width]


def _resize_entire_roi(value: np.ndarray) -> np.ndarray:
    if tuple(value.shape[::-1]) == TARGET_SIZE:
        return value
    height, width = value.shape
    downsample = width >= TARGET_SIZE[0] and height >= TARGET_SIZE[1]
    resampling = Image.Resampling.BOX if downsample else Image.Resampling.BILINEAR
    image = Image.fromarray(value.astype(np.float32), mode="F")
    return np.asarray(image.resize(TARGET_SIZE, resample=resampling), dtype=np.float32)


def _intensity_range(
    frames: list[tuple[Path, np.ndarray]], settings: dict[str, Any]
) -> tuple[float, float, str]:
    mode = str(settings.get("mode", "global_percentile"))
    if mode == "fixed_range":
        low = float(settings["black_level"])
        high = float(settings["white_level"])
    elif mode == "global_percentile":
        stride = max(1, int(settings.get("sample_stride", 16)))
        sampled = np.concatenate([value[::stride, ::stride].reshape(-1) for _, value in frames])
        low = float(np.percentile(sampled, float(settings.get("lower_percentile", 0.1))))
        high = float(np.percentile(sampled, float(settings.get("upper_percentile", 99.9))))
    else:
        raise ValueError("intensity.mode must be fixed_range or global_percentile")
    if not high > low:
        raise RuntimeError(f"Invalid common intensity range [{low}, {high}]")
    return low, high, mode


def process(config_path: Path, input_override: str | None, output_override: str | None) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    input_dir = _resolve(config_path, input_override or raw["input_dir"])
    output_dir = _resolve(config_path, output_override or raw["output_dir"])
    if input_dir == output_dir:
        raise ValueError("input_dir and output_dir must differ")
    if tuple(raw.get("target_size_wh", TARGET_SIZE)) != TARGET_SIZE:
        raise ValueError("Canonical MoE4 physical target_size_wh must be [956, 956]")
    if any(bool(raw.get(name, False)) for name in ("flip_vertical", "flip_horizontal")):
        raise ValueError("This standalone processor must not flip CCD images")
    roi = raw.get("roi_xywh")
    paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No CCD images found in {input_dir}")
    if len({path.stem for path in paths}) != len(paths):
        raise RuntimeError("Input files with different extensions share a basename")
    selected = []
    for path in paths:
        source = _read_frame(path)
        selected.append((path, _select_roi(source, roi, path), source.shape))
    low, high, intensity_mode = _intensity_range(
        [(path, value) for path, value, _ in selected], raw.get("intensity", {})
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, (path, value, source_shape) in enumerate(selected, 1):
        resized = _resize_entire_roi(value)
        encoded = np.clip((resized - low) * (255.0 / (high - low)), 0.0, 255.0)
        encoded = np.rint(encoded).astype(np.uint8)
        destination = output_dir / f"{path.stem}.png"
        Image.fromarray(encoded, mode="L").save(destination)
        rows.append(
            {
                "source": str(path),
                "output": str(destination),
                "source_height": int(source_shape[0]),
                "source_width": int(source_shape[1]),
                "roi_xywh": "" if roi is None else ",".join(map(str, roi)),
                "output_height": 956,
                "output_width": 956,
                "output_dtype": "uint8",
                "flip_applied": False,
            }
        )
        if index % 20 == 0 or index == len(selected):
            print(f"[process_moe4_ccd] {index}/{len(selected)}", flush=True)
    with (output_dir / "processing_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "frames": len(rows),
        "roi_xywh": roi,
        "resize_rule": "resize the entire selected ROI; no center crop",
        "target_size_wh": [956, 956],
        "output": "8-bit grayscale PNG (PIL mode L)",
        "flip_applied": False,
        "intensity_mode": intensity_mode,
        "common_black_level": low,
        "common_white_level": high,
    }
    (output_dir / "processing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MoE4 CCD images as uint8 956x956 PNG")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    process(Path(args.config).expanduser().resolve(), args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
