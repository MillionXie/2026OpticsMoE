"""Minimal offline processing for manually configured camera-ROI frames.

The quantitative path contains only system-background subtraction and an
optional BOX area downsample.  It performs no affine/homography correction,
registration, per-image normalization, denoising, or contrast enhancement.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from .calibration_common import (
        FRAME_SUFFIXES,
        corrected_frame,
        json_dump,
        load_frame,
        load_yaml_config,
        preview_uint8,
        resolve_path,
        save_tiff,
        utc_now,
    )
except ImportError:
    from calibration_common import (
        FRAME_SUFFIXES,
        corrected_frame,
        json_dump,
        load_frame,
        load_yaml_config,
        preview_uint8,
        resolve_path,
        save_tiff,
        utc_now,
    )


def _saturation_fraction(raw: np.ndarray) -> float:
    if np.issubdtype(raw.dtype, np.integer):
        return float(np.mean(raw >= np.iinfo(raw.dtype).max))
    return 0.0


def _optional_area_resize(
    corrected: np.ndarray, settings: dict[str, Any]
) -> tuple[np.ndarray, bool]:
    if not bool(settings.get("resize_enabled", False)):
        return corrected.astype(np.float32, copy=False), False
    if str(settings.get("resize_mode", "area")).lower() != "area":
        raise ValueError("postprocess.resize_mode must be area")
    width = int(settings["target_width"])
    height = int(settings["target_height"])
    if width <= 0 or height <= 0:
        raise ValueError("postprocess target width/height must be positive")
    source_height, source_width = corrected.shape
    if width > source_width or height > source_height:
        raise ValueError(
            "Area resizing is only enabled for downsampling; set resize_enabled=false "
            "or choose a target no larger than the camera ROI."
        )
    if (source_width, source_height) == (width, height):
        return corrected.astype(np.float32, copy=False), False
    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(corrected.astype(np.float32), mode="F").resize(
        (width, height), resample=resampling.BOX
    )
    return np.asarray(resized, dtype=np.float32), True


def run_batch_postprocess(
    config_path: str | Path,
    input_dir: str | Path,
    background_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config, _ = load_yaml_config(config_path)
    input_dir = resolve_path(input_dir, Path.cwd())
    background_path = resolve_path(background_path, Path.cwd())
    output_dir = resolve_path(output_dir, Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)
    background = load_frame(background_path)
    settings = config["postprocess"]
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No supported raw CCD images found under {input_dir}")

    rows: list[dict[str, Any]] = []
    warning_messages: list[str] = []
    previews: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index, path in enumerate(files):
        raw = load_frame(path)
        if raw.shape != background.shape:
            raise ValueError(
                f"{path.name}: raw shape {raw.shape} does not match background "
                f"shape {background.shape}. The ROI changed; recapture background "
                "after setting the final camera.device_roi_xywh."
            )
        corrected = corrected_frame(raw, background)
        output, resized = _optional_area_resize(corrected, settings)
        saturation = _saturation_fraction(raw)
        if saturation > float(settings.get("saturation_fraction_warning", 0.001)):
            message = f"{path.name}: saturated pixel fraction={saturation:.6f}"
            warning_messages.append(message)
            warnings.warn(message, RuntimeWarning)

        output_files: dict[str, str] = {}
        if bool(settings.get("save_npy", True)):
            npy_path = output_dir / f"{path.stem}.npy"
            np.save(npy_path, output.astype(np.float32))
            output_files["npy"] = npy_path.name
        if bool(settings.get("save_tiff", True)):
            tif_path = output_dir / f"{path.stem}.tif"
            save_tiff(tif_path, output, force_uint16=True)
            output_files["tiff"] = tif_path.name
        if bool(settings.get("save_png_preview", False)):
            png_path = output_dir / f"{path.stem}_preview.png"
            Image.fromarray(preview_uint8(output), mode="L").save(png_path)
            output_files["png_preview"] = png_path.name

        rows.append(
            {
                "index": index,
                "input_file": path.name,
                "input_shape_hw": json.dumps(list(raw.shape)),
                "input_dtype": str(raw.dtype),
                "background_file": str(background_path),
                "background_subtracted": True,
                "geometry_transform": "none",
                "resized": resized,
                "resize_mode": "area" if resized else "none",
                "output_shape_hw": json.dumps(list(output.shape)),
                "output_npy": output_files.get("npy", ""),
                "output_tiff": output_files.get("tiff", ""),
                "output_png_preview": output_files.get("png_preview", ""),
                "saturated_pixel_fraction": saturation,
                "input_min": float(raw.min()),
                "input_max": float(raw.max()),
                "output_min": float(output.min()),
                "output_max": float(output.max()),
                "output_mean": float(output.mean()),
            }
        )
        if len(previews) < 6:
            previews.append((path.name, raw, output))

    with (output_dir / "processing_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "background": str(background_path),
        "file_count": len(rows),
        "input_shape_hw": list(background.shape),
        "output_shape_hw": json.loads(rows[0]["output_shape_hw"]),
        "background_subtraction": True,
        "geometry_transform": "none",
        "per_image_normalization": False,
        "resize_enabled": bool(settings.get("resize_enabled", False)),
        "warnings": warning_messages,
        "timestamp": utc_now(),
    }
    json_dump(output_dir / "processing_summary.json", summary)

    figure, axes = plt.subplots(
        len(previews),
        2,
        figsize=(10, 4 * len(previews)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_axes, (name, before, after) in zip(axes, previews):
        row_axes[0].imshow(preview_uint8(before), cmap="gray")
        row_axes[0].set_title(f"Raw: {name}")
        row_axes[1].imshow(preview_uint8(after), cmap="gray")
        row_axes[1].set_title("Background subtracted" + (" + area resized" if summary["resize_enabled"] else ""))
        for axis in row_axes:
            axis.set_xlabel("x pixel")
            axis.set_ylabel("y pixel")
    figure.savefig(output_dir / "before_after_preview.png", dpi=160)
    plt.close(figure)
    print(f"Processed {len(rows)} raw frames into {output_dir}; geometry transform=none")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Subtract one ROI-matched background and optionally area-downsample"
    )
    parser.add_argument("--config", default="configs/calibration/tucam.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_batch_postprocess(args.config, args.input_dir, args.background, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
