"""Offline background subtraction, geometric correction, and final resizing."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any

import cv2
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


def _background_for_raw(
    background: np.ndarray, raw_shape: tuple[int, int], calibration: dict[str, Any]
) -> np.ndarray:
    if background.shape == raw_shape:
        return background
    roi = calibration.get("camera_hardware_roi_xywh")
    if roi is not None:
        left, top, width, height = [int(value) for value in roi]
        cropped = background[top : top + height, left : left + width]
        if cropped.shape == raw_shape:
            return cropped
    raise ValueError(
        f"background size {background.shape} does not match raw image {raw_shape}; "
        "the calibration hardware ROI cannot produce an exact crop."
    )


def camera_to_regular_square(
    corrected: np.ndarray,
    calibration: dict[str, Any],
    target_size_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Warp at near-camera density, then area-downsample to the target size."""
    matrix = np.asarray(calibration["forward_matrix"], dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("calibration forward_matrix must be a finite 3x3 matrix")
    roi = calibration.get("amplitude_roi") or calibration.get("config", {}).get("amplitude_roi")
    if not roi:
        raise ValueError("calibration.json does not contain amplitude_roi")
    width = float(roi["width"])
    height = float(roi["height"])
    left = float(roi["center_x"]) - width / 2
    top = float(roi["center_y"]) - height / 2
    corners = np.asarray(
        [[left, top], [left + width, top], [left + width, top + height], [left, top + height]],
        dtype=np.float64,
    )
    mapped = cv2.perspectiveTransform(corners[None], matrix)[0]
    edge_lengths = [
        np.linalg.norm(mapped[1] - mapped[0]), np.linalg.norm(mapped[2] - mapped[3]),
        np.linalg.norm(mapped[3] - mapped[0]), np.linalg.norm(mapped[2] - mapped[1]),
    ]
    # One square side at approximately the observed camera sampling density.
    intermediate_side = max(
        int(round(float(np.mean(edge_lengths)))), int(target_size_wh[0]), int(target_size_wh[1])
    )
    slm_to_regular = np.asarray(
        [
            [intermediate_side / width, 0.0, -left * intermediate_side / width],
            [0.0, intermediate_side / height, -top * intermediate_side / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_to_slm = np.linalg.inv(matrix)
    camera_to_regular_matrix = slm_to_regular @ camera_to_slm
    intermediate = cv2.warpPerspective(
        corrected.astype(np.float32), camera_to_regular_matrix,
        (intermediate_side, intermediate_side), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    validity = cv2.warpPerspective(
        np.ones(corrected.shape, dtype=np.uint8), camera_to_regular_matrix,
        (intermediate_side, intermediate_side), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    target_width, target_height = target_size_wh
    if intermediate_side < target_width or intermediate_side < target_height:
        raise RuntimeError("Intermediate warp unexpectedly undersamples the requested target")
    output = cv2.resize(
        intermediate, (target_width, target_height), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    return output, intermediate, {
        "intermediate_size_wh": [intermediate_side, intermediate_side],
        "camera_to_regular_matrix": camera_to_regular_matrix.tolist(),
        "mapped_roi_corners_xy": mapped.tolist(),
        "validity_fraction": float(np.mean(validity > 0)),
    }


def _saturation_fraction(raw: np.ndarray) -> float:
    if np.issubdtype(raw.dtype, np.integer):
        return float(np.mean(raw >= np.iinfo(raw.dtype).max))
    return 0.0


def run_batch_postprocess(
    config_path: str | Path,
    input_dir: str | Path,
    calibration_path: str | Path,
    background_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config, resolved_config = load_yaml_config(config_path)
    base = resolved_config.parent
    input_dir = resolve_path(input_dir, Path.cwd())
    calibration_path = resolve_path(calibration_path, Path.cwd())
    background_path = resolve_path(background_path, Path.cwd())
    output_dir = resolve_path(output_dir, Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    background = load_frame(background_path)
    settings = config["postprocess"]
    target_size = (int(settings["target_width"]), int(settings["target_height"]))
    if str(settings.get("resize_mode", "area")).lower() != "area":
        raise ValueError("postprocess.resize_mode must be area")
    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No supported raw CCD images found under {input_dir}")
    expected_wh = calibration.get("camera_hardware_roi_width_height")
    rows: list[dict[str, Any]] = []
    warning_messages: list[str] = []
    preview_items: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index, path in enumerate(files):
        raw = load_frame(path)
        if expected_wh is not None and list(raw.shape[::-1]) != [int(v) for v in expected_wh]:
            message = (
                f"{path.name}: input size {list(raw.shape[::-1])} differs from calibration "
                f"size {expected_wh}"
            )
            warning_messages.append(message); warnings.warn(message, RuntimeWarning)
        matched_background = _background_for_raw(background, raw.shape, calibration)
        corrected = corrected_frame(raw, matched_background)
        output, intermediate, transform = camera_to_regular_square(
            corrected, calibration, target_size
        )
        saturation = _saturation_fraction(raw)
        if saturation > float(settings.get("saturation_fraction_warning", 0.001)):
            message = f"{path.name}: saturated pixel fraction={saturation:.6f}"
            warning_messages.append(message); warnings.warn(message, RuntimeWarning)
        blank_fraction = 1.0 - float(transform["validity_fraction"])
        if blank_fraction > float(settings.get("blank_fraction_warning", 0.20)):
            message = f"{path.name}: geometric warp blank fraction={blank_fraction:.4f}"
            warning_messages.append(message); warnings.warn(message, RuntimeWarning)
        outputs: dict[str, str] = {}
        if bool(settings.get("save_npy", True)):
            npy_path = output_dir / f"{path.stem}.npy"
            np.save(npy_path, output.astype(np.float32)); outputs["npy"] = npy_path.name
        if bool(settings.get("save_tiff", True)):
            tif_path = output_dir / f"{path.stem}.tif"
            save_tiff(tif_path, output, force_uint16=True); outputs["tiff"] = tif_path.name
        if bool(settings.get("save_png_preview", False)):
            png_path = output_dir / f"{path.stem}_preview.png"
            Image.fromarray(preview_uint8(output), mode="L").save(png_path)
            outputs["png_preview"] = png_path.name
        rows.append(
            {
                "index": index, "input_file": path.name,
                "input_shape_hw": json.dumps(list(raw.shape)), "input_dtype": str(raw.dtype),
                "background_file": str(background_path),
                "calibration_model": calibration["model_type"],
                "intermediate_shape_hw": json.dumps(list(intermediate.shape)),
                "output_shape_hw": json.dumps(list(output.shape)),
                "output_npy": outputs.get("npy", ""), "output_tiff": outputs.get("tiff", ""),
                "output_png_preview": outputs.get("png_preview", ""),
                "saturated_pixel_fraction": saturation,
                "warp_blank_fraction": blank_fraction,
                "input_min": float(raw.min()), "input_max": float(raw.max()),
                "output_min": float(output.min()), "output_max": float(output.max()),
                "output_mean": float(output.mean()),
            }
        )
        if len(preview_items) < 6:
            preview_items.append((path.name, raw, output))
    with (output_dir / "processing_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "calibration": str(calibration_path), "background": str(background_path),
        "file_count": len(rows), "target_size_wh": list(target_size),
        "per_image_normalization": False, "warnings": warning_messages,
        "timestamp": utc_now(),
    }
    json_dump(output_dir / "processing_summary.json", summary)
    fig, axes = plt.subplots(
        len(preview_items), 2, figsize=(10, 4 * len(preview_items)), squeeze=False,
        constrained_layout=True,
    )
    for row_axes, (name, before, after) in zip(axes, preview_items):
        row_axes[0].imshow(preview_uint8(before), cmap="gray"); row_axes[0].set_title(f"Raw: {name}")
        row_axes[1].imshow(preview_uint8(after), cmap="gray"); row_axes[1].set_title("Corrected + warped + area resize")
        for axis in row_axes: axis.set_xlabel("x pixel"); axis.set_ylabel("y pixel")
    fig.savefig(output_dir / "before_after_preview.png", dpi=160); plt.close(fig)
    print(f"Processed {len(rows)} raw frames into {output_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-process raw hardware-ROI CCD frames")
    parser.add_argument("--config", default="configs/calibration/tucam.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_batch_postprocess(
        args.config, args.input_dir, args.calibration, args.background, args.output_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
