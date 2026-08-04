"""Create a coordinate overlay and fixed-size ROI recommendation from CCD frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path)
    else:
        value = np.asarray(Image.open(path).convert("L"))
    value = np.asarray(value, dtype=np.float64).squeeze()
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"CCD frame must be a finite 2-D array, got {value.shape}")
    return value


def recommend_roi(
    checkerboard: np.ndarray,
    reference: np.ndarray | None,
    expected_size_wh: tuple[int, int],
    threshold_percentile: float = 90.0,
) -> tuple[tuple[int, int, int, int], dict[str, float]]:
    if reference is not None and reference.shape != checkerboard.shape:
        raise ValueError(
            f"checkerboard/reference shape mismatch: {checkerboard.shape} vs {reference.shape}"
        )
    signal = (
        np.abs(checkerboard - reference)
        if reference is not None
        else checkerboard - checkerboard.min()
    )
    threshold = float(np.percentile(signal, threshold_percentile))
    weights = np.where(signal >= threshold, signal, 0.0)
    if float(weights.sum()) <= 0:
        raise RuntimeError("No checkerboard signal was found above the threshold")
    yy, xx = np.indices(weights.shape)
    center_x = float((weights * xx).sum() / weights.sum())
    center_y = float((weights * yy).sum() / weights.sum())
    width, height = expected_size_wh
    frame_height, frame_width = checkerboard.shape
    if width > frame_width or height > frame_height:
        raise ValueError(
            f"requested ROI {expected_size_wh} exceeds CCD frame {(frame_width, frame_height)}"
        )
    # Use explicit round-half-up instead of Python's round-to-even.  A stable
    # one-pixel convention matters when the calibrated center lies at x.5.
    x = min(
        max(int(np.floor(center_x - width / 2 + 0.5)), 0),
        frame_width - width,
    )
    y = min(
        max(int(np.floor(center_y - height / 2 + 0.5)), 0),
        frame_height - height,
    )
    return (x, y, width, height), {
        "signal_threshold": threshold,
        "signal_center_x": center_x,
        "signal_center_y": center_y,
        "signal_max": float(signal.max()),
        "signal_mean": float(signal.mean()),
    }


def save_overlay(frame: np.ndarray, roi: tuple[int, int, int, int], output: Path) -> None:
    low, high = np.percentile(frame, [1.0, 99.8])
    normalized = np.clip((frame - low) / max(float(high - low), 1.0e-12), 0.0, 1.0)
    image = Image.fromarray(np.rint(normalized * 255).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, width, height = roi
    draw.rectangle((x, y, x + width - 1, y + height - 1), outline=(255, 0, 0), width=3)
    step = max(50, min(image.size) // 10)
    for tick in range(0, image.width, step):
        draw.line((tick, 0, tick, min(12, image.height)), fill=(0, 255, 0), width=1)
        draw.text((tick + 2, 2), str(tick), fill=(0, 255, 0))
    for tick in range(0, image.height, step):
        draw.line((0, tick, min(12, image.width), tick), fill=(0, 255, 0), width=1)
        draw.text((2, tick + 2), str(tick), fill=(0, 255, 0))
    draw.text((x + 5, y + 5), f"ROI x={x}, y={y}, w={width}, h={height}", fill=(255, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend and visualize the CCD active ROI")
    parser.add_argument("--checkerboard", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None, help="Black-field capture with identical exposure")
    parser.add_argument("--expected-width", type=int, required=True)
    parser.add_argument("--expected-height", type=int, required=True)
    parser.add_argument("--threshold-percentile", type=float, default=90.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkerboard = load_frame(args.checkerboard)
    reference = load_frame(args.reference) if args.reference is not None else None
    roi, stats = recommend_roi(
        checkerboard,
        reference,
        (args.expected_width, args.expected_height),
        args.threshold_percentile,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_overlay(checkerboard, roi, args.output_dir / "ccd_roi_overlay.png")
    report = {
        "checkerboard": str(args.checkerboard.resolve()),
        "reference": None if args.reference is None else str(args.reference.resolve()),
        "frame_shape_hw": list(checkerboard.shape),
        "recommended_roi_xywh": list(roi),
        "threshold_percentile": args.threshold_percentile,
        **stats,
    }
    (args.output_dir / "roi_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Recommended ROI [x,y,w,h] = {list(roi)}")
    print(f"Overlay: {args.output_dir / 'ccd_roi_overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
