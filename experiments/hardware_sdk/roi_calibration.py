"""ROI, system-background, and amplitude-response calibration.

This module deliberately reuses ``devices.build_slm`` and
``devices.build_camera``.  It does not contain a second hardware driver stack.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from .calibration_common import (
        corrected_frame,
        json_dump,
        load_frame,
        load_yaml_config,
        median_capture,
        preview_uint8,
        resolve_path,
        save_preview,
        save_tiff,
        utc_now,
    )
    from .devices import build_camera, build_slm
    from .phase_slm_demo import BlinkPhaseSLM, prepare_phase_frame
except ImportError:  # direct execution inside hardware_sdk/
    from calibration_common import (
        corrected_frame,
        json_dump,
        load_frame,
        load_yaml_config,
        median_capture,
        preview_uint8,
        resolve_path,
        save_preview,
        save_tiff,
        utc_now,
    )
    from devices import build_camera, build_slm
    from phase_slm_demo import BlinkPhaseSLM, prepare_phase_frame


def _paths(config: dict[str, Any], config_path: Path) -> tuple[Path, Path]:
    base = config_path.parent
    paths = config.get("paths", {})
    masks = resolve_path(paths.get("masks_dir", "../calibration_masks"), base)
    results = resolve_path(paths.get("results_dir", "../calibration_results"), base)
    return masks, results


def _amplitude_roi(config: dict[str, Any]) -> tuple[float, float, float, float]:
    roi = config["amplitude_roi"]
    width, height = float(roi["width"]), float(roi["height"])
    center_x, center_y = float(roi["center_x"]), float(roi["center_y"])
    return center_x - width / 2, center_y - height / 2, width, height


def calibration_source_points(
    config: dict[str, Any], grid_size: int
) -> list[tuple[float, float]]:
    """Return source points in full amplitude-SLM pixel coordinates."""
    left, top, width, height = _amplitude_roi(config)
    margin = float(config["calibration"]["marker_margin_px"])
    xs = np.linspace(left + margin, left + width - margin, grid_size)
    ys = np.linspace(top + margin, top + height - margin, grid_size)
    if grid_size == 2:
        return [
            (left + width / 2, top + height / 2),
            (xs[0], ys[0]),
            (xs[-1], ys[0]),
            (xs[0], ys[-1]),
            (xs[-1], ys[-1]),
        ]
    return [(float(x), float(y)) for y in ys for x in xs]


def gaussian_marker(
    size_wh: tuple[int, int],
    center_xy: tuple[float, float],
    marker_size: int,
    sigma: float,
    peak: int = 255,
) -> np.ndarray:
    width, height = (int(size_wh[0]), int(size_wh[1]))
    if marker_size <= 0 or sigma <= 0:
        raise ValueError("marker_size and sigma must be positive")
    value = np.zeros((height, width), dtype=np.uint8)
    half = marker_size / 2
    x0 = max(0, int(math.floor(center_xy[0] - half)))
    x1 = min(width, int(math.ceil(center_xy[0] + half)))
    y0 = max(0, int(math.floor(center_xy[1] - half)))
    y1 = min(height, int(math.ceil(center_xy[1] + half)))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    spot = peak * np.exp(
        -((xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2) / (2 * sigma**2)
    )
    value[y0:y1, x0:x1] = np.rint(np.clip(spot, 0, 255)).astype(np.uint8)
    return value


def exposure_patch(
    size_wh: tuple[int, int],
    center_xy: tuple[float, float],
    gray: int,
    outer_size: int,
    inner_size: int,
    edge_taper: int,
) -> np.ndarray:
    if not 0 <= gray <= 255:
        raise ValueError("gray must be in [0,255]")
    if inner_size + 2 * edge_taper != outer_size:
        raise ValueError("patch_inner_size_px + 2*edge_taper_px must equal patch_size_px")
    coordinate = np.arange(outer_size, dtype=np.float32)
    distance_to_edge = np.minimum(coordinate + 0.5, outer_size - coordinate - 0.5)
    ramp = np.ones(outer_size, dtype=np.float32)
    tapered = distance_to_edge < edge_taper
    ramp[tapered] = 0.5 - 0.5 * np.cos(
        np.pi * distance_to_edge[tapered] / float(edge_taper)
    )
    patch = float(gray) * np.outer(ramp, ramp)
    width, height = size_wh
    output = np.zeros((height, width), dtype=np.uint8)
    x0 = int(round(center_xy[0] - outer_size / 2))
    y0 = int(round(center_xy[1] - outer_size / 2))
    output[y0 : y0 + outer_size, x0 : x0 + outer_size] = np.rint(patch).astype(np.uint8)
    return output


def generate_calibration_files(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    masks, _ = _paths(config, config_path)
    amplitude_dir = masks / "amplitude"
    phase_dir = masks / "phase"
    exposure_dir = masks / "exposure"
    for directory in (amplitude_dir, phase_dir, exposure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    amp_size = (int(config["amplitude_slm"]["width"]), int(config["amplitude_slm"]["height"]))
    phase_size = (int(config["phase_slm"]["width"]), int(config["phase_slm"]["height"]))
    Image.new("L", amp_size, 0).save(amplitude_dir / "amplitude_zero.bmp")
    Image.new("L", phase_size, 0).save(phase_dir / "phase_zero.bmp")
    marker_size = int(config["calibration"]["marker_size_px"])
    sigma = float(config["calibration"]["marker_sigma_px"])
    phase_name = "phase/phase_zero.bmp"
    rows: list[dict[str, Any]] = []

    coarse_points = calibration_source_points(config, 2)
    combined = np.zeros((amp_size[1], amp_size[0]), dtype=np.uint8)
    for index, point in enumerate(coarse_points):
        marker = gaussian_marker(amp_size, point, marker_size, sigma)
        combined = np.maximum(combined, marker)
        name = f"coarse_{index + 1:02d}.bmp"
        Image.fromarray(marker, mode="L").save(amplitude_dir / name)
        rows.append(
            dict(
                pattern_id=f"coarse_{index + 1:02d}", calibration_type="coarse",
                source_x=point[0], source_y=point[1], gray_value=255,
                amplitude_filename=f"amplitude/{name}", phase_filename=phase_name,
            )
        )
    Image.fromarray(combined, mode="L").save(amplitude_dir / "verify_5points.bmp")

    fine_points = calibration_source_points(config, 3)
    for index, point in enumerate(fine_points):
        marker = gaussian_marker(amp_size, point, marker_size, sigma)
        name = f"fine_{index + 1:02d}.bmp"
        Image.fromarray(marker, mode="L").save(amplitude_dir / name)
        rows.append(
            dict(
                pattern_id=f"fine_{index + 1:02d}", calibration_type="fine",
                source_x=point[0], source_y=point[1], gray_value=255,
                amplitude_filename=f"amplitude/{name}", phase_filename=phase_name,
            )
        )

    exposure = config["exposure_calibration"]
    center = (float(config["amplitude_roi"]["center_x"]), float(config["amplitude_roi"]["center_y"]))
    gray_values = list(
        range(int(exposure["gray_start"]), int(exposure["gray_stop"]) + 1, int(exposure["gray_step"]))
    )
    for gray in gray_values:
        pattern = exposure_patch(
            amp_size, center, gray, int(exposure["patch_size_px"]),
            int(exposure["patch_inner_size_px"]), int(exposure["edge_taper_px"]),
        )
        name = f"gray_{gray:03d}.bmp"
        Image.fromarray(pattern, mode="L").save(exposure_dir / name)
        rows.append(
            dict(
                pattern_id=f"exposure_{gray:03d}", calibration_type="exposure",
                source_x=center[0], source_y=center[1], gray_value=gray,
                amplitude_filename=f"exposure/{name}", phase_filename=phase_name,
            )
        )
    with (masks / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "masks_dir": str(masks), "coarse_patterns": len(coarse_points),
        "fine_patterns": len(fine_points), "exposure_patterns": len(gray_values),
        "amplitude_size_wh": list(amp_size), "phase_size_wh": list(phase_size),
    }
    print(f"Generated calibration BMPs and manifest under {masks}")
    return report


class _PhaseController:
    def __init__(self, config: dict[str, Any], config_path: Path) -> None:
        self.config = config["phase_slm"]
        self.base = config_path.parent
        self.driver = str(self.config.get("driver", "manual")).lower()
        self._context: BlinkPhaseSLM | None = None

    def open(self) -> None:
        if self.driver == "manual":
            return
        if self.driver != "meadowlark":
            raise ValueError("phase_slm.driver must be manual or meadowlark")
        self._context = BlinkPhaseSLM(
            resolve_path(self.config["sdk_root"], self.base),
            resolve_path(self.config["lut_file"], self.base),
        )
        self._context.__enter__()

    def display(self, path: Path) -> None:
        if self.driver == "manual":
            print(f"[phase SLM/manual] 请加载并保持: {path}")
            return
        expected = (int(self.config["width"]), int(self.config["height"]))
        correction = (
            resolve_path(self.config["wavefront_correction_file"], self.base)
            if bool(self.config.get("apply_wavefront_correction", False)) else None
        )
        frame = prepare_phase_frame(
            path, expected, wavefront_correction=correction,
            flip_vertical=bool(self.config.get("flip_vertical", False)),
        )
        assert self._context is not None
        self._context.show(frame)
        print(f"[phase SLM/meadowlark] displayed {path.name}")

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None


@contextmanager
def _slm_session(
    config: dict[str, Any], config_path: Path, amplitude_files: list[Path]
) -> Iterator[tuple[Any, _PhaseController]]:
    amplitude_config = dict(config["amplitude_slm"])
    amplitude_config.setdefault(
        "expected_resolution_wh",
        [amplitude_config["width"], amplitude_config["height"]],
    )
    amplitude = build_slm(amplitude_config, config_path.parent)
    phase = _PhaseController(config, config_path)
    amplitude.validate_runtime()
    try:
        amplitude.open()
        amplitude.preload_files(amplitude_files)
        phase.open()
        yield amplitude, phase
    finally:
        # No blank/zero replacement is sent here.  The last displayed pattern
        # remains the requested optical state while SDK resources are released.
        phase.close()
        amplitude.close()


def _camera_config(config: dict[str, Any], exposure_us: float | None = None) -> dict[str, Any]:
    value = dict(config["camera"])
    value["saved_frame_size_wh"] = None
    value["saved_frame_resize_mode"] = "none"
    if exposure_us is not None:
        value["auto_exposure"] = False
        value["exposure_us"] = float(exposure_us)
    return value


def _operator_confirm(message: str, assume_yes: bool) -> None:
    print(message)
    if not assume_yes:
        input("按 Enter 继续；如状态不正确，请先中止并修正设备... ")


def acquire_background(
    config: dict[str, Any], config_path: Path, *, assume_yes: bool = False
) -> dict[str, Any]:
    masks, results = _paths(config, config_path)
    amplitude_zero = masks / "amplitude" / "amplitude_zero.bmp"
    phase_zero = masks / "phase" / "phase_zero.bmp"
    if not amplitude_zero.is_file() or not phase_zero.is_file():
        generate_calibration_files(config, config_path)
    results.mkdir(parents=True, exist_ok=True)
    frames = int(config["background"]["frames"])
    with _slm_session(config, config_path, [amplitude_zero]) as (amplitude, phase):
        phase.display(phase_zero)
        amplitude.display_file(amplitude_zero)
        _operator_confirm(
            "\n背景采集状态：\n"
            "1. 请保持激光开启。\n"
            "2. 相位 SLM 应为 phase_zero.bmp。\n"
            "3. 振幅 SLM 应为 amplitude_zero.bmp。\n"
            f"4. 将采集 {frames} 帧并取逐像素中位数。",
            assume_yes,
        )
        camera = build_camera(_camera_config(config), config_path.parent)
        camera.validate_runtime()
        with camera:
            background, frame_metadata = median_capture(camera, results, "background", frames)
            camera_info = camera.device_info()
    np.save(results / "background.npy", background.astype(np.float32))
    save_tiff(results / "background.tif", background.astype(np.float32))
    save_preview(results / "background_preview.png", background)
    metadata = {
        "background_type": "laser_on_both_slms_zero_pattern",
        "is_camera_dark_frame": False,
        "frames": frames,
        "shape_hw": list(background.shape),
        "dtype": "float32_median",
        "source_frame_metadata": frame_metadata,
        "camera": camera_info,
        "amplitude_pattern": str(amplitude_zero),
        "phase_pattern": str(phase_zero),
        "timestamp": utc_now(),
    }
    json_dump(results / "background_metadata.json", metadata)
    print(f"Background saved under {results}; the last SLM images were not replaced.")
    return metadata


def _background_for_frame(background: np.ndarray, frame_shape: tuple[int, int], results: Path) -> np.ndarray:
    if background.shape == frame_shape:
        return background
    coarse_path = results / "coarse_calibration.json"
    if coarse_path.is_file():
        import json

        coarse = json.loads(coarse_path.read_text(encoding="utf-8"))
        left, top, width, height = coarse["recommended_hardware_roi_xywh"]
        cropped = background[top : top + height, left : left + width]
        if cropped.shape == frame_shape:
            return cropped
    raise ValueError(
        f"Background shape {background.shape} does not match camera frame {frame_shape}. "
        "If hardware ROI was changed after background capture, rerun background with the "
        "hardware ROI active or ensure coarse_calibration.json describes that exact crop."
    )


def detect_marker_centroid(
    corrected: np.ndarray,
    detection_config: dict[str, Any],
    expected_xy: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], dict[str, Any]]:
    value = np.asarray(corrected, dtype=np.float32)
    peak = float(value.max())
    if peak <= 0:
        raise RuntimeError("No positive marker signal remains after background subtraction")
    nonzero = value[value > 0]
    percentile = float(np.percentile(nonzero, float(detection_config["threshold_percentile"])))
    threshold = max(
        peak * float(detection_config["threshold_fraction_of_peak"]), percentile
    )
    binary = (value >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = int(detection_config["minimum_component_area_px"])
    max_area = int(value.size * float(detection_config["maximum_component_area_fraction"]))
    candidates: list[dict[str, Any]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        component = labels == label
        # Include dimmer Gaussian shoulders around the connected core.
        component = cv2.dilate(component.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        weights = np.where(component, value, 0.0)
        energy = float(weights.sum())
        if energy <= 0:
            continue
        yy, xx = np.indices(value.shape)
        cx = float((weights * xx).sum() / energy)
        cy = float((weights * yy).sum() / energy)
        distance = (
            0.0 if expected_xy is None
            else float(math.hypot(cx - expected_xy[0], cy - expected_xy[1]))
        )
        candidates.append(
            {"x": cx, "y": cy, "area": area, "energy": energy, "distance": distance}
        )
    if not candidates:
        raise RuntimeError(
            f"No reasonable connected marker found (threshold={threshold:.6g}, peak={peak:.6g})"
        )
    if expected_xy is None:
        chosen = max(candidates, key=lambda item: item["energy"])
    else:
        energy_max = max(item["energy"] for item in candidates)
        plausible = [item for item in candidates if item["energy"] >= 0.05 * energy_max]
        chosen = min(plausible, key=lambda item: item["distance"])
    return (chosen["x"], chosen["y"]), {
        "peak": peak, "threshold": threshold, "chosen": chosen, "candidates": candidates,
    }


def recommend_roi(
    checkerboard: np.ndarray,
    reference: np.ndarray | None,
    expected_size_wh: tuple[int, int],
    threshold_percentile: float = 90.0,
) -> tuple[tuple[int, int, int, int], dict[str, float]]:
    """Backward-compatible simple ROI helper used by the earlier checker demo."""
    if reference is not None and reference.shape != checkerboard.shape:
        raise ValueError(
            f"checkerboard/reference shape mismatch: {checkerboard.shape} vs {reference.shape}"
        )
    signal = (
        np.abs(checkerboard.astype(np.float64) - reference.astype(np.float64))
        if reference is not None else checkerboard.astype(np.float64) - float(checkerboard.min())
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
        raise ValueError("requested ROI exceeds CCD frame")
    x = min(max(int(np.floor(center_x - width / 2 + 0.5)), 0), frame_width - width)
    y = min(max(int(np.floor(center_y - height / 2 + 0.5)), 0), frame_height - height)
    return (x, y, width, height), {
        "signal_threshold": threshold, "signal_center_x": center_x,
        "signal_center_y": center_y, "signal_max": float(signal.max()),
        "signal_mean": float(signal.mean()),
    }


def fit_affine(source_xy: np.ndarray, destination_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    design = np.column_stack([source_xy, np.ones(len(source_xy))])
    coefficients, _, _, _ = np.linalg.lstsq(design, destination_xy, rcond=None)
    matrix = np.vstack([coefficients.T, [0.0, 0.0, 1.0]])
    projected = cv2.perspectiveTransform(source_xy[None].astype(np.float64), matrix)[0]
    residuals = np.linalg.norm(projected - destination_xy, axis=1)
    return matrix, residuals, float(np.sqrt(np.mean(residuals**2)))


def fit_homography(source_xy: np.ndarray, destination_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    matrix, _ = cv2.findHomography(source_xy.astype(np.float64), destination_xy.astype(np.float64), 0)
    if matrix is None or not np.isfinite(matrix).all():
        raise RuntimeError("Homography fit failed")
    projected = cv2.perspectiveTransform(source_xy[None].astype(np.float64), matrix)[0]
    residuals = np.linalg.norm(projected - destination_xy, axis=1)
    return matrix, residuals, float(np.sqrt(np.mean(residuals**2)))


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points[None].astype(np.float64), matrix)[0]


def _roi_corners(config: dict[str, Any]) -> np.ndarray:
    left, top, width, height = _amplitude_roi(config)
    return np.asarray(
        [[left, top], [left + width, top], [left + width, top + height], [left, top + height]],
        dtype=np.float64,
    )


def _aligned_hardware_roi(
    corners: np.ndarray,
    frame_shape: tuple[int, int],
    padding_ratio: float,
    constraints: dict[str, Any],
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape
    min_x, min_y = corners.min(axis=0)
    max_x, max_y = corners.max(axis=0)
    pad_x = (max_x - min_x) * padding_ratio
    pad_y = (max_y - min_y) * padding_ratio
    x_step = int(constraints.get("offset_step_x", 1))
    y_step = int(constraints.get("offset_step_y", 1))
    w_step = int(constraints.get("width_step", 1))
    h_step = int(constraints.get("height_step", 1))
    left = max(0, int(math.floor((min_x - pad_x) / x_step) * x_step))
    top = max(0, int(math.floor((min_y - pad_y) / y_step) * y_step))
    right = min(frame_w, int(math.ceil((max_x + pad_x) / w_step) * w_step))
    bottom = min(frame_h, int(math.ceil((max_y + pad_y) / h_step) * h_step))
    width = max(int(constraints.get("min_width", 1)), right - left)
    height = max(int(constraints.get("min_height", 1)), bottom - top)
    width = int(math.ceil(width / w_step) * w_step)
    height = int(math.ceil(height / h_step) * h_step)
    if left + width > frame_w:
        left = max(0, (frame_w - width) // x_step * x_step)
    if top + height > frame_h:
        top = max(0, (frame_h - height) // y_step * y_step)
    return left, top, min(width, frame_w - left), min(height, frame_h - top)


def _capture_patterns(
    config: dict[str, Any], config_path: Path, pattern_paths: list[Path],
    background: np.ndarray, results: Path, expected_points: list[tuple[float, float]] | None = None,
    nominal_source_points: list[tuple[float, float]] | None = None,
) -> tuple[list[np.ndarray], list[tuple[float, float]], list[dict[str, Any]], dict[str, Any]]:
    frames_per = int(config["calibration"]["frames_per_pattern"])
    settle = float(config.get("settle_delay_ms", 40)) / 1000
    detected: list[tuple[float, float]] = []
    frames: list[np.ndarray] = []
    detections: list[dict[str, Any]] = []
    with _slm_session(config, config_path, pattern_paths) as (amplitude, phase):
        phase.display(_paths(config, config_path)[0] / "phase" / "phase_zero.bmp")
        camera = build_camera(_camera_config(config), config_path.parent)
        camera.validate_runtime()
        with camera:
            for index, pattern in enumerate(pattern_paths):
                amplitude.display_file(pattern)
                time.sleep(settle)
                raw, _ = median_capture(camera, results, pattern.stem, frames_per)
                matched_background = _background_for_frame(background, raw.shape, results)
                corrected = corrected_frame(raw, matched_background)
                expected = None if expected_points is None else expected_points[index]
                if nominal_source_points is not None:
                    if index == 0:
                        expected = (corrected.shape[1] / 2, corrected.shape[0] / 2)
                    elif detected:
                        scale = (
                            float(config["amplitude_slm"]["pixel_pitch_um"])
                            * float(config.get("relay_nominal_magnification", 1.0))
                            / float(config["camera"]["pixel_pitch_um"])
                        )
                        center_source = nominal_source_points[0]
                        expected = (
                            detected[0][0] + (nominal_source_points[index][0] - center_source[0]) * scale,
                            detected[0][1] + (nominal_source_points[index][1] - center_source[1]) * scale,
                        )
                point, details = detect_marker_centroid(
                    corrected, config["calibration"]["detection"], expected
                )
                frames.append(corrected)
                detected.append(point)
                detections.append(details)
                print(f"[{pattern.stem}] centroid=({point[0]:.3f},{point[1]:.3f})")
            camera_info = camera.device_info()
    return frames, detected, detections, camera_info


def _save_calibration_overlay(
    output: Path, frame: np.ndarray, points: np.ndarray, polygon: np.ndarray | None,
    rectangle: tuple[int, int, int, int] | None, title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.imshow(preview_uint8(frame), cmap="gray", origin="upper")
    ax.scatter(points[:, 0], points[:, 1], c="lime", marker="x", s=80, label="detected centroids")
    for index, point in enumerate(points):
        ax.text(point[0] + 4, point[1] + 4, str(index + 1), color="yellow")
    if polygon is not None:
        closed = np.vstack([polygon, polygon[0]])
        ax.plot(closed[:, 0], closed[:, 1], "c-", linewidth=2, label="mapped optical ROI")
    if rectangle is not None:
        left, top, width, height = rectangle
        patch = plt.Rectangle((left, top), width, height, fill=False, edgecolor="red", linewidth=2,
                              label=f"hardware ROI left={left} top={top} width={width} height={height}")
        ax.add_patch(patch)
    ax.set_xlabel("CCD x pixel")
    ax.set_ylabel("CCD y pixel")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_coarse(config: dict[str, Any], config_path: Path, *, assume_yes: bool = False) -> dict[str, Any]:
    masks, results = _paths(config, config_path)
    if not (masks / "manifest.csv").is_file():
        generate_calibration_files(config, config_path)
    if not (results / "background.npy").is_file():
        print("No unified background found; acquiring it first.")
        acquire_background(config, config_path, assume_yes=assume_yes)
    _operator_confirm(
        "粗标定要求：相机当前必须为完整视野；激光开启；相位 SLM 使用 phase_zero.bmp。",
        assume_yes,
    )
    background = load_frame(results / "background.npy")
    patterns = [masks / "amplitude" / f"coarse_{index:02d}.bmp" for index in range(1, 6)]
    source = np.asarray(calibration_source_points(config, 2), dtype=np.float64)
    frames, detected, detections, camera_info = _capture_patterns(
        config, config_path, patterns, background, results,
        nominal_source_points=[tuple(point) for point in source],
    )
    destination = np.asarray(detected, dtype=np.float64)
    matrix, residuals, rms = fit_affine(source, destination)
    mapped_corners = _transform_points(matrix, _roi_corners(config))
    rectangle = _aligned_hardware_roi(
        mapped_corners, frames[0].shape,
        float(config["calibration"]["camera_roi_padding_ratio"]),
        config["camera"].get("roi_constraints", {}),
    )
    report = {
        "model_type": "affine", "forward_matrix": matrix.tolist(),
        "source_points_xy": source.tolist(), "ccd_points_xy": destination.tolist(),
        "point_residual_px": residuals.tolist(), "rms_residual_px": rms,
        "mapped_amplitude_roi_corners_xy": mapped_corners.tolist(),
        "recommended_hardware_roi_xywh": list(rectangle),
        "full_frame_shape_hw": list(frames[0].shape), "detections": detections,
        "camera": camera_info, "timestamp": utc_now(),
    }
    json_dump(results / "coarse_calibration.json", report)
    _save_calibration_overlay(
        results / "coarse_overlay.png", np.maximum.reduce(frames), destination,
        mapped_corners, rectangle, "Coarse ROI calibration",
    )
    # Leave the human verification image as the final amplitude pattern.
    verify = masks / "amplitude" / "verify_5points.bmp"
    with _slm_session(config, config_path, [verify]) as (amplitude, phase):
        phase.display(masks / "phase" / "phase_zero.bmp")
        amplitude.display_file(verify)
        print("verify_5points.bmp is displayed; no replacement image will be sent.")
    left, top, width, height = rectangle
    print(f"left={left}\ntop={top}\nwidth={width}\nheight={height}")
    print("请在相机厂商软件中输入以上硬件 ROI，并观察保持显示的五点图案。")
    return report


def _coarse_expected_points(config: dict[str, Any], results: Path, source: np.ndarray) -> list[tuple[float, float]] | None:
    path = results / "coarse_calibration.json"
    if not path.is_file():
        return None
    import json

    coarse = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(coarse["forward_matrix"], dtype=np.float64)
    predicted_full = _transform_points(matrix, source)
    roi = coarse.get("recommended_hardware_roi_xywh")
    if roi is not None:
        predicted_full[:, 0] -= float(roi[0])
        predicted_full[:, 1] -= float(roi[1])
    return [tuple(point) for point in predicted_full]


def run_fine(config: dict[str, Any], config_path: Path, *, assume_yes: bool = False) -> dict[str, Any]:
    masks, results = _paths(config, config_path)
    if not (results / "background.npy").is_file():
        acquire_background(config, config_path, assume_yes=assume_yes)
    _operator_confirm(
        "精标定要求：请先在厂商软件中设置并保持粗标定给出的硬件 ROI。",
        assume_yes,
    )
    background = load_frame(results / "background.npy")
    patterns = [masks / "amplitude" / f"fine_{index:02d}.bmp" for index in range(1, 10)]
    source = np.asarray(calibration_source_points(config, 3), dtype=np.float64)
    expected = _coarse_expected_points(config, results, source)
    frames, detected, detections, camera_info = _capture_patterns(
        config, config_path, patterns, background, results, expected
    )
    destination = np.asarray(detected, dtype=np.float64)
    affine, affine_residuals, affine_rms = fit_affine(source, destination)
    homography, homography_residuals, homography_rms = fit_homography(source, destination)
    improvement = (affine_rms - homography_rms) / max(affine_rms, 1e-12)
    required = float(config["calibration"].get("homography_min_rms_improvement", 0.20))
    if improvement >= required:
        model, matrix, residuals, rms = "homography", homography, homography_residuals, homography_rms
    else:
        model, matrix, residuals, rms = "affine", affine, affine_residuals, affine_rms
    hardware_roi = camera_info.get("device_roi_xywh")
    if hardware_roi is None and (results / "coarse_calibration.json").is_file():
        import json

        hardware_roi = json.loads(
            (results / "coarse_calibration.json").read_text(encoding="utf-8")
        ).get("recommended_hardware_roi_xywh")
    report = {
        "camera_hardware_roi_xywh": hardware_roi,
        "camera_hardware_roi_width_height": [frames[0].shape[1], frames[0].shape[0]],
        "model_type": model, "forward_matrix": matrix.tolist(),
        "inverse_matrix": np.linalg.inv(matrix).tolist(),
        "source_points_xy": source.tolist(), "ccd_points_xy": destination.tolist(),
        "point_residual_px": residuals.tolist(), "rms_residual_px": rms,
        "max_residual_px": float(residuals.max()),
        "affine": {"matrix": affine.tolist(), "rms_residual_px": affine_rms,
                   "point_residual_px": affine_residuals.tolist()},
        "homography": {"matrix": homography.tolist(), "rms_residual_px": homography_rms,
                       "point_residual_px": homography_residuals.tolist(),
                       "relative_rms_improvement": improvement},
        "detections": detections, "camera": camera_info,
        "amplitude_roi": config["amplitude_roi"],
        "camera_pixel_pitch_um": config["camera"]["pixel_pitch_um"],
        "amplitude_slm_pixel_pitch_um": config["amplitude_slm"]["pixel_pitch_um"],
        "relay_nominal_magnification": config["relay_nominal_magnification"],
        "timestamp": utc_now(), "config": config,
    }
    json_dump(results / "calibration.json", report)
    combined = np.maximum.reduce(frames)
    mapped_corners = _transform_points(matrix, _roi_corners(config))
    _save_calibration_overlay(
        results / "fine_overlay.png", combined, destination, mapped_corners, None,
        f"Fine calibration ({model}, RMS={rms:.3f}px)",
    )
    projected = _transform_points(matrix, source)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(preview_uint8(combined), cmap="gray")
    ax.quiver(projected[:, 0], projected[:, 1], destination[:, 0] - projected[:, 0],
              destination[:, 1] - projected[:, 1], angles="xy", scale_units="xy", scale=1,
              color="red", width=0.004)
    ax.scatter(destination[:, 0], destination[:, 1], c="lime", s=25)
    ax.set_title(f"Residual vectors; RMS={rms:.3f}px, max={residuals.max():.3f}px")
    ax.set_xlabel("CCD x pixel"); ax.set_ylabel("CCD y pixel")
    fig.savefig(results / "residual_vectors.png", dpi=180); plt.close(fig)
    if rms > float(config["calibration"].get("large_residual_px", 8.0)):
        warnings.warn(
            "Both simple geometric models have a large residual. Check the upstream 4f relay, "
            "final relay, CCD tilt, and whether another diffraction order was detected. No "
            "higher-order model was fitted.", RuntimeWarning,
        )
    print(f"Fine calibration selected {model}: RMS={rms:.4f}px, max={residuals.max():.4f}px")
    return report


def _measurement_window(
    frame: np.ndarray, config: dict[str, Any]
) -> tuple[int, int, int, int]:
    detection = config["calibration"]["detection"]
    point, details = detect_marker_centroid(frame, detection)
    chosen = details["chosen"]
    side = int(max(math.sqrt(chosen["area"]) * 3.0, 24))
    side = int(round(side * (1 + float(config["exposure_calibration"].get("measurement_window_padding_ratio", 0.1)))))
    x = max(0, min(frame.shape[1] - side, int(round(point[0] - side / 2))))
    y = max(0, min(frame.shape[0] - side, int(round(point[1] - side / 2))))
    return x, y, min(side, frame.shape[1]), min(side, frame.shape[0])


def run_exposure(config: dict[str, Any], config_path: Path, *, assume_yes: bool = False) -> dict[str, Any]:
    masks, results = _paths(config, config_path)
    if not (masks / "manifest.csv").is_file():
        generate_calibration_files(config, config_path)
    if not (results / "background.npy").is_file():
        acquire_background(config, config_path, assume_yes=assume_yes)
    background = load_frame(results / "background.npy")
    exposure_cfg = config["exposure_calibration"]
    gray_values = list(range(int(exposure_cfg["gray_start"]), int(exposure_cfg["gray_stop"]) + 1,
                             int(exposure_cfg["gray_step"])))
    exposure_times = [float(value) for value in exposure_cfg["exposure_times_us"]]
    patterns = [masks / "exposure" / f"gray_{gray:03d}.bmp" for gray in gray_values]
    phase_zero = masks / "phase" / "phase_zero.bmp"
    frames_per = int(exposure_cfg["frames_per_gray"])
    settle = float(config.get("settle_delay_ms", 40)) / 1000
    rows: list[dict[str, Any]] = []
    preview_frames: dict[tuple[float, int], np.ndarray] = {}
    windows: dict[float, tuple[int, int, int, int]] = {}
    sensor_limits: dict[float, float] = {}
    for exposure_us in exposure_times:
        # Determine the fixed window once from gray=255 (or the highest configured gray).
        with _slm_session(config, config_path, patterns) as (amplitude, phase):
            phase.display(phase_zero)
            camera = build_camera(_camera_config(config, exposure_us), config_path.parent)
            camera.validate_runtime()
            with camera:
                highest_path = patterns[-1]
                amplitude.display_file(highest_path); time.sleep(settle)
                raw_high, _ = median_capture(camera, results, f"response_{int(exposure_us)}_window", frames_per)
                bg = _background_for_frame(background, raw_high.shape, results)
                corrected_high = corrected_frame(raw_high, bg)
                window = _measurement_window(corrected_high, config)
                windows[exposure_us] = window
                for gray, pattern in zip(gray_values, patterns):
                    amplitude.display_file(pattern); time.sleep(settle)
                    raw, _ = median_capture(camera, results, f"response_{int(exposure_us)}_{gray:03d}", frames_per)
                    corrected = corrected_frame(raw, bg)
                    x, y, width, height = window
                    region = corrected[y : y + height, x : x + width]
                    raw_region = raw[y : y + height, x : x + width]
                    integer_limit = np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype, np.integer) else float(raw.max())
                    sensor_limits[exposure_us] = float(integer_limit)
                    rows.append(
                        {
                            "gray_value": gray, "exposure_us": exposure_us,
                            "mean_intensity": float(region.mean()),
                            "integrated_energy": float(region.sum()),
                            "p99": float(np.percentile(region, 99)),
                            "max_intensity": float(region.max()),
                            "saturated_pixel_fraction": float(np.mean(raw_region >= integer_limit)),
                        }
                    )
                    if gray in {0, 64, 128, 192, 255}:
                        preview_frames[(exposure_us, gray)] = corrected
    with (results / "slm_response.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    messages: list[str] = []
    for exposure_us in exposure_times:
        subset = [row for row in rows if row["exposure_us"] == exposure_us]
        if max(row["saturated_pixel_fraction"] for row in subset) > float(exposure_cfg["saturation_fraction_warning"]):
            messages.append(f"{exposure_us:g} us: 明显饱和，建议降低曝光。")
        response = subset[-1]["integrated_energy"] - subset[0]["integrated_energy"]
        if response <= 0 or subset[-1]["max_intensity"] < float(exposure_cfg["low_response_fraction_of_sensor_range"]) * sensor_limits[exposure_us]:
            messages.append(f"{exposure_us:g} us: gray=255 响应较低，请人工判断是否提高曝光。")
        differences = np.diff([row["integrated_energy"] for row in subset])
        if np.mean(differences < 0) > 0.15:
            messages.append(f"{exposure_us:g} us: 曲线存在严重反转/跳变，请检查光路和衍射级次。")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for exposure_us in exposure_times:
        subset = [row for row in rows if row["exposure_us"] == exposure_us]
        ax.plot([row["gray_value"] for row in subset], [row["integrated_energy"] for row in subset],
                marker=".", linewidth=1, label=f"{exposure_us:g} us")
    ax.set_xlabel("Amplitude SLM gray value"); ax.set_ylabel("Background-corrected integrated energy")
    ax.grid(alpha=.25); ax.legend(); fig.savefig(results / "response_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for exposure_us in exposure_times:
        subset = [row for row in rows if row["exposure_us"] == exposure_us]
        energy = np.asarray([row["integrated_energy"] for row in subset], dtype=np.float64)
        normalized = (energy - energy[0]) / max(float((energy - energy[0]).max()), 1e-12)
        ax.plot([row["gray_value"] for row in subset], normalized, marker=".", linewidth=1,
                label=f"{exposure_us:g} us")
    ax.set_xlabel("Amplitude SLM gray value"); ax.set_ylabel("Normalized integrated energy")
    ax.grid(alpha=.25); ax.legend(); fig.savefig(results / "response_curve_normalized.png", dpi=180); plt.close(fig)
    selected_exposure = exposure_times[0]
    selected = [(gray, preview_frames.get((selected_exposure, gray))) for gray in (0, 64, 128, 192, 255)]
    selected = [(gray, frame) for gray, frame in selected if frame is not None]
    fig, axes = plt.subplots(1, len(selected) + 1, figsize=(4 * (len(selected) + 1), 4), constrained_layout=True)
    axes[0].imshow(preview_uint8(background), cmap="gray"); axes[0].set_title("Background")
    x, y, width, height = windows[selected_exposure]
    for axis, (gray, frame) in zip(axes[1:], selected):
        axis.imshow(preview_uint8(frame), cmap="gray")
        axis.add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor="red"))
        axis.set_title(f"gray={gray}")
    for axis in axes: axis.set_xlabel("x"); axis.set_ylabel("y")
    fig.savefig(results / "exposure_preview.png", dpi=160); plt.close(fig)
    summary = {
        "exposure_times_us": exposure_times, "gray_values": gray_values,
        "measurement_windows_xywh": {str(key): list(value) for key, value in windows.items()},
        "messages": messages, "timestamp": utc_now(),
        "note": "Software does not select or change the formal exposure automatically.",
    }
    json_dump(results / "exposure_summary.json", summary)
    for message in messages: print(f"WARNING: {message}")
    print(f"Exposure-response outputs saved under {results}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware ROI/background/exposure calibration")
    parser.add_argument("command", choices=("generate", "background", "coarse", "fine", "exposure"))
    parser.add_argument("--config", default="configs/calibration/tucam.yaml")
    parser.add_argument("--yes", action="store_true", help="skip operator Enter prompts")
    args = parser.parse_args()
    config, config_path = load_yaml_config(args.config)
    functions: dict[str, Callable[..., dict[str, Any]]] = {
        "generate": generate_calibration_files,
        "background": acquire_background,
        "coarse": run_coarse,
        "fine": run_fine,
        "exposure": run_exposure,
    }
    if args.command == "generate":
        functions[args.command](config, config_path)
    else:
        functions[args.command](config, config_path, assume_yes=args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
