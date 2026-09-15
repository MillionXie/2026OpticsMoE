"""Manual-ROI mask generation and raw gray-response checks.

ROI geometry is deliberately configured by the operator.  This module does
not estimate affine/homography transforms and does not warp camera images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

try:
    from ..devices import build_camera, build_slm, verify_camera_roi
    from ..demos.phase_slm_demo import BlinkPhaseSLM, prepare_phase_frame
    from .calibration_common import (
        json_dump,
        load_yaml_config,
        median_capture,
        resolve_path,
        utc_now,
    )
except ImportError:  # direct execution from workflows/
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from devices import build_camera, build_slm, verify_camera_roi
    from demos.phase_slm_demo import BlinkPhaseSLM, prepare_phase_frame
    from workflows.calibration_common import (
        json_dump,
        load_yaml_config,
        median_capture,
        resolve_path,
        utc_now,
    )


def _paths(config: dict[str, Any], config_path: Path) -> tuple[Path, Path]:
    paths = config["paths"]
    return (
        resolve_path(paths["masks_dir"], config_path.parent),
        resolve_path(paths["results_dir"], config_path.parent),
    )


def _amplitude_roi(config: dict[str, Any]) -> tuple[float, float, float, float]:
    roi = config["amplitude_roi"]
    width, height = float(roi["width"]), float(roi["height"])
    center_x, center_y = float(roi["center_x"]), float(roi["center_y"])
    return center_x - width / 2, center_y - height / 2, width, height


def _configured_slm_size(device: dict[str, Any]) -> tuple[int, int]:
    expected = device.get("expected_resolution_wh")
    if expected is not None:
        if not isinstance(expected, (list, tuple)) or len(expected) != 2:
            raise ValueError("expected_resolution_wh must be [width, height]")
        return int(expected[0]), int(expected[1])
    try:
        return int(device["width"]), int(device["height"])
    except KeyError as exc:
        raise ValueError(
            "SLM config requires expected_resolution_wh or width/height"
        ) from exc


def roi_boundary_source_points(config: dict[str, Any]) -> list[tuple[float, float]]:
    """Return center plus the four exact configured amplitude-ROI corners."""

    left, top, width, height = _amplitude_roi(config)
    return [
        (left + width / 2, top + height / 2),
        (left, top),
        (left + width, top),
        (left, top + height),
        (left + width, top + height),
    ]


def gaussian_marker(
    canvas_size_wh: tuple[int, int],
    center_xy: tuple[float, float],
    marker_size_px: int,
    sigma_px: float,
) -> np.ndarray:
    width, height = canvas_size_wh
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx, cy = center_xy
    radius = marker_size_px / 2
    inside = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
    value = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_px**2))
    return np.rint(np.where(inside, value, 0.0) * 255).astype(np.uint8)


def rectangle_marker(
    canvas_size_wh: tuple[int, int], center_xy: tuple[float, float], marker_size_px: int
) -> np.ndarray:
    width, height = canvas_size_wh
    cx, cy = center_xy
    x0 = max(0, int(round(cx - marker_size_px / 2)))
    y0 = max(0, int(round(cy - marker_size_px / 2)))
    x1 = min(width, x0 + marker_size_px)
    y1 = min(height, y0 + marker_size_px)
    value = np.zeros((height, width), dtype=np.uint8)
    value[y0:y1, x0:x1] = 255
    return value


def exposure_patch(
    canvas_size_wh: tuple[int, int],
    center_xy: tuple[float, float],
    gray_value: int,
    patch_size_px: int,
    inner_size_px: int,
    edge_taper_px: int,
) -> np.ndarray:
    if not 0 <= gray_value <= 255:
        raise ValueError("gray_value must be in 0..255")
    if inner_size_px + 2 * edge_taper_px > patch_size_px:
        raise ValueError("inner_size_px + 2*edge_taper_px exceeds patch_size_px")
    width, height = canvas_size_wh
    cx, cy = center_xy
    x = np.abs(np.arange(width, dtype=np.float32) - cx)
    y = np.abs(np.arange(height, dtype=np.float32) - cy)
    inner_half = inner_size_px / 2
    outer_half = patch_size_px / 2

    def axis_weight(distance: np.ndarray) -> np.ndarray:
        result = np.zeros_like(distance)
        result[distance <= inner_half] = 1.0
        taper = (distance > inner_half) & (distance < outer_half)
        denominator = max(outer_half - inner_half, 1e-6)
        result[taper] = 0.5 * (
            1 + np.cos(np.pi * (distance[taper] - inner_half) / denominator)
        )
        return result

    value = np.outer(axis_weight(y), axis_weight(x)) * float(gray_value)
    return np.rint(value).astype(np.uint8)


def resolve_brightness_gray_values(settings: dict[str, Any]) -> list[int]:
    """Resolve a small deterministic 0..255 response sweep.

    ``gray_values`` has highest priority.  ``gray_point_count`` then creates an
    approximately uniform inclusive grid (32 points by default).  The old
    start/stop/step fields remain readable for existing custom configs, but the
    repository profiles use the bounded point-count form.
    """

    explicit = settings.get("gray_values")
    legacy_range = False
    if explicit is not None:
        if not isinstance(explicit, (list, tuple)) or not explicit:
            raise ValueError("exposure_calibration.gray_values must be a non-empty list")
        values = [int(value) for value in explicit]
    elif "gray_point_count" in settings:
        count = int(settings["gray_point_count"])
        if not 2 <= count <= 256:
            raise ValueError("gray_point_count must be in 2..256")
        values = np.rint(np.linspace(0.0, 255.0, count)).astype(np.int64).tolist()
    elif any(key in settings for key in ("gray_start", "gray_stop", "gray_step")):
        legacy_range = True
        start = int(settings.get("gray_start", 0))
        stop = int(settings.get("gray_stop", 255))
        step = int(settings.get("gray_step", 1))
        if step <= 0 or stop < start:
            raise ValueError("legacy gray_start/gray_stop/gray_step are invalid")
        values = list(range(start, stop + 1, step))
    else:
        values = np.rint(np.linspace(0.0, 255.0, 32)).astype(np.int64).tolist()
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("brightness gray values must all be in 0..255")
    if values != sorted(set(values)):
        raise ValueError("brightness gray values must be unique and strictly increasing")
    if not legacy_range and (values[0] != 0 or values[-1] != 255):
        raise ValueError("brightness gray values must include closed=0 and open=255")
    return values


def _nearest_preview_values(gray_values: list[int], requested: Any) -> set[int]:
    anchors = [0, 128, 255] if requested is None else [int(value) for value in requested]
    if not anchors:
        return set()
    return {
        min(gray_values, key=lambda candidate: (abs(candidate - anchor), candidate))
        for anchor in anchors
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_publication_style() -> dict[str, Any]:
    requested = "Arial"
    resolved_path: str | None = None
    resolved_family: str | None = None
    for candidate in ("Arial", "Helvetica", "DejaVu Sans"):
        try:
            path = font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        resolved_path = str(Path(path).resolve())
        resolved_family = font_manager.FontProperties(fname=path).get_name()
        break
    if resolved_family is None:
        path = font_manager.findfont("sans-serif", fallback_to_default=True)
        resolved_path = str(Path(path).resolve())
        resolved_family = font_manager.FontProperties(fname=path).get_name()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [resolved_family, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fallback = resolved_family.casefold() != requested.casefold()
    if fallback:
        print(
            f"[brightness figure] Arial unavailable; using {resolved_family} "
            f"from {resolved_path}"
        )
    return {
        "requested_font_family": requested,
        "resolved_font_family": resolved_family,
        "resolved_font_path": resolved_path,
        "font_fallback_used": fallback,
        "font_size_pt": 7.0,
        "raster_dpi": 600,
        "response_figure_size_mm": [183.0, 55.0],
        "preview_figure_size_mm": [183.0, 55.0],
    }


def _save_response_figure(fig: Any, base: Path) -> list[str]:
    outputs: list[str] = []
    for suffix, options in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = base.with_suffix(suffix)
        fig.savefig(path, **options)
        outputs.append(path.name)
    return outputs


def generate_calibration_files(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Generate manual-ROI verification, zero, and exposure patterns."""

    masks, _ = _paths(config, config_path)
    amplitude_dir = masks / "amplitude"
    phase_dir = masks / "phase"
    exposure_dir = masks / "exposure"
    for directory in (amplitude_dir, phase_dir, exposure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    amplitude_size = _configured_slm_size(config["amplitude_slm"])
    phase_size = _configured_slm_size(config["phase_slm"])
    Image.new("L", amplitude_size, 0).save(amplitude_dir / "amplitude_zero.bmp")
    Image.new("L", phase_size, 0).save(phase_dir / "phase_zero.bmp")

    calibration = config["calibration"]
    points = roi_boundary_source_points(config)
    point_pattern = np.maximum.reduce(
        [
            gaussian_marker(
                amplitude_size,
                point,
                int(calibration["marker_size_px"]),
                float(calibration["marker_sigma_px"]),
            )
            for point in points
        ]
    )
    rectangle_pattern = np.maximum.reduce(
        [
            rectangle_marker(
                amplitude_size, point, int(calibration["marker_size_px"])
            )
            for point in points
        ]
    )
    Image.fromarray(point_pattern, mode="L").save(amplitude_dir / "verify_roi_5points.bmp")
    Image.fromarray(point_pattern, mode="L").save(amplitude_dir / "verify_5points.bmp")
    Image.fromarray(rectangle_pattern, mode="L").save(
        amplitude_dir / "verify_roi_5rectangles.bmp"
    )

    outline = np.zeros(point_pattern.shape, dtype=np.uint8)
    left, top, roi_width, roi_height = _amplitude_roi(config)
    x0, y0 = int(round(left)), int(round(top))
    x1, y1 = int(round(left + roi_width)), int(round(top + roi_height))
    line_width = max(1, int(calibration.get("roi_outline_width_px", 4)))
    outline[y0 : y0 + line_width, x0:x1] = 255
    outline[y1 - line_width : y1, x0:x1] = 255
    outline[y0:y1, x0 : x0 + line_width] = 255
    outline[y0:y1, x1 - line_width : x1] = 255
    Image.fromarray(outline, mode="L").save(amplitude_dir / "verify_roi_outline.bmp")

    exposure = config["exposure_calibration"]
    gray_values = resolve_brightness_gray_values(exposure)
    expected_exposure_names = {f"gray_{gray:03d}.bmp" for gray in gray_values}
    removed_stale_exposure_masks = 0
    for stale in exposure_dir.glob("gray_*.bmp"):
        if stale.is_file() and stale.name not in expected_exposure_names:
            stale.unlink()
            removed_stale_exposure_masks += 1
    center = (
        float(config["amplitude_roi"]["center_x"]),
        float(config["amplitude_roi"]["center_y"]),
    )
    for gray in gray_values:
        pattern = exposure_patch(
            amplitude_size,
            center,
            gray,
            int(exposure["patch_size_px"]),
            int(exposure["patch_inner_size_px"]),
            int(exposure["edge_taper_px"]),
        )
        Image.fromarray(pattern, mode="L").save(exposure_dir / f"gray_{gray:03d}.bmp")

    manifest_rows: list[dict[str, Any]] = []
    for pattern_id, filename in (
        ("amplitude_zero", "amplitude/amplitude_zero.bmp"),
        ("verify_roi_5points", "amplitude/verify_roi_5points.bmp"),
        ("verify_roi_5rectangles", "amplitude/verify_roi_5rectangles.bmp"),
        ("verify_roi_outline", "amplitude/verify_roi_outline.bmp"),
    ):
        manifest_rows.append(
            {
                "pattern_id": pattern_id,
                "calibration_type": "manual_roi",
                "source_x": config["amplitude_roi"]["center_x"],
                "source_y": config["amplitude_roi"]["center_y"],
                "gray_value": 255 if "zero" not in pattern_id else 0,
                "amplitude_filename": filename,
                "phase_filename": "phase/phase_zero.bmp",
            }
        )
    for gray in gray_values:
        manifest_rows.append(
            {
                "pattern_id": f"gray_{gray:03d}",
                "calibration_type": "exposure",
                "source_x": center[0],
                "source_y": center[1],
                "gray_value": gray,
                "amplitude_filename": f"exposure/gray_{gray:03d}.bmp",
                "phase_filename": "phase/phase_zero.bmp",
            }
        )
    with (masks / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    report = {
        "amplitude_size_wh": list(amplitude_size),
        "amplitude_polarity": {
            "bright_value_uint8": 255,
            "dark_value_uint8": 0,
            "invert_before_export": False,
        },
        "phase_size_wh": list(phase_size),
        "roi_boundary_points": [[float(x), float(y)] for x, y in points],
        "verification_patterns": 3,
        "exposure_patterns": len(gray_values),
        "exposure_gray_values": gray_values,
        "removed_stale_generated_exposure_masks": removed_stale_exposure_masks,
        "automatic_geometry_calibration": False,
    }
    print(f"Generated manual-ROI masks under {masks}")
    return report


class _PhaseController:
    def __init__(self, config: dict[str, Any], config_path: Path) -> None:
        self.config = config["phase_slm"]
        self.base = config_path.parent
        self._context: BlinkPhaseSLM | None = None

    def open(self) -> None:
        driver = str(self.config.get("driver", "manual")).lower()
        if driver == "meadowlark":
            self._context = BlinkPhaseSLM(self.config, self.base)
            self._context.__enter__()
        elif driver != "manual":
            raise ValueError("phase_slm.driver must be manual or meadowlark")

    def display(self, path: Path) -> None:
        if self._context is None:
            print(f"[phase SLM/manual] 请手动加载并保持：{path}")
            return
        expected = _configured_slm_size(self.config)
        correction = (
            resolve_path(self.config["wavefront_correction_file"], self.base)
            if bool(self.config.get("apply_wavefront_correction", False))
            else None
        )
        frame = prepare_phase_frame(
            path,
            expected,
            wavefront_correction=correction,
            flip_vertical=bool(self.config.get("flip_vertical", False)),
        )
        self._context.show(frame)
        print(f"[phase SLM/meadowlark] 已加载：{path.name}")

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None


@contextmanager
def _slm_session(
    config: dict[str, Any], config_path: Path, amplitude_files: list[Path]
) -> Iterator[tuple[Any, _PhaseController]]:
    amplitude_config = dict(config["amplitude_slm"])
    if "expected_resolution_wh" not in amplitude_config:
        amplitude_config["expected_resolution_wh"] = list(
            _configured_slm_size(amplitude_config)
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
        phase.close()
        amplitude.close()


def _camera_config(config: dict[str, Any], exposure_us: float | None = None) -> dict[str, Any]:
    value = dict(config["camera"])
    # Brightness calibration measures the detector response before the formal
    # transport's fixed uint16->uint8 conversion.  Keeping the source dtype
    # avoids hiding weak closed-state leakage or quantizing small response
    # differences.  The temporary NPY frames are deleted by median_capture.
    value["saved_frame_size_wh"] = None
    value["saved_frame_resize_mode"] = "none"
    value["saved_frame_bit_depth"] = None
    value["saved_frame_input_range"] = None
    if exposure_us is not None:
        value["auto_exposure"] = False
        value["exposure_us"] = float(exposure_us)
    return value


def _confirm(message: str, assume_yes: bool) -> None:
    print(message)
    if not assume_yes:
        input("确认设备状态和图像均正确后按 Enter；否则请 Ctrl+C 中止：")


def _open_verified_camera(config: dict[str, Any], config_path: Path, exposure_us: float | None = None):
    camera_config = _camera_config(config, exposure_us)
    expected = verify_camera_roi(camera_config)
    camera = build_camera(camera_config, config_path.parent)
    camera.validate_runtime()
    camera.open()
    try:
        actual = verify_camera_roi(camera_config, camera.device_info())
        if expected is not None:
            print(f"[camera ROI] active [left, top, width, height] = {list(actual or expected)}")
        return camera
    except Exception:
        camera.close()
        raise


def _measurement_window(shape: tuple[int, int], size: int) -> tuple[slice, slice]:
    height, width = shape
    size = min(int(size), height, width)
    x0 = (width - size) // 2
    y0 = (height - size) // 2
    return slice(y0, y0 + size), slice(x0, x0 + size)


def _captured_dtype_max(metadata: list[dict[str, Any]]) -> float:
    dtype_names = {
        str(item.get("dtype") or item.get("source_dtype") or "").strip()
        for item in metadata
    }
    dtype_names.discard("")
    if len(dtype_names) != 1:
        raise RuntimeError(f"camera capture dtype changed during a median group: {dtype_names}")
    dtype = np.dtype(next(iter(dtype_names)))
    if not np.issubdtype(dtype, np.integer):
        raise RuntimeError(f"brightness calibration requires integer detector data, got {dtype}")
    return float(np.iinfo(dtype).max)


def run_exposure(
    config: dict[str, Any], config_path: Path, *, assume_yes: bool = False
) -> dict[str, Any]:
    masks, results = _paths(config, config_path)
    # Regenerate the small deterministic allowlist every time.  This also
    # removes stale gray_*.bmp files left by the historical 256-level sweep.
    generate_calibration_files(config, config_path)
    results.mkdir(parents=True, exist_ok=True)
    for legacy_plot in (
        results / "response_curve.png",
        results / "response_curve_normalized.png",
    ):
        legacy_plot.unlink(missing_ok=True)
    settings = config["exposure_calibration"]
    exposure_times = [float(value) for value in settings["exposure_times_us"]]
    if len(exposure_times) != 1 or exposure_times[0] <= 0.0:
        raise ValueError(
            "fast brightness calibration requires exactly one positive fixed exposure"
        )
    gray_values = resolve_brightness_gray_values(settings)
    frames_per_gray = int(settings.get("frames_per_gray", 3))
    if frames_per_gray != 3:
        raise ValueError(
            "the audited fast brightness calibration requires frames_per_gray=3"
        )
    preview_values = _nearest_preview_values(
        gray_values, settings.get("preview_gray_values")
    )
    patterns = [masks / "exposure" / f"gray_{gray:03d}.bmp" for gray in gray_values]
    missing_patterns = [path for path in patterns if not path.is_file()]
    if missing_patterns:
        raise FileNotFoundError(f"brightness calibration masks are missing: {missing_patterns[:3]}")
    phase_zero = masks / "phase" / "phase_zero.bmp"
    settle_seconds = float(config.get("settle_delay_ms", 200)) / 1000.0
    rows: list[dict[str, Any]] = []
    previews: dict[int, np.ndarray] = {}
    camera_settings: dict[str, dict[str, Any]] = {}
    with _slm_session(config, config_path, patterns) as (amplitude, phase):
        phase.display(phase_zero)
        _confirm(
            "曝光响应检查直接记录当前手动硬件 ROI 的原始强度，不扣 background；"
            "请确认相位 SLM 为 phase_zero.bmp。",
            assume_yes,
        )
        for exposure_us in exposure_times:
            camera = _open_verified_camera(config, config_path, float(exposure_us))
            try:
                camera_settings[str(exposure_us)] = dict(camera.device_info())
                roi = camera.device_info()["device_roi_xywh"]
                frame_shape = (int(roi[3]), int(roi[2]))
                window = _measurement_window(
                    frame_shape,
                    int(settings.get("measurement_window_size_px", 128)),
                )
                for gray, pattern in zip(gray_values, patterns):
                    amplitude.display_file(pattern)
                    time.sleep(settle_seconds)
                    raw, capture_metadata = median_capture(
                        camera,
                        results,
                        f"exposure_{int(exposure_us)}_{gray:03d}",
                        frames_per_gray,
                    )
                    if raw.shape != frame_shape:
                        raise RuntimeError(
                            f"Camera returned shape {raw.shape}, expected ROI shape {frame_shape}"
                        )
                    measured = raw[window].astype(np.float32)
                    dtype_max = _captured_dtype_max(capture_metadata)
                    rows.append(
                        {
                            "gray_value": gray,
                            "exposure_us": float(exposure_us),
                            "frames_in_median": frames_per_gray,
                            "pattern_sha256": _sha256(pattern),
                            "mean_intensity": float(measured.mean()),
                            "median_intensity": float(np.median(measured)),
                            "std_intensity": float(measured.std()),
                            "integrated_energy": float(measured.sum()),
                            "p01": float(np.percentile(measured, 1)),
                            "p99": float(np.percentile(measured, 99)),
                            "max_intensity": float(measured.max()),
                            "detector_dtype_max": dtype_max,
                            "saturated_pixel_fraction": float(np.mean(raw[window] >= dtype_max)),
                        }
                    )
                    if gray in preview_values:
                        previews[gray] = raw
            finally:
                camera.close()
    exposure_summaries: dict[str, dict[str, Any]] = {}
    for exposure_us in exposure_times:
        group = [row for row in rows if row["exposure_us"] == float(exposure_us)]
        closed = float(group[0]["integrated_energy"])
        opened = float(group[-1]["integrated_energy"])
        span = opened - closed
        if span <= 0.0:
            raise RuntimeError(
                f"open-state energy must exceed closed-state leakage at {exposure_us} us"
            )
        energies = np.asarray([row["integrated_energy"] for row in group], dtype=np.float64)
        normalized = (energies - closed) / span
        for row, response in zip(group, normalized):
            row["response_normalized_curve_only"] = float(response)
        negative_steps = np.diff(energies)
        monotonic_violations = int(np.count_nonzero(negative_steps < -0.01 * span))
        exposure_summaries[str(float(exposure_us))] = {
            "closed_state_leakage_integrated_energy": closed,
            "open_state_integrated_energy": opened,
            "open_minus_closed_dynamic_range": span,
            "open_to_closed_ratio": opened / max(closed, 1.0e-12),
            "large_monotonicity_violations": monotonic_violations,
            "curve_only_normalization": "(I-I_closed)/(I_open-I_closed)",
            "closed_state_is_not_a_background_frame": True,
        }

    with (results / "slm_response.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_style = _configure_publication_style()
    response_fig, response_axes = plt.subplots(
        1,
        2,
        figsize=(7.2047244094, 2.1653543307),  # 183 mm x 55 mm
        constrained_layout=True,
    )
    axis, normalized_axis = response_axes
    for exposure_us in exposure_times:
        group = [row for row in rows if row["exposure_us"] == float(exposure_us)]
        x = np.asarray([row["gray_value"] for row in group])
        y = np.asarray([row["integrated_energy"] for row in group], dtype=np.float64)
        axis.plot(x, y, marker="o", markersize=2.2, linewidth=1.0, color="#4C78A8")
        shifted = y - y[0]
        normalized = shifted / max(float(y[-1] - y[0]), 1e-12)
        normalized_axis.plot(
            x,
            normalized,
            marker="o",
            markersize=2.2,
            linewidth=1.0,
            color="#4C78A8",
        )
    axis.set_xlabel("Amplitude SLM gray value")
    axis.set_ylabel("Raw integrated CCD intensity")
    axis.set_title("a  Measured response", loc="left", fontweight="bold")
    normalized_axis.set_xlabel("Amplitude SLM gray value")
    normalized_axis.set_ylabel("(I-I_closed)/(I_open-I_closed)")
    normalized_axis.set_title("b  Curve-only normalization", loc="left", fontweight="bold")
    for current in response_axes:
        current.set_xlim(0, 255)
        current.tick_params(width=0.7, length=2.5)
        current.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    response_exports = _save_response_figure(
        response_fig, results / "brightness_response"
    )
    plt.close(response_fig)
    preview_exports: list[str] = []
    if previews:
        fig, axes = plt.subplots(
            1,
            len(previews),
            figsize=(7.2047244094, 2.1653543307),  # 183 mm x 55 mm
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        # A shared display scale is essential here. Per-panel autoscaling makes
        # the weak gray=0 residual look as bright as gray=255 and is therefore
        # unsuitable for judging extinction or exposure.
        preview_min = min(float(frame.min()) for frame in previews.values())
        preview_max = max(float(frame.max()) for frame in previews.values())
        if preview_max <= preview_min:
            preview_max = preview_min + 1.0
        for axis_item, gray in zip(axes, sorted(previews)):
            frame = previews[gray]
            axis_item.imshow(
                frame, cmap="gray", vmin=preview_min, vmax=preview_max
            )
            axis_item.set_title(
                f"gray={gray}\nmean={float(frame.mean()):.1f} max={float(frame.max()):.0f}"
            )
            axis_item.set_xlabel("CCD x")
            axis_item.set_ylabel("CCD y")
            axis_item.tick_params(width=0.7, length=2.0)
        preview_path = results / "exposure_preview.png"
        fig.savefig(preview_path, dpi=600)
        preview_exports.append(preview_path.name)
        plt.close(fig)
    max_saturation = max(row["saturated_pixel_fraction"] for row in rows)
    summary = {
        "row_count": len(rows),
        "unique_gray_point_count": len(gray_values),
        "gray_values": gray_values,
        "frames_per_gray": frames_per_gray,
        "camera_roi_xywh": config["camera"]["device_roi_xywh"],
        "closed_state_name": "amplitude_slm_gray_0_closed_state_leakage",
        "closed_state_is_background": False,
        "background_capture_performed": False,
        "background_subtraction": False,
        "measurement_window": "fixed_center_window",
        "preview_scaling": "shared_raw_min_max_across_preview_frames",
        "preview_gray_values": sorted(previews),
        "figure_style": figure_style,
        "figure_exports": response_exports + preview_exports,
        "raw_full_frames_persisted": False,
        "camera_readback_by_exposure_us": camera_settings,
        "response_by_exposure_us": exposure_summaries,
        "max_saturated_pixel_fraction": max_saturation,
        "warning": (
            "Saturation detected; reduce exposure."
            if max_saturation > float(settings.get("saturation_fraction_warning", 0.001))
            else None
        ),
        "timestamp": utc_now(),
    }
    json_dump(results / "exposure_summary.json", summary)
    print(f"Exposure response saved under {results}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate manual-ROI masks or check the raw 0..255 gray response"
    )
    parser.add_argument("phase", choices=("generate", "exposure"))
    parser.add_argument("--config", default="configs/tucam_windows.yaml")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    config, config_path = load_yaml_config(args.config)
    if args.phase == "generate":
        generate_calibration_files(config, config_path)
    else:
        run_exposure(config, config_path, assume_yes=args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
