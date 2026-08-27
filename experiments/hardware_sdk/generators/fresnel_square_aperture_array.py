"""Generate full-aperture Fresnel arrays with matched amplitude masks.

This is deliberately separate from ``fresnel_roi_vertex_array`` (v2).  The
v2 masks partition the 478-pixel support into inward-facing quarter/half
lenses.  Here every n1/n4/n9 marker is produced by a complete, isolated square
aperture centred on its requested physical focus.  A square pupil keeps an
ordinary quadratic Fresnel phase while producing a controllable sinc-like PSF
whose axial sidelobes form a visible cross.

The amplitude mask is part of the optical contract.  Pixels outside the pupil
are 0 (closed), pixels inside are 255 (open).  Playing an all-zero amplitude
frame cannot produce the simulated focus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


SCHEMA_VERSION = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reflect_vertical_about_edge_center(
    value: np.ndarray,
    center_y_edge: float,
) -> np.ndarray:
    """Flip raster content about a configured edge-coordinate centre.

    A full-panel ``flipud`` reflects about ``height/2`` and would move a
    calibrated centre such as y=590 to y=610 on a 1200-row SLM.  Hardware
    phase export instead flips the active content while keeping its requested
    raster centre fixed.  Exact nearest-pixel reflection requires the centre
    to lie on an integer or half-integer edge coordinate.
    """

    source = np.asarray(value)
    if source.ndim != 2:
        raise ValueError("phase-centred reflection expects a 2-D raster")
    twice_center = 2.0 * float(center_y_edge)
    rounded_twice_center = int(round(twice_center))
    if not math.isclose(twice_center, rounded_twice_center, abs_tol=1.0e-9):
        raise ValueError(
            "phase center_y must be an integer or half-integer edge coordinate"
        )
    source_rows = np.arange(source.shape[0], dtype=np.int64)
    destination_rows = rounded_twice_center - 1 - source_rows
    valid = (destination_rows >= 0) & (destination_rows < source.shape[0])
    if np.any(source[~valid] != 0):
        raise ValueError(
            "phase content would be clipped by reflection about the configured centre"
        )
    result = np.zeros_like(source)
    result[destination_rows[valid]] = source[source_rows[valid]]
    return result


def _save_u8_image(
    array: np.ndarray,
    path: Path,
    *,
    image_format: str,
    expected_size_wh: tuple[int, int],
    report_root: Path,
) -> dict[str, Any]:
    value = np.asarray(array, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="L").save(path, format=image_format)
    with Image.open(path) as image:
        if (
            image.format != image_format
            or image.mode != "L"
            or image.size != expected_size_wh
        ):
            raise RuntimeError(
                f"Invalid image {path}: format={image.format}, mode={image.mode}, "
                f"size={image.size}, expected={expected_size_wh}"
            )
    return {
        "path": path.relative_to(report_root).as_posix(),
        "sha256": _sha256(path),
        "format": image_format,
        "mode": "L",
        "size_wh": list(expected_size_wh),
        "min_uint8": int(value.min()),
        "max_uint8": int(value.max()),
    }


def roi_axis_targets_um(
    *, amplitude_active_size_px: int, amplitude_pixel_pitch_um: float, grid_size: int
) -> list[float]:
    """Return target coordinates relative to the shared optical-axis centre."""

    width_um = float(amplitude_active_size_px) * float(amplitude_pixel_pitch_um)
    if grid_size == 1:
        return [0.0]
    if grid_size == 2:
        return [-width_um / 2.0, width_um / 2.0]
    if grid_size == 3:
        return [-width_um / 2.0, 0.0, width_um / 2.0]
    raise ValueError("grid_size must be 1, 2, or 3")


def _axis_selection(
    *,
    size: int,
    center_edge_coordinate: float,
    pixel_pitch_um: float,
    target_um: float,
    requested_width_um: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    centers_um = (
        np.arange(int(size), dtype=np.float64) + 0.5 - float(center_edge_coordinate)
    ) * float(pixel_pitch_um)
    selected = np.abs(centers_um - float(target_um)) <= float(requested_width_um) / 2
    indexes = np.flatnonzero(selected)
    if indexes.size == 0:
        raise RuntimeError("Requested aperture selects no hardware pixels")
    if np.any(np.diff(indexes) != 1):
        raise RuntimeError("A square-aperture axis selection must be contiguous")
    left = int(indexes[0])
    right = int(indexes[-1]) + 1
    physical_left = (left - float(center_edge_coordinate)) * float(pixel_pitch_um)
    physical_right = (right - float(center_edge_coordinate)) * float(pixel_pitch_um)
    if not physical_left < float(target_um) < physical_right:
        raise RuntimeError("The full aperture does not surround its target")
    return selected, {
        "index_bounds_edge": [left, right],
        "physical_bounds_um": [physical_left, physical_right],
        "pixel_count": int(indexes.size),
        "actual_width_um": physical_right - physical_left,
        "center_error_um": (physical_left + physical_right) / 2 - float(target_um),
    }


def build_matched_fresnel_pair(
    *,
    grid_size: int,
    aperture_width_phase_px: int,
    wavelength_nm: float,
    propagation_cm: float,
    amplitude_size_wh: tuple[int, int],
    amplitude_pitch_um: float,
    amplitude_center_xy: tuple[float, float],
    amplitude_active_size_px: int,
    phase_size_wh: tuple[int, int],
    phase_pitch_um: float,
    phase_center_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return amplitude BMP, logical phase, mapped illumination, and lenslets.

    The requested aperture width is specified on the 8 um phase grid, but the
    physical pupil is rasterised on the 17 um amplitude SLM.  Its exact pixel
    edges are then mapped back to the phase plane.  Consequently simulation,
    amplitude BMP, and phase support describe the same physical rectangles.
    """

    if int(aperture_width_phase_px) <= 0:
        raise ValueError("aperture_width_phase_px must be positive")
    amplitude_width, amplitude_height = map(int, amplitude_size_wh)
    phase_width, phase_height = map(int, phase_size_wh)
    requested_width_um = float(aperture_width_phase_px) * float(phase_pitch_um)
    targets_um = roi_axis_targets_um(
        amplitude_active_size_px=int(amplitude_active_size_px),
        amplitude_pixel_pitch_um=float(amplitude_pitch_um),
        grid_size=int(grid_size),
    )

    amplitude = np.zeros((amplitude_height, amplitude_width), dtype=np.uint8)
    logical_phase = np.zeros((phase_height, phase_width), dtype=np.uint8)
    mapped_illumination = np.zeros((phase_height, phase_width), dtype=bool)
    phase_x_um = (
        np.arange(phase_width, dtype=np.float64) + 0.5 - float(phase_center_xy[0])
    ) * float(phase_pitch_um)
    phase_y_um = (
        np.arange(phase_height, dtype=np.float64) + 0.5 - float(phase_center_xy[1])
    ) * float(phase_pitch_um)
    wavelength_um = float(wavelength_nm) * 1.0e-3
    distance_um = float(propagation_cm) * 1.0e4
    lenslets: list[dict[str, Any]] = []

    for row, target_y_um in enumerate(targets_um):
        amp_y, amp_y_info = _axis_selection(
            size=amplitude_height,
            center_edge_coordinate=float(amplitude_center_xy[1]),
            pixel_pitch_um=float(amplitude_pitch_um),
            target_um=target_y_um,
            requested_width_um=requested_width_um,
        )
        for column, target_x_um in enumerate(targets_um):
            amp_x, amp_x_info = _axis_selection(
                size=amplitude_width,
                center_edge_coordinate=float(amplitude_center_xy[0]),
                pixel_pitch_um=float(amplitude_pitch_um),
                target_um=target_x_um,
                requested_width_um=requested_width_um,
            )
            amplitude_region = np.outer(amp_y, amp_x)
            if np.any((amplitude > 0) & amplitude_region):
                raise RuntimeError("Independent amplitude pupils overlap")
            amplitude[amplitude_region] = 255

            x_left_um, x_right_um = amp_x_info["physical_bounds_um"]
            y_top_um, y_bottom_um = amp_y_info["physical_bounds_um"]
            phase_x = (phase_x_um >= x_left_um) & (phase_x_um < x_right_um)
            phase_y = (phase_y_um >= y_top_um) & (phase_y_um < y_bottom_um)
            phase_region = np.outer(phase_y, phase_x)
            if np.any(mapped_illumination & phase_region):
                raise RuntimeError("Independent full phase pupils overlap")
            phase_x_indexes = np.flatnonzero(phase_x)
            phase_y_indexes = np.flatnonzero(phase_y)
            if phase_x_indexes.size == 0 or phase_y_indexes.size == 0:
                raise RuntimeError("Mapped amplitude pupil misses the phase SLM")
            if (
                phase_x_indexes[0] == 0
                or phase_x_indexes[-1] == phase_width - 1
                or phase_y_indexes[0] == 0
                or phase_y_indexes[-1] == phase_height - 1
            ):
                raise RuntimeError("A full lenslet aperture is clipped by the phase SLM")
            mapped_illumination |= phase_region

            yy_um, xx_um = np.meshgrid(
                phase_y_um[phase_y], phase_x_um[phase_x], indexing="ij"
            )
            phase_rad = np.mod(
                -math.pi
                * (
                    (xx_um - target_x_um) ** 2
                    + (yy_um - target_y_um) ** 2
                )
                / (wavelength_um * distance_um),
                2 * math.pi,
            )
            encoded = np.rint(phase_rad * 255.0 / (2 * math.pi)).astype(np.uint8)
            logical_phase[np.ix_(phase_y, phase_x)] = encoded

            phase_left = int(phase_x_indexes[0])
            phase_right = int(phase_x_indexes[-1]) + 1
            phase_top = int(phase_y_indexes[0])
            phase_bottom = int(phase_y_indexes[-1]) + 1
            target_phase_xy = [
                float(phase_center_xy[0]) + target_x_um / float(phase_pitch_um),
                float(phase_center_xy[1]) + target_y_um / float(phase_pitch_um),
            ]
            target_amplitude_xy = [
                float(amplitude_center_xy[0])
                + target_x_um / float(amplitude_pitch_um),
                float(amplitude_center_xy[1])
                + target_y_um / float(amplitude_pitch_um),
            ]
            lenslets.append(
                {
                    "logical_index_row_major": row * int(grid_size) + column,
                    "logical_row": row,
                    "logical_column": column,
                    "aperture_kind": "full_independent_square",
                    "target_physical_xy_um": [target_x_um, target_y_um],
                    "target_amplitude_edge_xy": target_amplitude_xy,
                    "target_phase_edge_xy_before_export_flip": target_phase_xy,
                    "target_phase_edge_xy_in_exported_bmp": [
                        target_phase_xy[0],
                        2.0 * float(phase_center_xy[1]) - target_phase_xy[1],
                    ],
                    "amplitude_aperture_bounds_edge_xyxy": [
                        amp_x_info["index_bounds_edge"][0],
                        amp_y_info["index_bounds_edge"][0],
                        amp_x_info["index_bounds_edge"][1],
                        amp_y_info["index_bounds_edge"][1],
                    ],
                    "amplitude_aperture_size_wh_px": [
                        amp_x_info["pixel_count"],
                        amp_y_info["pixel_count"],
                    ],
                    "amplitude_aperture_actual_size_wh_um": [
                        amp_x_info["actual_width_um"],
                        amp_y_info["actual_width_um"],
                    ],
                    "amplitude_aperture_center_error_xy_um": [
                        amp_x_info["center_error_um"],
                        amp_y_info["center_error_um"],
                    ],
                    "phase_mapped_aperture_bounds_edge_xyxy_before_flip": [
                        phase_left,
                        phase_top,
                        phase_right,
                        phase_bottom,
                    ],
                    "phase_mapped_aperture_size_wh_px": [
                        phase_right - phase_left,
                        phase_bottom - phase_top,
                    ],
                    "phase_mapped_aperture_bounds_edge_xyxy_in_exported_bmp": [
                        phase_left,
                        2.0 * float(phase_center_xy[1]) - phase_bottom,
                        phase_right,
                        2.0 * float(phase_center_xy[1]) - phase_top,
                    ],
                    "full_aperture_not_clipped": True,
                }
            )

    return amplitude, logical_phase, mapped_illumination, lenslets


def _connected_width(values: np.ndarray, center: int, threshold: float) -> int:
    left = int(center)
    right = int(center)
    while left > 0 and values[left - 1] >= threshold:
        left -= 1
    while right + 1 < values.size and values[right + 1] >= threshold:
        right += 1
    return right - left + 1


def simulate_focus_array(
    logical_phase: np.ndarray,
    mapped_illumination: np.ndarray,
    *,
    targets_axis_um: list[float],
    phase_center_xy: tuple[float, float],
    phase_pitch_um: float,
    wavelength_nm: float,
    propagation_cm: float,
    pad_size: int,
    actual_aperture_width_um: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Propagate the quantised BMP pair and measure focus/cross quality."""

    phase = np.asarray(logical_phase, dtype=np.uint8)
    illumination = np.asarray(mapped_illumination, dtype=bool)
    phase_height, phase_width = phase.shape
    pad_size = int(pad_size)
    pad_center = pad_size / 2
    left = int(round(pad_center - float(phase_center_xy[0])))
    top = int(round(pad_center - float(phase_center_xy[1])))
    if left < 0 or top < 0 or left + phase_width > pad_size or top + phase_height > pad_size:
        raise ValueError("pad_size cannot contain the centred phase SLM")

    field = np.zeros((pad_size, pad_size), dtype=np.complex64)
    phase_rad = phase.astype(np.float32) * np.float32(2 * math.pi / 255)
    field[top : top + phase_height, left : left + phase_width] = (
        illumination * np.exp(1j * phase_rad)
    ).astype(np.complex64)

    pitch_m = float(phase_pitch_um) * 1.0e-6
    wavelength_m = float(wavelength_nm) * 1.0e-9
    distance_m = float(propagation_cm) * 1.0e-2
    frequencies = np.fft.fftfreq(pad_size, d=pitch_m)
    fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
    transfer = np.exp(
        1j
        * (2 * math.pi / wavelength_m)
        * distance_m
        * np.sqrt(
            np.maximum(0.0, 1.0 - (wavelength_m * fx) ** 2 - (wavelength_m * fy) ** 2)
        )
    ).astype(np.complex64)
    propagated = np.fft.ifft2(np.fft.fft2(field) * transfer)
    intensity = (np.abs(propagated) ** 2).astype(np.float32)

    first_zero_px = (
        wavelength_m * distance_m / (float(actual_aperture_width_um) * 1.0e-6)
    ) / pitch_m
    theoretical_fwhm_px = 0.886 * first_zero_px
    if len(targets_axis_um) > 1:
        target_spacing_px = abs(targets_axis_um[1] - targets_axis_um[0]) / float(
            phase_pitch_um
        )
        maximum_radius = int(math.floor(target_spacing_px / 3))
    else:
        target_spacing_px = None
        maximum_radius = 256
    analysis_radius = min(maximum_radius, max(48, int(math.ceil(8 * first_zero_px))))
    search_radius = max(6, int(math.ceil(2 * first_zero_px)))
    yy_all, xx_all = np.ogrid[:pad_size, :pad_size]
    target_windows = np.zeros((pad_size, pad_size), dtype=bool)
    peaks: list[dict[str, Any]] = []
    peak_values: list[float] = []

    for row, target_y_um in enumerate(targets_axis_um):
        for column, target_x_um in enumerate(targets_axis_um):
            expected_x = pad_center + target_x_um / float(phase_pitch_um)
            expected_y = pad_center + target_y_um / float(phase_pitch_um)
            nearest_x = int(math.floor(expected_x - 0.5))
            nearest_y = int(math.floor(expected_y - 0.5))
            patch_left = max(0, nearest_x - search_radius)
            patch_right = min(pad_size, nearest_x + search_radius + 2)
            patch_top = max(0, nearest_y - search_radius)
            patch_bottom = min(pad_size, nearest_y + search_radius + 2)
            patch = intensity[patch_top:patch_bottom, patch_left:patch_right]
            local_y, local_x = np.unravel_index(int(np.argmax(patch)), patch.shape)
            peak_y = patch_top + int(local_y)
            peak_x = patch_left + int(local_x)
            peak_value = float(intensity[peak_y, peak_x])
            peak_values.append(peak_value)
            error_x = peak_x + 0.5 - expected_x
            error_y = peak_y + 0.5 - expected_y

            normalized_x = intensity[peak_y, :] / max(peak_value, 1.0e-30)
            normalized_y = intensity[:, peak_x] / max(peak_value, 1.0e-30)
            fwhm_x = _connected_width(normalized_x, peak_x, 0.5)
            fwhm_y = _connected_width(normalized_y, peak_y, 0.5)

            local_left = max(0, peak_x - analysis_radius)
            local_right = min(pad_size, peak_x + analysis_radius + 1)
            local_top = max(0, peak_y - analysis_radius)
            local_bottom = min(pad_size, peak_y + analysis_radius + 1)
            local = intensity[local_top:local_bottom, local_left:local_right]
            local_y_grid, local_x_grid = np.ogrid[
                local_top:local_bottom, local_left:local_right
            ]
            dx = local_x_grid - peak_x
            dy = local_y_grid - peak_y
            central_half = max(1, int(math.ceil(theoretical_fwhm_px)))
            axial = (
                ((np.abs(dx) <= central_half) | (np.abs(dy) <= central_half))
                & ((np.abs(dx) > central_half) | (np.abs(dy) > central_half))
            )
            diagonal = (
                (np.abs(dx) > central_half)
                & (np.abs(dy) > central_half)
                & (dx**2 + dy**2 <= analysis_radius**2)
            )
            axial_mean = float(np.mean(local[axial])) if np.any(axial) else 0.0
            diagonal_mean = float(np.mean(local[diagonal])) if np.any(diagonal) else 0.0
            cross_ratio = axial_mean / max(diagonal_mean, 1.0e-30)

            line_radius = analysis_radius
            x_start = max(0, peak_x - line_radius)
            x_stop = min(pad_size, peak_x + line_radius + 1)
            y_start = max(0, peak_y - line_radius)
            y_stop = min(pad_size, peak_y + line_radius + 1)
            x_offsets = np.arange(x_start, x_stop) - peak_x
            y_offsets = np.arange(y_start, y_stop) - peak_y
            x_above = np.abs(x_offsets[normalized_x[x_start:x_stop] >= 1.0e-3])
            y_above = np.abs(y_offsets[normalized_y[y_start:y_stop] >= 1.0e-3])
            x_extent = int(x_above.max()) if x_above.size else 0
            y_extent = int(y_above.max()) if y_above.size else 0

            outside_main_x = np.abs(x_offsets) >= max(2, int(math.ceil(first_zero_px)))
            outside_main_y = np.abs(y_offsets) >= max(2, int(math.ceil(first_zero_px)))
            side_x = (
                float(np.max(normalized_x[x_start:x_stop][outside_main_x]))
                if np.any(outside_main_x)
                else 0.0
            )
            side_y = (
                float(np.max(normalized_y[y_start:y_stop][outside_main_y]))
                if np.any(outside_main_y)
                else 0.0
            )
            maximum_sidelobe = max(side_x, side_y, 1.0e-30)

            window = (
                (np.abs(xx_all - peak_x) <= analysis_radius)
                & (np.abs(yy_all - peak_y) <= analysis_radius)
            )
            target_windows |= window
            peaks.append(
                {
                    "logical_row": row,
                    "logical_column": column,
                    "expected_output_edge_xy_phase_grid": [expected_x, expected_y],
                    "peak_output_index_xy_phase_grid": [peak_x, peak_y],
                    "position_error_xy_phase_px": [error_x, error_y],
                    "peak_intensity": peak_value,
                    "measured_fwhm_xy_phase_px": [fwhm_x, fwhm_y],
                    "axis_extent_at_minus30db_xy_phase_px": [x_extent, y_extent],
                    "maximum_axial_sidelobe_db": 10 * math.log10(maximum_sidelobe),
                    "cross_axis_to_diagonal_mean_energy_ratio": cross_ratio,
                }
            )

    unique_peaks = len({tuple(item["peak_output_index_xy_phase_grid"]) for item in peaks}) == len(peaks)
    max_error = max(
        abs(value)
        for item in peaks
        for value in item["position_error_xy_phase_px"]
    )
    minimum_peak = min(peak_values)
    maximum_background = float(np.max(intensity[~target_windows]))
    peak_to_background = minimum_peak / max(maximum_background, 1.0e-30)
    mean_peak = sum(peak_values) / len(peak_values)
    peak_cv = (
        math.sqrt(sum((value - mean_peak) ** 2 for value in peak_values) / len(peak_values))
        / max(mean_peak, 1.0e-30)
    )
    target_energy_fraction = float(np.sum(intensity[target_windows])) / float(
        np.sum(intensity)
    )
    passed = max_error <= 1.0 and unique_peaks and peak_to_background >= 10.0
    return intensity, {
        "method": "angular-spectrum propagation of quantized phase with mapped 17um binary amplitude pupil",
        "pad_size": pad_size,
        "propagation_cm": float(propagation_cm),
        "theoretical_square_aperture_first_zero_radius_phase_px": first_zero_px,
        "theoretical_square_aperture_fwhm_phase_px": theoretical_fwhm_px,
        "analysis_radius_phase_px": analysis_radius,
        "target_spacing_phase_px": target_spacing_px,
        "max_abs_position_error_phase_px": max_error,
        "unique_peak_assignment": unique_peaks,
        "minimum_target_peak_to_max_background": peak_to_background,
        "minimum_target_peak_to_max_background_db": 10 * math.log10(
            max(peak_to_background, 1.0e-30)
        ),
        "target_window_energy_fraction": target_energy_fraction,
        "target_peak_uniformity_cv": peak_cv,
        "passed": passed,
        "peaks": peaks,
    }


def _camera_preview(
    intensity: np.ndarray,
    *,
    phase_pitch_um: float,
    camera_size_wh: tuple[int, int],
    camera_pitch_um: float,
    log_floor_db: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    camera_width, camera_height = map(int, camera_size_wh)
    if camera_width != camera_height:
        raise ValueError("The current preview helper requires a square camera")
    physical_width_um = camera_width * float(camera_pitch_um)
    crop_size = int(round(physical_width_um / float(phase_pitch_um)))
    if crop_size > min(intensity.shape):
        raise ValueError("The simulated grid is smaller than the camera field")
    center_y = intensity.shape[0] // 2
    center_x = intensity.shape[1] // 2
    left = center_x - crop_size // 2
    top = center_y - crop_size // 2
    crop = intensity[top : top + crop_size, left : left + crop_size]
    maximum = float(np.max(crop))
    normalized = np.clip(crop / max(maximum, 1.0e-30), 0.0, 1.0)
    linear_float = Image.fromarray(normalized.astype(np.float32), mode="F").resize(
        (camera_width, camera_height), Image.Resampling.BILINEAR
    )
    linear = np.rint(np.asarray(linear_float) * 255).astype(np.uint8)
    db = 10 * np.log10(np.maximum(normalized, 10 ** (float(log_floor_db) / 10)))
    log_normalized = np.clip(
        (db - float(log_floor_db)) / -float(log_floor_db), 0.0, 1.0
    )
    log_float = Image.fromarray(log_normalized.astype(np.float32), mode="F").resize(
        (camera_width, camera_height), Image.Resampling.BILINEAR
    )
    log_image = np.rint(np.asarray(log_float) * 255).astype(np.uint8)
    return linear, log_image, {
        "camera_size_wh": [camera_width, camera_height],
        "camera_pixel_pitch_um": float(camera_pitch_um),
        "assumed_lateral_magnification": 1.0,
        "simulated_phase_grid_crop_size_wh": [crop_size, crop_size],
        "log_floor_db": float(log_floor_db),
        "normalization": "per-pattern global maximum",
    }


def _write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    widths = ", ".join(
        str(value) for value in manifest["aperture_sweep"]["requested_phase_widths_px"]
    )
    output_dir.joinpath("README.md").write_text(
        f"""# 完整方孔 Fresnel 十字状焦斑标定（v3）

该目录不会覆盖 v2。这里仍是普通二次 Fresnel 相位，不是 CGH 图案。
每个焦点使用完整、独立、以目标点为中心的方孔，不再把 ROI 顶点处的透镜截成
quarter/half lens。方孔焦斑为 sinc²×sinc²：中央仍是焦点，轴向旁瓣形成可见十字。

## 固定参数

- 532 nm，传播 10 cm；相位 SLM 8 μm、1920×1200，中心 `(980,590)`。
- 振幅 SLM 17 μm、1024×1024，中心 `(512,512)`，`255=open`、`0=closed`。
- 相位 BMP 已围绕配置中心 y=590 做纵向翻转，导出中心仍为 `(980,590)`；
  不是围绕整块面板 y=600 翻转，播放端不得再次翻转。
- n4 焦点是 478×17 μm 物理 ROI 的四个精确顶点；n9 还包含边中点和中心。
- 请求相位孔径宽度档为 `{widths}` px；实际物理孔径由 17 μm 振幅像素量化，manifest
  同时记录请求值和实际值。

## 必须成对播放

`amplitude_bmp/` 与 `phase_bmp/` 中 basename 相同的文件是一对。禁止用全 0 振幅
配合相位图：当前正确极性下全 0 表示完全关闭，只会看到漏光、杂散和相机噪声。
也不要用全白振幅替代匹配孔径，否则平相位区域会提高背景。

## 选择孔径

孔径越小，焦斑和十字越大，这与“增大透镜直径会让十字变大”的猜测正好相反；代价是
总光通量降低。建议先播放 `a64px`（实际约 510 μm，默认的大十字档），若仍太小则换
`a48px`（实际约 374 μm，最大十字档）；若杂散或相邻旁瓣影响观察，可改用 `a96px`
（约 782 μm，折中档）或 `a128px`（约 1020 μm，紧凑高通量档）。曝光不得饱和。
孔径只改变焦斑尺寸，不改变 n1/n4/n9 的目标中心坐标。phase BMP 只含各完整方孔内的
普通二次相位，没有人为十字、描边圆圈或额外 CGH 图案。

`ideal_ccd_linear/` 是逐图峰值归一化的线性预览；`ideal_ccd_log/` 显示低强度旁瓣。
它们假设 1×横向倍率和 6.5 μm、2048×2048 相机，仅用于检查数学图案。实验 ROI
仍必须由实测四顶点确定，不能直接照抄理想 CCD 坐标。

`numerical_metrics.csv/json` 和 manifest 记录位置误差、FWHM、-30 dB 轴向范围、
旁瓣、目标外背景、能量占比和多点均匀性。
""",
        encoding="utf-8",
        newline="\n",
    )


def generate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(raw["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelength_nm = float(raw["wavelength_nm"])
    propagation_cm = float(raw["propagation_cm"])
    amplitude_raw = raw["amplitude_slm"]
    phase_raw = raw["phase_slm"]
    support_raw = raw["shared_optical_support"]
    simulation_raw = raw["simulation"]
    amplitude_size = tuple(map(int, amplitude_raw["size_wh"]))
    phase_size = tuple(map(int, phase_raw["size_wh"]))
    amplitude_pitch = float(amplitude_raw["pixel_pitch_um"])
    phase_pitch = float(phase_raw["pixel_pitch_um"])
    amplitude_center = tuple(map(float, amplitude_raw["center_xy"]))
    phase_center = tuple(map(float, phase_raw["center_xy"]))
    active_size = int(support_raw["amplitude_active_size_px"])
    widths = [int(value) for value in raw["aperture_widths_phase_px"]]
    array_counts = [int(value) for value in raw.get("array_counts", [1, 4, 9])]
    if sorted(set(array_counts)) != [1, 4, 9]:
        raise ValueError("array_counts must contain exactly 1, 4, and 9")
    if not bool(phase_raw.get("flip_vertical", True)) or bool(
        phase_raw.get("flip_horizontal", False)
    ):
        raise ValueError("v3 formal export requires vertical-only phase flip")

    amplitude_dir = output_dir / "amplitude_bmp"
    phase_dir = output_dir / "phase_bmp"
    linear_dir = output_dir / "ideal_ccd_linear"
    log_dir = output_dir / "ideal_ccd_log"
    support_dir = output_dir / "aperture_preview"
    pairs: list[dict[str, Any]] = []
    numerical_rows: list[dict[str, Any]] = []

    for aperture_width in widths:
        for array_count in array_counts:
            grid_size = int(round(math.sqrt(array_count)))
            amplitude, logical_phase, mapped_illumination, lenslets = (
                build_matched_fresnel_pair(
                    grid_size=grid_size,
                    aperture_width_phase_px=aperture_width,
                    wavelength_nm=wavelength_nm,
                    propagation_cm=propagation_cm,
                    amplitude_size_wh=amplitude_size,
                    amplitude_pitch_um=amplitude_pitch,
                    amplitude_center_xy=amplitude_center,
                    amplitude_active_size_px=active_size,
                    phase_size_wh=phase_size,
                    phase_pitch_um=phase_pitch,
                    phase_center_xy=phase_center,
                )
            )
            exported_phase = reflect_vertical_about_edge_center(
                logical_phase, phase_center[1]
            )
            base = f"fresnel_square_n{array_count}_a{aperture_width}px_z10cm"
            amplitude_report = _save_u8_image(
                amplitude,
                amplitude_dir / f"amplitude_{base}_17um_1024x1024.bmp",
                image_format="BMP",
                expected_size_wh=amplitude_size,
                report_root=output_dir,
            )
            phase_report = _save_u8_image(
                exported_phase,
                phase_dir / f"phase_{base}_532nm_8um_1920x1200.bmp",
                image_format="BMP",
                expected_size_wh=phase_size,
                report_root=output_dir,
            )
            support_report = _save_u8_image(
                mapped_illumination.astype(np.uint8) * 255,
                support_dir / f"support_{base}_before_flip.png",
                image_format="PNG",
                expected_size_wh=phase_size,
                report_root=output_dir,
            )

            actual_width_um = float(
                lenslets[0]["amplitude_aperture_actual_size_wh_um"][0]
            )
            targets_um = roi_axis_targets_um(
                amplitude_active_size_px=active_size,
                amplitude_pixel_pitch_um=amplitude_pitch,
                grid_size=grid_size,
            )
            intensity, metrics = simulate_focus_array(
                logical_phase,
                mapped_illumination,
                targets_axis_um=targets_um,
                phase_center_xy=phase_center,
                phase_pitch_um=phase_pitch,
                wavelength_nm=wavelength_nm,
                propagation_cm=propagation_cm,
                pad_size=int(simulation_raw["pad_size"]),
                actual_aperture_width_um=actual_width_um,
            )
            if not metrics["passed"]:
                raise RuntimeError(
                    f"Numerical validation failed for n{array_count}, aperture "
                    f"{aperture_width}px: {metrics}"
                )
            linear, log_image, camera_contract = _camera_preview(
                intensity,
                phase_pitch_um=phase_pitch,
                camera_size_wh=tuple(map(int, simulation_raw["camera_size_wh"])),
                camera_pitch_um=float(simulation_raw["camera_pixel_pitch_um"]),
                log_floor_db=float(simulation_raw.get("log_floor_db", -50.0)),
            )
            linear_report = _save_u8_image(
                linear,
                linear_dir / f"ideal_ccd_{base}_linear.png",
                image_format="PNG",
                expected_size_wh=tuple(map(int, simulation_raw["camera_size_wh"])),
                report_root=output_dir,
            )
            log_report = _save_u8_image(
                log_image,
                log_dir / f"ideal_ccd_{base}_log.png",
                image_format="PNG",
                expected_size_wh=tuple(map(int, simulation_raw["camera_size_wh"])),
                report_root=output_dir,
            )
            pair = {
                "pair_id": base,
                "array_count": array_count,
                "grid_size": grid_size,
                "requested_aperture_width_phase_px": aperture_width,
                "requested_aperture_width_um": aperture_width * phase_pitch,
                "actual_amplitude_aperture_width_um": actual_width_um,
                "amplitude_file": amplitude_report,
                "phase_file": phase_report,
                "phase_support_preview": support_report,
                "ideal_ccd_linear": linear_report,
                "ideal_ccd_log": log_report,
                "ideal_camera_contract": camera_contract,
                "lenslets": lenslets,
                "numerical_validation": metrics,
            }
            pairs.append(pair)
            numerical_rows.append(
                {
                    "pair_id": base,
                    "array_count": array_count,
                    "requested_aperture_width_phase_px": aperture_width,
                    "actual_aperture_width_um": actual_width_um,
                    "theoretical_fwhm_phase_px": metrics[
                        "theoretical_square_aperture_fwhm_phase_px"
                    ],
                    "max_position_error_phase_px": metrics[
                        "max_abs_position_error_phase_px"
                    ],
                    "minimum_peak_to_background": metrics[
                        "minimum_target_peak_to_max_background"
                    ],
                    "minimum_peak_to_background_db": metrics[
                        "minimum_target_peak_to_max_background_db"
                    ],
                    "target_energy_fraction": metrics[
                        "target_window_energy_fraction"
                    ],
                    "peak_uniformity_cv": metrics["target_peak_uniformity_cv"],
                    "mean_measured_fwhm_x_phase_px": sum(
                        item["measured_fwhm_xy_phase_px"][0]
                        for item in metrics["peaks"]
                    )
                    / len(metrics["peaks"]),
                    "mean_measured_fwhm_y_phase_px": sum(
                        item["measured_fwhm_xy_phase_px"][1]
                        for item in metrics["peaks"]
                    )
                    / len(metrics["peaks"]),
                    "mean_cross_axis_to_diagonal_ratio": sum(
                        item["cross_axis_to_diagonal_mean_energy_ratio"]
                        for item in metrics["peaks"]
                    )
                    / len(metrics["peaks"]),
                    "passed": metrics["passed"],
                }
            )

    metrics_json = output_dir / "numerical_metrics.json"
    metrics_json.write_text(
        json.dumps(numerical_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    metrics_csv = output_dir / "numerical_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(numerical_rows[0]))
        writer.writeheader()
        writer.writerows(numerical_rows)

    exact_width_um = active_size * amplitude_pitch
    phase_exact_width_px = exact_width_um / phase_pitch
    phase_roi_axis = [
        phase_center[0] - phase_exact_width_px / 2,
        phase_center[0],
        phase_center[0] + phase_exact_width_px / 2,
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "ordinary Fresnel full-square-aperture sinc-cross focus and ROI calibration",
        "config_source": config_path.name,
        "wavelength_nm": wavelength_nm,
        "propagation_cm": propagation_cm,
        "amplitude_polarity": {
            "open_uint8": 255,
            "closed_uint8": 0,
            "all_zero_forbidden": True,
        },
        "amplitude_slm": {
            "size_wh": list(amplitude_size),
            "pixel_pitch_um": amplitude_pitch,
            "center_edge_xy": list(amplitude_center),
        },
        "phase_slm": {
            "size_wh": list(phase_size),
            "pixel_pitch_um": phase_pitch,
            "center_edge_xy": list(phase_center),
            "flip_vertical_on_export": True,
            "flip_horizontal_on_export": False,
            "flip_axis_edge_y": phase_center[1],
            "configured_center_preserved_in_exported_bmp": True,
            "phase_gray_contract": "0..255 represents one wrapped 0..2pi phase period via the laboratory 532nm LUT",
        },
        "roi_geometry": {
            "amplitude_active_size_px": active_size,
            "exact_physical_width_um": exact_width_um,
            "exact_phase_width_px": phase_exact_width_px,
            "phase_axis_targets_before_export_flip": phase_roi_axis,
            "n4_targets": "exact ROI vertices",
            "n9_targets": "vertices, edge midpoints, and center",
        },
        "aperture_sweep": {
            "shape": "full independent square pupils",
            "requested_phase_widths_px": widths,
            "physical_width_is_quantized_on": "17um amplitude SLM",
            "smaller_aperture_effect": "wider sinc-cross PSF with lower throughput",
            "recommended_start": "a64px (large visible cross)",
            "alternatives": {
                "a48px": "widest cross, lowest throughput",
                "a96px": "balanced size and throughput",
                "a128px": "most compact cross, highest throughput",
            },
            "phase_content": "ordinary quadratic Fresnel phase only; no drawn cross, outline, or CGH",
        },
        "ideal_simulation": {
            "background_subtraction": False,
            "camera_is_ideal": True,
            "does_not_model": [
                "measured phase LUT error",
                "SLM fill factor",
                "aberration",
                "misregistration",
                "camera noise or saturation",
            ],
            "metrics_json": {
                "path": metrics_json.relative_to(output_dir).as_posix(),
                "sha256": _sha256(metrics_json),
            },
            "metrics_csv": {
                "path": metrics_csv.relative_to(output_dir).as_posix(),
                "sha256": _sha256(metrics_csv),
            },
        },
        "pairs": pairs,
        "pairing_rule": "play amplitude_file and phase_file from the same pair_id; never substitute all-zero or full-white amplitude",
    }
    manifest_path = output_dir / "fresnel_square_aperture_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    _write_readme(output_dir, manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pair_count": len(pairs),
                "all_numerical_validation_passed": all(
                    item["numerical_validation"]["passed"] for item in pairs
                ),
                "manifest_sha256": _sha256(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    generate(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
