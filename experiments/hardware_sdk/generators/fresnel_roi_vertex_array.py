"""Generate Fresnel calibration masks whose foci lie on the true ROI support points.

This generator intentionally does *not* reuse the historical ``tile-center``
layout.  A 2x2 mask focuses at the four physical support vertices, while a 3x3
mask focuses at the vertices, edge midpoints, and center.  Every phase pixel in
the quantized support belongs to exactly one inward-facing lenslet aperture.
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

from experiments.hardware_sdk.generators.fresnel_phase_array import (
    symmetric_tile_sizes,
)
from experiments.hardware_sdk.workflows.reconstruct_slm import place_at_center


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_bmp(
    array: np.ndarray,
    path: Path,
    expected_size_wh: tuple[int, int],
    *,
    report_root: Path | None = None,
) -> dict[str, Any]:
    value = np.asarray(array, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="L").save(path, format="BMP")
    with Image.open(path) as image:
        if image.format != "BMP" or image.mode != "L" or image.size != expected_size_wh:
            raise RuntimeError(
                f"Invalid BMP {path}: format={image.format}, mode={image.mode}, "
                f"size={image.size}, expected={expected_size_wh}"
            )
    return {
        "path": (
            path.relative_to(report_root).as_posix()
            if report_root is not None
            else path.name
        ),
        "sha256": _sha256(path),
        "size_wh": list(expected_size_wh),
        "mode": "L",
        "min_uint8": int(value.min()),
        "max_uint8": int(value.max()),
    }


def _axis_partition_edges(active_size: int, grid_size: int) -> list[int]:
    edges = [0]
    for size in symmetric_tile_sizes(int(active_size), int(grid_size)):
        edges.append(edges[-1] + size)
    return edges


def exact_support_axis_targets(
    *,
    active_size: int,
    exact_physical_width_px: float,
    grid_size: int,
) -> list[float]:
    """Return continuous target coordinates in local pixel-edge coordinates.

    Pixel ``i`` occupies edge coordinates ``[i, i+1]`` and its center is
    ``i+0.5``.  The physical 478x17 um support is 1015.75 phase pixels wide,
    whereas its raster container is 1016 pixels wide.  Consequently the exact
    physical edges are 0.125 and 1015.875, not the container edges 0 and 1016.
    """

    active_size = int(active_size)
    exact_width = float(exact_physical_width_px)
    grid_size = int(grid_size)
    if grid_size not in {1, 2, 3}:
        raise ValueError("grid_size must be 1, 2, or 3")
    if exact_width <= 0.0 or exact_width > active_size:
        raise ValueError("exact physical width must be in (0, active_size]")
    center = active_size / 2.0
    left = center - exact_width / 2.0
    right = center + exact_width / 2.0
    if grid_size == 1:
        return [center]
    if grid_size == 2:
        return [left, right]
    return [left, center, right]


def fresnel_roi_vertex_phase(
    active_size: int,
    grid_size: int,
    *,
    exact_physical_width_px: float,
    pixel_pitch_um: float,
    wavelength_nm: float,
    propagation_cm: float,
    nyquist_safety_factor: float = 0.92,
    out_of_aperture_mode: str = "flat_zero",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build an inward-aperture Fresnel array with exact physical ROI targets.

    The support is split into equal-area raster cells.  Each cell carries the
    quadratic phase of a lens centered on its assigned support point, rather
    than a lens centered on the cell.  Corner cells therefore contain inward
    quarter-lenses, edge cells inward half-lenses, and the center cell a full
    lens.  Cells cover the support exactly once, but only the circular region
    inside the phase-sampling Nyquist radius carries lens phase.  The remainder
    is explicitly flat phase because a phase-only SLM cannot make it dark.
    """

    active_size = int(active_size)
    grid_size = int(grid_size)
    pitch_m = float(pixel_pitch_um) * 1.0e-6
    wavelength_m = float(wavelength_nm) * 1.0e-9
    distance_m = float(propagation_cm) * 1.0e-2
    safety_factor = float(nyquist_safety_factor)
    if min(active_size, pitch_m, wavelength_m, distance_m) <= 0:
        raise ValueError("size and optical parameters must be positive")
    if not 0.0 < safety_factor <= 1.0:
        raise ValueError("nyquist_safety_factor must be in (0,1]")
    if out_of_aperture_mode != "flat_zero":
        raise ValueError("Only the explicit flat_zero background is supported")

    # A sampled quadratic phase must satisfy |Delta phi| <= pi per axis.
    # The safety factor keeps the circular aperture away from that boundary.
    nyquist_radius_px = wavelength_m * distance_m / (2.0 * pitch_m**2)
    safe_radius_px = safety_factor * nyquist_radius_px

    edges = _axis_partition_edges(active_size, grid_size)
    targets = exact_support_axis_targets(
        active_size=active_size,
        exact_physical_width_px=exact_physical_width_px,
        grid_size=grid_size,
    )
    result = np.zeros((active_size, active_size), dtype=np.uint8)
    lenslets: list[dict[str, Any]] = []

    for row, target_y in enumerate(targets):
        top, bottom = edges[row], edges[row + 1]
        y_m = (np.arange(top, bottom, dtype=np.float64) + 0.5 - target_y) * pitch_m
        for column, target_x in enumerate(targets):
            left, right = edges[column], edges[column + 1]
            x_m = (np.arange(left, right, dtype=np.float64) + 0.5 - target_x) * pitch_m
            yy_m, xx_m = np.meshgrid(y_m, x_m, indexing="ij")
            phase_rad = np.mod(
                -math.pi * (xx_m * xx_m + yy_m * yy_m) / (wavelength_m * distance_m),
                2.0 * math.pi,
            )
            encoded = np.rint(phase_rad * (255.0 / (2.0 * math.pi))).astype(np.uint8)
            radial_squared_px = (xx_m / pitch_m) ** 2 + (yy_m / pitch_m) ** 2
            valid_aperture = radial_squared_px <= safe_radius_px**2
            tile = result[top:bottom, left:right]
            tile[valid_aperture] = encoded[valid_aperture]
            # Explicit flat phase outside the sampling-safe aperture.  This is
            # phase-only hardware, so "outside" cannot be made optically dark.
            tile[~valid_aperture] = 0
            result[top:bottom, left:right] = tile
            active_phase_pixels = int(np.count_nonzero(valid_aperture))
            lenslets.append(
                {
                    "logical_index_row_major": row * grid_size + column,
                    "logical_row": row,
                    "logical_column": column,
                    "logical_aperture_bounds_local_edge_xyxy": [
                        left,
                        top,
                        right,
                        bottom,
                    ],
                    "logical_focus_target_local_edge_xy": [target_x, target_y],
                    "aperture_pixel_count": (right - left) * (bottom - top),
                    "active_lens_phase_pixel_count": active_phase_pixels,
                    "active_lens_phase_fraction_of_cell": active_phase_pixels
                    / ((right - left) * (bottom - top)),
                    "nyquist_radius_phase_px": nyquist_radius_px,
                    "safe_circular_radius_phase_px": safe_radius_px,
                    "nyquist_safety_factor": safety_factor,
                    "outside_safe_aperture_phase_uint8": 0,
                    "outside_safe_aperture_mode": out_of_aperture_mode,
                    "aperture_kind": (
                        "full"
                        if grid_size == 1
                        or (grid_size == 3 and row == 1 and column == 1)
                        else (
                            "inward_quarter"
                            if row in {0, grid_size - 1}
                            and column in {0, grid_size - 1}
                            else "inward_half"
                        )
                    ),
                }
            )
    return result, lenslets


def angular_spectrum_focus_validation(
    phase_uint8: np.ndarray,
    *,
    target_axis_edge_coordinates: list[float],
    pixel_pitch_um: float,
    wavelength_nm: float,
    propagation_cm: float,
    search_radius_px: int = 8,
    pad_size: int | None = None,
    target_exclusion_radius_px: int = 12,
    min_peak_to_global_median: float = 100.0,
    min_peak_to_max_outside_targets: float = 50.0,
) -> dict[str, Any]:
    """Numerically propagate a quantized phase mask and locate every focus.

    The input amplitude is one only inside ``phase_uint8`` and zero outside,
    matching the central 478-pixel white-window calibration frame.  Padding is
    required because support-edge foci sit at the edge of the illuminated ROI.
    """

    phase = np.asarray(phase_uint8, dtype=np.uint8)
    if phase.ndim != 2 or phase.shape[0] != phase.shape[1]:
        raise ValueError("phase_uint8 must be a square 2-D array")
    active_size = int(phase.shape[0])
    minimum_pad = 2 * active_size
    if pad_size is None:
        pad_size = 1 << int(math.ceil(math.log2(minimum_pad)))
    pad_size = int(pad_size)
    if pad_size < minimum_pad:
        raise ValueError(f"pad_size must be at least {minimum_pad}")

    pitch_m = float(pixel_pitch_um) * 1.0e-6
    wavelength_m = float(wavelength_nm) * 1.0e-9
    distance_m = float(propagation_cm) * 1.0e-2
    field = np.zeros((pad_size, pad_size), dtype=np.complex64)
    offset = (pad_size - active_size) // 2
    phase_rad = phase.astype(np.float32) * np.float32(2.0 * math.pi / 255.0)
    field[offset : offset + active_size, offset : offset + active_size] = np.exp(
        1j * phase_rad
    ).astype(np.complex64)

    frequencies = np.fft.fftfreq(pad_size, d=pitch_m)
    fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
    normalized_squared = (wavelength_m * fx) ** 2 + (wavelength_m * fy) ** 2
    transfer = np.exp(
        1j
        * (2.0 * math.pi / wavelength_m)
        * distance_m
        * np.sqrt(np.maximum(0.0, 1.0 - normalized_squared))
    )
    propagated = np.fft.ifft2(np.fft.fft2(field) * transfer)
    intensity = np.abs(propagated) ** 2
    background_median = float(np.median(intensity))

    peaks: list[dict[str, Any]] = []
    max_abs_error = 0.0
    target_neighborhood = np.zeros(intensity.shape, dtype=bool)
    for row, target_y in enumerate(target_axis_edge_coordinates):
        for column, target_x in enumerate(target_axis_edge_coordinates):
            expected_x = offset + float(target_x)
            expected_y = offset + float(target_y)
            nearest_x = int(math.floor(expected_x - 0.5))
            nearest_y = int(math.floor(expected_y - 0.5))
            left = max(0, nearest_x - search_radius_px)
            right = min(pad_size, nearest_x + search_radius_px + 2)
            top = max(0, nearest_y - search_radius_px)
            bottom = min(pad_size, nearest_y + search_radius_px + 2)
            patch = intensity[top:bottom, left:right]
            local_y, local_x = np.unravel_index(int(np.argmax(patch)), patch.shape)
            peak_x_index = left + int(local_x)
            peak_y_index = top + int(local_y)
            peak_x_edge = peak_x_index - offset + 0.5
            peak_y_edge = peak_y_index - offset + 0.5
            error_x = peak_x_edge - float(target_x)
            error_y = peak_y_edge - float(target_y)
            max_abs_error = max(max_abs_error, abs(error_x), abs(error_y))
            peak_value = float(intensity[peak_y_index, peak_x_index])
            peaks.append(
                {
                    "logical_row": row,
                    "logical_column": column,
                    "target_local_edge_xy": [float(target_x), float(target_y)],
                    "peak_local_pixel_index_xy": [
                        peak_x_index - offset,
                        peak_y_index - offset,
                    ],
                    "peak_local_pixel_center_edge_xy": [
                        peak_x_edge,
                        peak_y_edge,
                    ],
                    "error_phase_pixels_xy": [error_x, error_y],
                    "peak_intensity": peak_value,
                    "peak_to_global_median": peak_value
                    / max(background_median, 1.0e-30),
                }
            )

            mask_radius = int(target_exclusion_radius_px)
            mask_left = max(0, int(math.floor(expected_x)) - mask_radius)
            mask_right = min(pad_size, int(math.ceil(expected_x)) + mask_radius + 1)
            mask_top = max(0, int(math.floor(expected_y)) - mask_radius)
            mask_bottom = min(pad_size, int(math.ceil(expected_y)) + mask_radius + 1)
            mask_y, mask_x = np.ogrid[mask_top:mask_bottom, mask_left:mask_right]
            local_disk = (mask_x + 0.5 - expected_x) ** 2 + (
                mask_y + 0.5 - expected_y
            ) ** 2 <= mask_radius**2
            target_neighborhood[
                mask_top:mask_bottom, mask_left:mask_right
            ] |= local_disk

    peak_indexes = {tuple(item["peak_local_pixel_index_xy"]) for item in peaks}
    unique_peak_assignment = len(peak_indexes) == len(peaks)
    minimum_peak = min(float(item["peak_intensity"]) for item in peaks)
    maximum_outside = float(np.max(intensity[~target_neighborhood]))
    minimum_peak_to_median = minimum_peak / max(background_median, 1.0e-30)
    minimum_peak_to_outside = minimum_peak / max(maximum_outside, 1.0e-30)
    passed = (
        max_abs_error <= 0.75
        and unique_peak_assignment
        and minimum_peak_to_median >= float(min_peak_to_global_median)
        and minimum_peak_to_outside >= float(min_peak_to_max_outside_targets)
    )
    return {
        "method": "zero-padded angular spectrum propagation of quantized uint8 phase",
        "pad_size": pad_size,
        "search_radius_px": int(search_radius_px),
        "propagation_cm": float(propagation_cm),
        "max_abs_position_error_phase_px": max_abs_error,
        "acceptance_max_abs_error_phase_px": 0.75,
        "target_exclusion_radius_px": int(target_exclusion_radius_px),
        "unique_peak_assignment": unique_peak_assignment,
        "minimum_target_peak_to_global_median": minimum_peak_to_median,
        "acceptance_minimum_target_peak_to_global_median": float(
            min_peak_to_global_median
        ),
        "minimum_target_peak_to_max_outside_targets": minimum_peak_to_outside,
        "acceptance_minimum_target_peak_to_max_outside_targets": float(
            min_peak_to_max_outside_targets
        ),
        "maximum_intensity_outside_target_neighborhoods": maximum_outside,
        "passed": passed,
        "peaks": peaks,
    }


def _focal_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _layout_name(grid_size: int) -> str:
    return {
        1: "center",
        2: "exact_roi_vertices",
        3: "exact_roi_vertices_edge_midpoints_center",
    }[int(grid_size)]


def _write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    support = manifest["shared_optical_support"]
    output_dir.joinpath("README.md").write_text(
        f"""# ROI 顶点菲涅尔标定阵列（修正版）

本目录是全新修正版，不覆盖历史的“分区中心”阵列。历史四点间距为508个相位像素，
不能直接当作ROI四个顶点；本目录四点目标间距为精确物理宽度
`{support['exact_phase_support_width_px']}`个相位像素。

## 几何约定

- 振幅有效输入：478×478，17 µm，物理宽度8126 µm。
- 相位SLM：1920×1200，8 µm，中心采用像素边界坐标(980,590)。
- 8126/8=1015.75，因此精确物理边界是
  `[472.125,82.125] → [1487.875,1097.875]`。
- 相位承载栅格量化为1016×1016，半开边界为
  `[472,82,1488,1098)`；它比精确物理宽度多2 µm。
- BMP像素索引(x,y)的中心坐标是(x+0.5,y+0.5)。精确焦点可位于像素中心之间，
  因此单个最亮像素允许与连续目标相差不超过0.5个像素；应使用光斑质心做亚像素定位。
- BMP导出前已经执行纵向翻转，禁止播放端再次翻转。

## 文件用途

- `amplitude_bmp/amplitude_focus_full_white_1024x1024.bmp`：仅建议配合单透镜寻找焦面。
- `amplitude_bmp/amplitude_roi478_white_black_1024x1024.bmp`：精确ROI标定推荐输入；
  中央478×478为255，外围为0，无缩放、无插值。
- `phase_bmp/*n1*`：单个中心焦点，用于寻找5/10/15 cm焦面。
- `phase_bmp/*n4*`：焦点直接位于精确ROI四个物理顶点，不需要再外推半个间距。
- `phase_bmp/*n9*`：四角、四个边中点和中心，用于检查旋转、剪切和非线性畸变。

相位不是简单移动旧透镜坐标：有效支撑被完整分区，每个子孔径使用以指定ROI点为
二次相位中心的向内透镜。四角是向内quarter-lens，边中点是向内half-lens，中心为
full-lens。为避免二次相位采样混叠，实际透镜相位还会与圆形安全孔径相交：
`r_safe=0.92*lambda*z/(2*p^2)`，5/10/15 cm分别约为191.2/382.4/573.6个相位像素。
安全圆外明确写0相位；这是平相位而不是遮光，因此四点和九点标定应配合中央478白窗
振幅图使用。

`numerical_focus_validation.json/csv` 使用实际量化后的8-bit相位做零填充角谱传播。
生成命令只有在每个焦点的位置误差不超过0.75个相位像素、焦点一一对应、最弱目标峰
至少是全局背景中位数100倍且至少是所有目标邻域外最强伪峰50倍时才成功。

推荐顺序：全白振幅+单透镜找焦面；随后切换中央478白窗，播放同一距离的四点和九点
阵列。四点/九点阵列本身对称，翻转身份应结合已知BMP纵翻约定或另行使用非对称图案
确认，不能仅凭四个等强光点判断。
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
    distances_cm = [float(value) for value in raw["propagation_distances_cm"]]
    amplitude = raw["amplitude_slm"]
    phase = raw["phase_slm"]
    shared = raw["shared_optical_support"]
    validation_config = raw.get("numerical_validation", {})
    phase_aperture = raw.get("phase_aperture", {})

    amplitude_size = tuple(int(value) for value in amplitude["size_wh"])
    amplitude_center = tuple(float(value) for value in amplitude["center_xy"])
    amplitude_pitch_um = float(amplitude["pixel_pitch_um"])
    amplitude_active_size = int(shared["amplitude_active_size_px"])
    phase_size = tuple(int(value) for value in phase["size_wh"])
    phase_center = tuple(float(value) for value in phase["center_xy"])
    phase_pitch_um = float(phase["pixel_pitch_um"])
    flip_vertical = bool(phase.get("flip_vertical", True))
    flip_horizontal = bool(phase.get("flip_horizontal", False))
    bright_value = int(amplitude.get("bright_value_uint8", 255))
    dark_value = int(amplitude.get("dark_value_uint8", 0))
    if (bright_value, dark_value) != (255, 0):
        raise ValueError("Corrected amplitude contract requires bright=255 and dark=0")

    physical_width_um = amplitude_active_size * amplitude_pitch_um
    exact_phase_width_px = physical_width_um / phase_pitch_um
    native_active_size = int(round(exact_phase_width_px))
    exact_local_left = native_active_size / 2.0 - exact_phase_width_px / 2.0
    exact_local_right = native_active_size / 2.0 + exact_phase_width_px / 2.0

    full_white = np.full(
        (amplitude_size[1], amplitude_size[0]), bright_value, dtype=np.uint8
    )
    full_white_report = _save_bmp(
        full_white,
        output_dir / "amplitude_bmp" / "amplitude_focus_full_white_1024x1024.bmp",
        amplitude_size,
        report_root=output_dir,
    )
    active_white = Image.fromarray(
        np.full(
            (amplitude_active_size, amplitude_active_size),
            bright_value,
            dtype=np.uint8,
        ),
        mode="L",
    )
    amplitude_window, amplitude_bounds, actual_amplitude_center = place_at_center(
        active_white,
        slm_size_wh=amplitude_size,
        center_xy=amplitude_center,
    )
    amplitude_window_report = _save_bmp(
        np.asarray(amplitude_window, dtype=np.uint8),
        output_dir / "amplitude_bmp" / "amplitude_roi478_white_black_1024x1024.bmp",
        amplitude_size,
        report_root=output_dir,
    )

    blank_active = Image.fromarray(
        np.zeros((native_active_size, native_active_size), dtype=np.uint8), mode="L"
    )
    _, phase_bounds, actual_phase_center = place_at_center(
        blank_active,
        slm_size_wh=phase_size,
        center_xy=phase_center,
    )
    phase_left, phase_top, _, _ = phase_bounds

    phase_reports: list[dict[str, Any]] = []
    validation_reports: list[dict[str, Any]] = []
    center_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for propagation_cm in distances_cm:
        for grid_size in (1, 2, 3):
            active_phase, lenslets = fresnel_roi_vertex_phase(
                native_active_size,
                grid_size,
                exact_physical_width_px=exact_phase_width_px,
                pixel_pitch_um=phase_pitch_um,
                wavelength_nm=wavelength_nm,
                propagation_cm=propagation_cm,
                nyquist_safety_factor=float(
                    phase_aperture.get("nyquist_safety_factor", 0.92)
                ),
                out_of_aperture_mode=str(
                    phase_aperture.get("out_of_aperture_mode", "flat_zero")
                ),
            )
            targets = exact_support_axis_targets(
                active_size=native_active_size,
                exact_physical_width_px=exact_phase_width_px,
                grid_size=grid_size,
            )
            validation = angular_spectrum_focus_validation(
                active_phase,
                target_axis_edge_coordinates=targets,
                pixel_pitch_um=phase_pitch_um,
                wavelength_nm=wavelength_nm,
                propagation_cm=propagation_cm,
                search_radius_px=int(validation_config.get("search_radius_px", 8)),
                pad_size=(
                    None
                    if validation_config.get("pad_size") is None
                    else int(validation_config["pad_size"])
                ),
                target_exclusion_radius_px=int(
                    validation_config.get("target_exclusion_radius_px", 12)
                ),
                min_peak_to_global_median=float(
                    validation_config.get("min_peak_to_global_median", 100.0)
                ),
                min_peak_to_max_outside_targets=float(
                    validation_config.get("min_peak_to_max_outside_targets", 50.0)
                ),
            )
            if not validation["passed"]:
                raise RuntimeError(
                    f"Numerical focus validation failed for n{grid_size**2}, "
                    f"z={propagation_cm:g} cm: {validation}"
                )

            exported_active = active_phase
            if flip_vertical:
                exported_active = np.flipud(exported_active)
            if flip_horizontal:
                exported_active = np.fliplr(exported_active)
            exported_active = np.ascontiguousarray(exported_active)
            phase_canvas, exported_bounds, exported_center = place_at_center(
                Image.fromarray(exported_active, mode="L"),
                slm_size_wh=phase_size,
                center_xy=phase_center,
            )
            layout = _layout_name(grid_size)
            filename = (
                f"phase_fresnel_n{grid_size**2}_{layout}_"
                f"z{_focal_tag(propagation_cm)}cm_532nm_8um_1920x1200.bmp"
            )
            file_report = _save_bmp(
                np.asarray(phase_canvas, dtype=np.uint8),
                output_dir / "phase_bmp" / filename,
                phase_size,
                report_root=output_dir,
            )

            global_lenslets: list[dict[str, Any]] = []
            for lenslet in lenslets:
                item = dict(lenslet)
                target_x, target_y = item["logical_focus_target_local_edge_xy"]
                left, top, right, bottom = item[
                    "logical_aperture_bounds_local_edge_xyxy"
                ]
                exported_target_x = (
                    native_active_size - target_x if flip_horizontal else target_x
                )
                exported_target_y = (
                    native_active_size - target_y if flip_vertical else target_y
                )
                exported_left, exported_right = (
                    (native_active_size - right, native_active_size - left)
                    if flip_horizontal
                    else (left, right)
                )
                exported_top, exported_bottom = (
                    (native_active_size - bottom, native_active_size - top)
                    if flip_vertical
                    else (top, bottom)
                )
                item.update(
                    {
                        "logical_focus_target_phase_edge_xy": [
                            phase_left + target_x,
                            phase_top + target_y,
                        ],
                        "exported_focus_target_bmp_edge_xy": [
                            phase_left + exported_target_x,
                            phase_top + exported_target_y,
                        ],
                        "exported_aperture_bounds_bmp_edge_xyxy": [
                            phase_left + exported_left,
                            phase_top + exported_top,
                            phase_left + exported_right,
                            phase_top + exported_bottom,
                        ],
                    }
                )
                global_lenslets.append(item)
                center_rows.append(
                    {
                        "phase_bmp": filename,
                        "array_count": grid_size**2,
                        "propagation_cm": propagation_cm,
                        "logical_row": item["logical_row"],
                        "logical_column": item["logical_column"],
                        "logical_target_phase_edge_x": item[
                            "logical_focus_target_phase_edge_xy"
                        ][0],
                        "logical_target_phase_edge_y": item[
                            "logical_focus_target_phase_edge_xy"
                        ][1],
                        "exported_target_bmp_edge_x": item[
                            "exported_focus_target_bmp_edge_xy"
                        ][0],
                        "exported_target_bmp_edge_y": item[
                            "exported_focus_target_bmp_edge_xy"
                        ][1],
                        "aperture_kind": item["aperture_kind"],
                        "phase_sha256": file_report["sha256"],
                    }
                )

            validation.update(
                {
                    "phase_bmp": filename,
                    "array_count": grid_size**2,
                    "grid_size": grid_size,
                    "phase_sha256": file_report["sha256"],
                    "coordinate_frame": "logical active support before export flips",
                }
            )
            validation_reports.append(validation)
            for peak in validation["peaks"]:
                validation_rows.append(
                    {
                        "phase_bmp": filename,
                        "array_count": grid_size**2,
                        "propagation_cm": propagation_cm,
                        "logical_row": peak["logical_row"],
                        "logical_column": peak["logical_column"],
                        "target_local_edge_x": peak["target_local_edge_xy"][0],
                        "target_local_edge_y": peak["target_local_edge_xy"][1],
                        "peak_center_edge_x": peak["peak_local_pixel_center_edge_xy"][
                            0
                        ],
                        "peak_center_edge_y": peak["peak_local_pixel_center_edge_xy"][
                            1
                        ],
                        "error_phase_px_x": peak["error_phase_pixels_xy"][0],
                        "error_phase_px_y": peak["error_phase_pixels_xy"][1],
                        "peak_to_global_median": peak["peak_to_global_median"],
                        "run_min_peak_to_max_outside_targets": validation[
                            "minimum_target_peak_to_max_outside_targets"
                        ],
                        "unique_peak_assignment": validation["unique_peak_assignment"],
                        "passed": validation["passed"],
                    }
                )

            phase_reports.append(
                {
                    "array_count": grid_size**2,
                    "grid_size": grid_size,
                    "layout": layout,
                    "propagation_cm": propagation_cm,
                    "file": file_report,
                    "phase_active_bounds_bmp_edge_xyxy": list(exported_bounds),
                    "phase_active_center_bmp_edge_xy": list(exported_center),
                    "lenslets": global_lenslets,
                    "numerical_validation_passed": validation["passed"],
                }
            )

            preview = Image.fromarray(exported_active, mode="L")
            preview.thumbnail((508, 508), Image.Resampling.NEAREST)
            preview.save(preview_dir / filename.replace(".bmp", ".png"), format="PNG")

    with (output_dir / "fresnel_focus_targets.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(center_rows[0]))
        writer.writeheader()
        writer.writerows(center_rows)
    with (output_dir / "numerical_focus_validation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)
    (output_dir / "numerical_focus_validation.json").write_text(
        json.dumps(validation_reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    exact_bounds = [
        phase_left + exact_local_left,
        phase_top + exact_local_left,
        phase_left + exact_local_right,
        phase_top + exact_local_right,
    ]
    manifest = {
        "schema_version": 2,
        "purpose": "exact physical ROI-vertex Fresnel focus/ROI calibration",
        "supersedes_for_roi_vertices": (
            "The historical tile-center arrays remain archived but their n4 spacing "
            "is only 508 phase pixels and must not be interpreted as ROI vertices."
        ),
        "wavelength_nm": wavelength_nm,
        "propagation_distances_cm": distances_cm,
        "phase_formula": (
            "per aperture: phi=mod(-pi*((x-x_target)^2+(y-y_target)^2)/(lambda*z),2*pi)"
        ),
        "phase_encoding": "round(phi*255/(2*pi)) as uint8",
        "coordinate_convention": {
            "frame": "continuous pixel-edge coordinates",
            "pixel_i_support": "[i,i+1]",
            "pixel_i_center": "i+0.5",
            "target_note": (
                "Exact physical support edges may lie between phase pixel centers; "
                "use spot centroid for subpixel calibration."
            ),
            "logical_vs_exported": (
                "logical coordinates precede flips; exported coordinates are the actual BMP"
            ),
        },
        "amplitude": {
            "polarity": "255=white/bright/transmissive; 0=black/dark/blocked",
            "focus_full_white": full_white_report,
            "roi_window": {
                **amplitude_window_report,
                "active_bounds_edge_xyxy": list(amplitude_bounds),
                "active_center_edge_xy": list(actual_amplitude_center),
                "active_size_px": amplitude_active_size,
            },
        },
        "shared_optical_support": {
            "amplitude_active_size_px": amplitude_active_size,
            "amplitude_pixel_pitch_um": amplitude_pitch_um,
            "exact_physical_width_um": physical_width_um,
            "phase_pixel_pitch_um": phase_pitch_um,
            "exact_phase_support_width_px": exact_phase_width_px,
            "exact_phase_support_bounds_edge_xyxy": exact_bounds,
            "quantized_phase_active_size_px": native_active_size,
            "quantized_phase_active_bounds_half_open_xyxy": list(phase_bounds),
            "quantized_phase_physical_width_um": native_active_size * phase_pitch_um,
            "quantization_width_error_um": (
                native_active_size * phase_pitch_um - physical_width_um
            ),
            "phase_center_edge_xy": list(actual_phase_center),
        },
        "phase_slm": {
            "size_wh": list(phase_size),
            "requested_center_edge_xy": list(phase_center),
            "flip_vertical_before_export": flip_vertical,
            "flip_horizontal_before_export": flip_horizontal,
        },
        "array_design": {
            "n1": "one sampling-safe circular lens focused at the shared center",
            "n4": (
                "four inward, sampling-safe quarter-lenses focused at exact "
                "physical ROI vertices"
            ),
            "n9": (
                "equal-area support partition; foci at four vertices, four edge "
                "midpoints, and center"
            ),
            "cell_partition": (
                "every quantized support pixel belongs to exactly one geometric cell"
            ),
            "sampling_safe_aperture": {
                "nyquist_limit_phase_px": "lambda*z/(2*p^2)",
                "safety_factor": float(
                    phase_aperture.get("nyquist_safety_factor", 0.92)
                ),
                "shape": "circle intersected with each inward cell",
                "outside_mode": str(
                    phase_aperture.get("out_of_aperture_mode", "flat_zero")
                ),
                "outside_phase_uint8": 0,
                "phase_only_warning": (
                    "outside pixels are flat phase, not optically dark; use the "
                    "478 white-window amplitude BMP to limit illumination"
                ),
            },
        },
        "numerical_validation": {
            "file_json": "numerical_focus_validation.json",
            "file_csv": "numerical_focus_validation.csv",
            "all_passed": all(item["passed"] for item in validation_reports),
            "maximum_error_phase_px": max(
                item["max_abs_position_error_phase_px"] for item in validation_reports
            ),
        },
        "files": phase_reports,
        "background_subtraction": False,
    }
    (output_dir / "fresnel_roi_vertex_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    _write_readme(output_dir, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = generate(Path(args.config))
    print(
        f"Generated 2 amplitude BMPs and {len(report['files'])} phase BMPs; "
        f"numerical_validation={report['numerical_validation']['all_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
