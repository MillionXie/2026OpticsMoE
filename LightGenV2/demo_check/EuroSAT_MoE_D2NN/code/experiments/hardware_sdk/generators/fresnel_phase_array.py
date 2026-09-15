"""Generate 1/4/9-element Fresnel phase arrays for dual-SLM ROI registration."""

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

from experiments.hardware_sdk.workflows.reconstruct_slm import place_at_center


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symmetric_tile_sizes(total: int, grid_size: int) -> list[int]:
    """Split an integer support symmetrically without shifting its center."""

    total, grid_size = int(total), int(grid_size)
    if total <= 0 or grid_size not in {1, 2, 3}:
        raise ValueError("total must be positive and grid_size must be 1, 2, or 3")
    if grid_size == 1:
        return [total]
    if grid_size == 2:
        if total % 2:
            raise ValueError("2x2 array requires an even active size")
        return [total // 2, total // 2]
    edge = int(math.ceil(total / 3.0))
    middle = total - 2 * edge
    if middle <= 0:
        raise ValueError("active size is too small for a 3x3 array")
    return [edge, middle, edge]


def tile_bounds_and_centers(
    active_size: int,
    grid_size: int,
    *,
    active_origin_xy: tuple[int, int] = (0, 0),
) -> tuple[list[tuple[int, int, int, int]], list[tuple[float, float]]]:
    """Return half-open pixel bounds and edge-coordinate centers in row-major order."""

    sizes = symmetric_tile_sizes(active_size, grid_size)
    edges = [0]
    for size in sizes:
        edges.append(edges[-1] + size)
    origin_x, origin_y = map(int, active_origin_xy)
    bounds: list[tuple[int, int, int, int]] = []
    centers: list[tuple[float, float]] = []
    for row in range(grid_size):
        for column in range(grid_size):
            left, right = edges[column], edges[column + 1]
            top, bottom = edges[row], edges[row + 1]
            bounds.append(
                (origin_x + left, origin_y + top, origin_x + right, origin_y + bottom)
            )
            centers.append(
                (
                    origin_x + (left + right) / 2.0,
                    origin_y + (top + bottom) / 2.0,
                )
            )
    return bounds, centers


def fresnel_lens_phase(
    height: int,
    width: int,
    *,
    pixel_pitch_um: float,
    wavelength_nm: float,
    propagation_cm: float,
    aperture_fraction: float = 1.0,
) -> np.ndarray:
    """Return an 8-bit paraxial Fresnel lens phase on one rectangular tile.

    The implemented phase is ``mod(-pi*(x^2+y^2)/(lambda*z), 2*pi)`` and uses
    the same sign/quantization convention as the repository's established
    single-lens calibration generator.
    """

    height, width = int(height), int(width)
    pitch = float(pixel_pitch_um) * 1.0e-6
    wavelength = float(wavelength_nm) * 1.0e-9
    distance = float(propagation_cm) * 1.0e-2
    aperture_fraction = float(aperture_fraction)
    if min(height, width) <= 0 or min(pitch, wavelength, distance) <= 0.0:
        raise ValueError("lens dimensions and physical parameters must be positive")
    if not 0.0 < aperture_fraction <= 1.0:
        raise ValueError("aperture_fraction must be in (0,1]")
    x = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * pitch
    y = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) * pitch
    yy, xx = np.meshgrid(y, x, indexing="ij")
    phase = np.mod(
        -math.pi * (xx * xx + yy * yy) / (wavelength * distance),
        2.0 * math.pi,
    )
    encoded = np.rint(phase * (255.0 / (2.0 * math.pi))).astype(np.uint8)
    if aperture_fraction < 1.0:
        aperture_width = max(1, int(round(width * aperture_fraction)))
        aperture_height = max(1, int(round(height * aperture_fraction)))
        x0 = (width - aperture_width) // 2
        y0 = (height - aperture_height) // 2
        mask = np.zeros((height, width), dtype=bool)
        mask[y0 : y0 + aperture_height, x0 : x0 + aperture_width] = True
        encoded = np.where(mask, encoded, 0).astype(np.uint8)
    return encoded


def fresnel_phase_array(
    active_size: int,
    grid_size: int,
    *,
    pixel_pitch_um: float,
    wavelength_nm: float,
    propagation_cm: float,
    orientation_coded: bool,
    marker_aperture_fraction: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Tile one Fresnel lens per array cell and optionally weaken logical TL."""

    bounds, centers = tile_bounds_and_centers(active_size, grid_size)
    result = np.zeros((active_size, active_size), dtype=np.uint8)
    lenslets: list[dict[str, Any]] = []
    for index, ((left, top, right, bottom), center) in enumerate(
        zip(bounds, centers)
    ):
        aperture_fraction = (
            marker_aperture_fraction
            if orientation_coded and grid_size > 1 and index == 0
            else 1.0
        )
        tile = fresnel_lens_phase(
            bottom - top,
            right - left,
            pixel_pitch_um=pixel_pitch_um,
            wavelength_nm=wavelength_nm,
            propagation_cm=propagation_cm,
            aperture_fraction=aperture_fraction,
        )
        result[top:bottom, left:right] = tile
        lenslets.append(
            {
                "logical_index_row_major": index,
                "logical_row": index // grid_size,
                "logical_column": index % grid_size,
                "active_bounds_xyxy": [left, top, right, bottom],
                "active_center_edge_xy": list(center),
                "phase_pixel_index_center_xy": [center[0] - 0.5, center[1] - 0.5],
                "aperture_fraction": aperture_fraction,
                "orientation_marker": bool(
                    orientation_coded and grid_size > 1 and index == 0
                ),
            }
        )
    return result, lenslets


def _save_bmp(array: np.ndarray, path: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(path, format="BMP")
    with Image.open(path) as image:
        if image.format != "BMP" or image.mode != "L" or image.size != expected_size:
            raise RuntimeError(
                f"Invalid BMP {path}: format={image.format} mode={image.mode} "
                f"size={image.size}, expected={expected_size}"
            )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_wh": list(expected_size),
        "mode": "L",
        "min_uint8": int(array.min()),
        "max_uint8": int(array.max()),
    }


def _focal_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def generate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(raw["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()

    wavelength_nm = float(raw["wavelength_nm"])
    propagation_distances_cm = [
        float(value) for value in raw["propagation_distances_cm"]
    ]
    amplitude = raw["amplitude_slm"]
    phase = raw["phase_slm"]
    shared = raw["shared_optical_support"]
    amplitude_size = tuple(int(value) for value in amplitude["size_wh"])
    phase_size = tuple(int(value) for value in phase["size_wh"])
    phase_center = tuple(float(value) for value in phase["center_xy"])
    phase_pitch_um = float(phase["pixel_pitch_um"])
    flip_vertical = bool(phase.get("flip_vertical", True))
    flip_horizontal = bool(phase.get("flip_horizontal", False))
    amplitude_active_size = int(shared["amplitude_active_size_px"])
    amplitude_pitch_um = float(amplitude["pixel_pitch_um"])
    physical_width_um = amplitude_active_size * amplitude_pitch_um
    native_active_size = int(round(physical_width_um / phase_pitch_um))
    marker_fraction = float(raw["orientation_marker_aperture_fraction"])

    bright_value = int(amplitude.get("bright_value_uint8", 255))
    dark_value = int(amplitude.get("dark_value_uint8", 0))
    if (bright_value, dark_value) != (255, 0):
        raise ValueError(
            "The corrected amplitude-SLM contract requires bright=255 and dark=0"
        )
    amplitude_uniform = np.full(
        (amplitude_size[1], amplitude_size[0]), bright_value, dtype=np.uint8
    )
    amplitude_report = _save_bmp(
        amplitude_uniform,
        output_dir / "amplitude_bmp" / "amplitude_uniform_white_1024x1024.bmp",
        amplitude_size,
    )
    amplitude_report.update(
        {
            "optical_role": "uniform bright illumination",
            "bright_value_uint8": bright_value,
            "dark_value_uint8": dark_value,
            "black_white_inverted": False,
        }
    )

    # Determine exact placement once. All phase arrays share these bounds.
    blank_active = np.zeros((native_active_size, native_active_size), dtype=np.uint8)
    _, active_bounds, actual_phase_center = place_at_center(
        Image.fromarray(blank_active, mode="L"),
        slm_size_wh=phase_size,
        center_xy=phase_center,
    )
    active_left, active_top, _, _ = active_bounds

    phase_reports: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    (output_dir / "preview").mkdir(parents=True, exist_ok=True)
    for grid_size in (1, 2, 3):
        variants = ["uniform"] if grid_size == 1 else ["uniform", "flip_coded"]
        for variant in variants:
            orientation_coded = variant == "flip_coded"
            for propagation_cm in propagation_distances_cm:
                active_phase, lenslets = fresnel_phase_array(
                    native_active_size,
                    grid_size,
                    pixel_pitch_um=phase_pitch_um,
                    wavelength_nm=wavelength_nm,
                    propagation_cm=propagation_cm,
                    orientation_coded=orientation_coded,
                    marker_aperture_fraction=marker_fraction,
                )
                exported_active = active_phase
                if flip_vertical:
                    exported_active = np.flipud(exported_active)
                if flip_horizontal:
                    exported_active = np.fliplr(exported_active)
                exported_active = np.ascontiguousarray(exported_active)
                canvas, bounds, actual_center = place_at_center(
                    Image.fromarray(exported_active, mode="L"),
                    slm_size_wh=phase_size,
                    center_xy=phase_center,
                )
                array_count = grid_size * grid_size
                filename = (
                    f"phase_fresnel_n{array_count}_{grid_size}x{grid_size}_"
                    f"{variant}_z{_focal_tag(propagation_cm)}cm_"
                    "532nm_8um_1920x1200.bmp"
                )
                file_report = _save_bmp(
                    np.asarray(canvas, dtype=np.uint8),
                    output_dir / "phase_bmp" / filename,
                    phase_size,
                )

                global_lenslets: list[dict[str, Any]] = []
                for item in lenslets:
                    cx, cy = item["active_center_edge_xy"]
                    exported_cx = native_active_size - cx if flip_horizontal else cx
                    exported_cy = native_active_size - cy if flip_vertical else cy
                    global_item = dict(item)
                    global_item["logical_phase_center_edge_xy"] = [
                        active_left + cx,
                        active_top + cy,
                    ]
                    global_item["exported_bmp_center_edge_xy"] = [
                        active_left + exported_cx,
                        active_top + exported_cy,
                    ]
                    global_item["exported_bmp_pixel_index_center_xy"] = [
                        active_left + exported_cx - 0.5,
                        active_top + exported_cy - 0.5,
                    ]
                    global_lenslets.append(global_item)

                report = {
                    "array_count": array_count,
                    "grid_size": grid_size,
                    "variant": variant,
                    "propagation_cm": propagation_cm,
                    "file": file_report,
                    "phase_active_bounds_xyxy": list(bounds),
                    "phase_active_center_edge_xy": list(actual_center),
                    "lenslets": global_lenslets,
                }
                phase_reports.append(report)
                for item in global_lenslets:
                    csv_rows.append(
                        {
                            "phase_bmp": filename,
                            "array_count": array_count,
                            "grid_size": grid_size,
                            "variant": variant,
                            "propagation_cm": propagation_cm,
                            "lens_index": item["logical_index_row_major"],
                            "logical_row": item["logical_row"],
                            "logical_column": item["logical_column"],
                            "logical_phase_center_edge_x": item[
                                "logical_phase_center_edge_xy"
                            ][0],
                            "logical_phase_center_edge_y": item[
                                "logical_phase_center_edge_xy"
                            ][1],
                            "exported_bmp_center_edge_x": item[
                                "exported_bmp_center_edge_xy"
                            ][0],
                            "exported_bmp_center_edge_y": item[
                                "exported_bmp_center_edge_xy"
                            ][1],
                            "aperture_fraction": item["aperture_fraction"],
                            "orientation_marker": item["orientation_marker"],
                            "phase_sha256": file_report["sha256"],
                        }
                    )

                preview = Image.fromarray(exported_active, mode="L")
                preview.thumbnail((508, 508), Image.Resampling.NEAREST)
                preview.save(
                    output_dir / "preview" / filename.replace(".bmp", ".png"),
                    format="PNG",
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fresnel_lens_centers.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    report = {
        "schema_version": 1,
        "purpose": "Fresnel phase arrays for focus, flip identification, and CCD ROI calibration",
        "wavelength_nm": wavelength_nm,
        "propagation_distances_cm": propagation_distances_cm,
        "phase_formula": "phi=mod(-pi*(x^2+y^2)/(lambda*z), 2*pi)",
        "phase_encoding": "round(phi*255/(2*pi)) as uint8",
        "amplitude": amplitude_report,
        "shared_optical_support": {
            "amplitude_active_size_px": amplitude_active_size,
            "amplitude_pixel_pitch_um": amplitude_pitch_um,
            "physical_width_um": physical_width_um,
            "phase_pixel_pitch_um": phase_pitch_um,
            "ideal_phase_width_px": physical_width_um / phase_pitch_um,
            "quantized_phase_active_size_px": native_active_size,
            "phase_active_bounds_xyxy": list(active_bounds),
            "phase_center_edge_xy": list(actual_phase_center),
            "quantized_physical_width_um": native_active_size * phase_pitch_um,
            "physical_width_quantization_error_um": (
                native_active_size * phase_pitch_um - physical_width_um
            ),
        },
        "phase_slm": {
            "size_wh": list(phase_size),
            "flip_vertical_before_export": flip_vertical,
            "flip_horizontal_before_export": flip_horizontal,
        },
        "orientation_coding": {
            "uniform": "equal full-tile apertures; use for ROI fitting",
            "flip_coded": (
                "logical top-left lens uses a reduced square phase aperture; "
                "spot center is unchanged but its focus is weaker"
            ),
            "marker_aperture_fraction": marker_fraction,
            "warning": "uniform 2x2/3x3 arrays alone are symmetric and cannot identify flips",
        },
        "roi_protocol": {
            "recommended_file": "n4 2x2 uniform at the best-focused propagation distance",
            "axis_aligned_bounds_from_four_spots": (
                "left=x_left-dx/2; right=x_right+dx/2; "
                "top=y_top-dy/2; bottom=y_bottom+dy/2"
            ),
            "general_case": (
                "fit affine or homography from the four known logical phase centers "
                "to the four measured CCD centers; use flip_coded first to establish correspondence"
            ),
        },
        "files": phase_reports,
        "background_subtraction": False,
    }
    (output_dir / "fresnel_array_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        """# 532 nm Fresnel phase arrays

1. 振幅SLM固定播放 `amplitude_bmp/amplitude_uniform_white_1024x1024.bmp`。当前硬件合同是
   `255=白/透光`、`0=黑/遮光`，因此均匀照明必须使用全255，不能再使用旧版全0图。
2. 先用 `n1_1x1_uniform` 比较5/10/15 cm，确认相机处于哪个焦面。
3. 用对应距离的 `n4_2x2_flip_coded` 判断上下/左右对应关系：逻辑左上角透镜的
   有效相位口径较小，因此对应焦点应明显更弱，但中心坐标不变。
4. 再用 `n4_2x2_uniform` 提取四个等口径焦点中心并拟合CCD ROI。
5. `n9_3x3_uniform` 用于检查中心、边缘和非线性畸变；`flip_coded`用于复核翻转。

共同有效范围为相位SLM上的 `[472,82,1488,1098]`，即1016×1016，中心
`(980,590)`。2×2逻辑中心为 `(726,336)`、`(1234,336)`、`(726,844)`、
`(1234,844)`。BMP已经执行既有纵向翻转；CSV同时记录逻辑中心和实际BMP中心。

478是17 µm振幅/逻辑CCD像素数，508是每个半区在8 µm相位SLM上的像素数：
`239×17=4063 µm`，`508×8=4064 µm`，所以相邻焦点间距的物理误差只有1 µm。
四个焦点位于四个半区中心而不是有效区边界；轴对齐时必须再向外延伸半个焦点间距
才能恢复完整ROI，有旋转或剪切时则应拟合仿射/单应映射。

完整1920×1200平面使用全场均匀振幅进行5 cm数值传播复核时，有效透镜区外的平相位
只形成低背景，四个焦点峰值约为背景中位数的3.1×10^4倍，中心误差不超过0.5个相位像素。

四个完全相同的焦点具有翻转对称性，不能单独判断翻转；必须先使用
`flip_coded` 建立四点对应，再使用 `uniform` 做精确ROI拟合。
""",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = generate(Path(args.config))
    print(
        f"Generated 1 uniform-white amplitude BMP and {len(report['files'])} "
        f"Fresnel phase-array BMPs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
