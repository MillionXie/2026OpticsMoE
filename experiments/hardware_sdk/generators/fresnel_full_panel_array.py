"""Generate MATLAB-style square Fresnel arrays for the current dual-SLM rig.

The phase inside every square lenslet follows the laboratory MATLAB formula

    phase = -k * ((p*x)**2 + (p*y)**2) / (2*f)

and is wrapped to [0, 2*pi) before 8-bit export. Unlike the retired generator,
this implementation has no circular aperture, alias-safe clipping, rejection
grating, cylindrical-wave superposition, or synthetic cross phase. Pixels
outside the square phase windows encode 0 rad; they do not block light. The
amplitude SLM therefore stays full-white (255) throughout Fresnel calibration.

The n4/n9 lens centres are the exact physical vertices/midpoints of the formal
478 x 17 um field after mapping onto the 8 um phase SLM. Their outer spacing is
478*17/8 = 1015.75 phase pixels. A MATLAB window equal to that spacing cannot
fit twice inside the 1200-row panel, so window size is configured independently
from centre spacing and every square lenslet is required to fit without being
clipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from PIL import Image

from .fresnel_square_aperture_array import reflect_vertical_about_edge_center


SCHEMA_VERSION = 5
GRIDS = (1, 2, 3)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_u8(value: np.ndarray, path: Path, image_format: str) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path, format=image_format)
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "size_wh": [int(array.shape[1]), int(array.shape[0])],
        "min_uint8": int(array.min()),
        "max_uint8": int(array.max()),
        "unique_uint8": int(np.unique(array).size),
    }


def _phase_to_u8(phase: np.ndarray) -> np.ndarray:
    return np.floor(np.mod(phase, 2.0 * np.pi) * (256.0 / (2.0 * np.pi))).astype(
        np.uint8
    )


def _axis_targets_phase_edge(
    *,
    center_edge: float,
    target_span_px: int,
    target_pitch_um: float,
    phase_pitch_um: float,
    grid: int,
) -> list[float]:
    """Return center/vertices/midpoints in phase-SLM edge coordinates."""

    half_span = (
        float(target_span_px) * float(target_pitch_um) / float(phase_pitch_um) / 2.0
    )
    if grid == 1:
        offsets = [0.0]
    elif grid == 2:
        offsets = [-half_span, half_span]
    elif grid == 3:
        offsets = [-half_span, 0.0, half_span]
    else:
        raise ValueError("grid must be 1, 2, or 3")
    return [float(center_edge) + offset for offset in offsets]


def _window_for_grid(windows: Mapping[Any, Any], grid: int) -> int:
    count = grid * grid
    value = windows.get(count, windows.get(str(count)))
    if value is None:
        raise ValueError(f"lens_window_phase_px must define n{count}")
    window = int(value)
    if window <= 0:
        raise ValueError("Fresnel square-window sizes must be positive")
    return window


def _centered_integer_window(
    center_edge: float, window: int, limit: int
) -> tuple[int, int]:
    """Quantize a full square window around a subpixel physical target."""

    start = int(math.floor(float(center_edge) - int(window) / 2.0))
    stop = start + int(window)
    if start < 0 or stop > int(limit):
        raise ValueError(
            "MATLAB-style Fresnel window would be clipped: "
            f"center={center_edge:.6f}, window={window}, panel_limit={limit}, "
            f"bounds=[{start},{stop}). Reduce lens_window_phase_px; do not "
            "silently create a quarter/half lens."
        )
    return start, stop


def matlab_square_fresnel_phase(
    *,
    grid: int,
    size_wh: tuple[int, int],
    center_xy: tuple[float, float],
    phase_pitch_um: float,
    wavelength_nm: float,
    propagation_cm: float,
    target_span_px: int,
    target_pitch_um: float,
    lens_window_phase_px: Mapping[Any, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build an unflipped phase panel using the teacher's quadratic formula."""

    width, height = map(int, size_wh)
    if width <= 0 or height <= 0:
        raise ValueError("phase SLM dimensions must be positive")
    if min(phase_pitch_um, wavelength_nm, propagation_cm, target_pitch_um) <= 0.0:
        raise ValueError("Fresnel physical parameters must be positive")
    window = _window_for_grid(lens_window_phase_px, grid)
    x_targets = _axis_targets_phase_edge(
        center_edge=float(center_xy[0]),
        target_span_px=target_span_px,
        target_pitch_um=target_pitch_um,
        phase_pitch_um=phase_pitch_um,
        grid=grid,
    )
    y_targets = _axis_targets_phase_edge(
        center_edge=float(center_xy[1]),
        target_span_px=target_span_px,
        target_pitch_um=target_pitch_um,
        phase_pitch_um=phase_pitch_um,
        grid=grid,
    )
    phase = np.zeros((height, width), dtype=np.float64)
    occupied = np.zeros((height, width), dtype=bool)
    wavelength_m = float(wavelength_nm) * 1.0e-9
    pitch_m = float(phase_pitch_um) * 1.0e-6
    focal_m = float(propagation_cm) * 1.0e-2
    wave_number = 2.0 * np.pi / wavelength_m
    records: list[dict[str, Any]] = []

    for row, target_y in enumerate(y_targets):
        y0, y1 = _centered_integer_window(target_y, window, height)
        yy_m = (
            np.arange(y0, y1, dtype=np.float64) + 0.5 - float(target_y)
        )[:, None] * pitch_m
        for column, target_x in enumerate(x_targets):
            x0, x1 = _centered_integer_window(target_x, window, width)
            if bool(occupied[y0:y1, x0:x1].any()):
                raise RuntimeError(
                    "Fresnel square windows overlap; reduce lens_window_phase_px"
                )
            xx_m = (
                np.arange(x0, x1, dtype=np.float64) + 0.5 - float(target_x)
            )[None, :] * pitch_m
            # Exact laboratory MATLAB principle:
            # -k*((ps*(X-cx))^2 + (ps*(Y-cy))^2)/(2*f).
            local = -wave_number * (xx_m * xx_m + yy_m * yy_m) / (2.0 * focal_m)
            phase[y0:y1, x0:x1] = local
            occupied[y0:y1, x0:x1] = True
            records.append(
                {
                    "array_count": grid * grid,
                    "grid": grid,
                    "row": row,
                    "column": column,
                    "logical_target_phase_edge_xy": [
                        float(target_x),
                        float(target_y),
                    ],
                    "logical_target_phase_px_from_center_xy": [
                        float(target_x - center_xy[0]),
                        float(target_y - center_xy[1]),
                    ],
                    "square_phase_window_xyxy": [x0, y0, x1, y1],
                    "square_phase_window_size_px": window,
                    "square_phase_window_physical_um": float(
                        window * phase_pitch_um
                    ),
                    "formula": "-k*((p*x)^2+(p*y)^2)/(2*f)",
                    "outside_window_phase_rad": 0.0,
                    "window_clipped": False,
                }
            )
    return _phase_to_u8(phase), records


def _simulate(
    logical_phase_u8: np.ndarray,
    *,
    phase_pitch_um: float,
    wavelength_nm: float,
    propagation_cm: float,
    pad_size: int,
) -> np.ndarray:
    """Propagate a full-white field through the quantized phase BMP."""

    height, width = logical_phase_u8.shape
    if pad_size < max(height, width):
        raise ValueError("simulation pad_size must contain the phase panel")
    field = np.zeros((pad_size, pad_size), dtype=np.complex64)
    y0 = (pad_size - height) // 2
    x0 = (pad_size - width) // 2
    phase = logical_phase_u8.astype(np.float32) * (2.0 * np.pi / 256.0)
    field[y0 : y0 + height, x0 : x0 + width] = np.exp(1j * phase)

    wavelength_m = float(wavelength_nm) * 1.0e-9
    pitch_m = float(phase_pitch_um) * 1.0e-6
    distance_m = float(propagation_cm) * 1.0e-2
    frequency = np.fft.fftfreq(pad_size, d=pitch_m)
    fx, fy = np.meshgrid(frequency, frequency)
    root = np.maximum(0.0, 1.0 / wavelength_m**2 - fx * fx - fy * fy)
    transfer = np.exp(1j * 2.0 * np.pi * distance_m * np.sqrt(root))
    propagated = np.fft.ifft2(np.fft.fft2(field) * transfer)
    intensity = np.abs(propagated) ** 2
    intensity /= max(float(intensity.max()), 1.0e-12)
    return intensity.astype(np.float32)


def _remove_retired_outputs(output_dir: Path) -> None:
    """Remove only files owned by the superseded point/cross generator."""

    retired = [
        output_dir / f"P{count}_CROSS.bmp" for count in (1, 4, 9)
    ]
    for count in (1, 4, 9):
        retired.extend(
            [
                output_dir / "ideal" / f"I{count}_CROSS.npy",
                output_dir / "ideal" / f"I{count}_CROSS_linear.png",
                output_dir / "ideal" / f"I{count}_CROSS_log.png",
            ]
        )
    for path in retired:
        if path.is_file():
            path.unlink()


def generate(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(raw["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_retired_outputs(output_dir)

    amp = raw["amplitude_slm"]
    phase_slm = raw["phase_slm"]
    simulation = raw["simulation"]
    amp_size = tuple(map(int, amp["size_wh"]))
    phase_size = tuple(map(int, phase_slm["size_wh"]))
    center_xy = tuple(map(float, phase_slm["center_xy"]))
    phase_pitch_um = float(phase_slm["pixel_pitch_um"])
    target_span_px = int(raw["target_field"]["size_px"])
    target_pitch_um = float(raw["target_field"]["pixel_pitch_um"])
    target_span_phase_px = target_span_px * target_pitch_um / phase_pitch_um

    amplitude = np.full((amp_size[1], amp_size[0]), 255, dtype=np.uint8)
    amplitude_record = _save_u8(amplitude, output_dir / "A_WHITE.bmp", "BMP")
    files: list[dict[str, Any]] = [amplitude_record]
    targets: list[dict[str, Any]] = []

    for grid in GRIDS:
        logical, records = matlab_square_fresnel_phase(
            grid=grid,
            size_wh=phase_size,
            center_xy=center_xy,
            phase_pitch_um=phase_pitch_um,
            wavelength_nm=float(raw["wavelength_nm"]),
            propagation_cm=float(raw["propagation_cm"]),
            target_span_px=target_span_px,
            target_pitch_um=target_pitch_um,
            lens_window_phase_px=raw["lens_window_phase_px"],
        )
        exported = (
            reflect_vertical_about_edge_center(logical, center_xy[1])
            if bool(phase_slm.get("flip_vertical", True))
            else logical
        )
        if bool(phase_slm.get("flip_horizontal", False)):
            exported = np.fliplr(exported)
        count = grid * grid
        name = f"P{count}_POINT.bmp"
        record = _save_u8(exported, output_dir / name, "BMP")
        record.update(
            {
                "mode": "matlab_square_quadratic",
                "grid": grid,
                "array_count": count,
                "lens_window_phase_px": _window_for_grid(
                    raw["lens_window_phase_px"], grid
                ),
            }
        )
        files.append(record)
        for item in records:
            item["phase_bmp"] = name
            logical_x, logical_y = item["logical_target_phase_edge_xy"]
            exported_y = (
                2.0 * center_xy[1] - logical_y
                if bool(phase_slm.get("flip_vertical", True))
                else logical_y
            )
            exported_x = (
                2.0 * center_xy[0] - logical_x
                if bool(phase_slm.get("flip_horizontal", False))
                else logical_x
            )
            item["exported_target_bmp_edge_xy"] = [exported_x, exported_y]
            targets.append(item)

        intensity = _simulate(
            logical,
            phase_pitch_um=phase_pitch_um,
            wavelength_nm=float(raw["wavelength_nm"]),
            propagation_cm=float(raw["propagation_cm"]),
            pad_size=int(simulation["pad_size"]),
        )
        linear = np.rint(np.clip(intensity, 0.0, 1.0) * 255.0).astype(np.uint8)
        floor_db = float(simulation.get("log_floor_db", -50.0))
        db = 10.0 * np.log10(np.maximum(intensity, 10.0 ** (floor_db / 10.0)))
        log_view = np.rint((db - floor_db) / -floor_db * 255.0).astype(np.uint8)
        _save_u8(linear, output_dir / "ideal" / f"I{count}_POINT_linear.png", "PNG")
        _save_u8(log_view, output_dir / "ideal" / f"I{count}_POINT_log.png", "PNG")
        np.save(output_dir / "ideal" / f"I{count}_POINT.npy", intensity)

    with (output_dir / "targets.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": "matlab_square_fresnel_roi_vertex_calibration",
        "source_principle": "teacher MATLAB quadratic square-window array",
        "formula": "phase=-k*((p*x)^2+(p*y)^2)/(2*f); phase=mod(phase,2*pi)",
        "wavelength_nm": float(raw["wavelength_nm"]),
        "propagation_cm": float(raw["propagation_cm"]),
        "amplitude_contract": {
            "size_wh": list(amp_size),
            "pixel_pitch_um": float(amp["pixel_pitch_um"]),
            "every_pixel_uint8": 255,
            "meaning": "full-white transmissive illumination; no amplitude aperture",
        },
        "phase_contract": {
            "size_wh": list(phase_size),
            "pixel_pitch_um": phase_pitch_um,
            "center_xy": list(center_xy),
            "flip_vertical_about_center_y": bool(
                phase_slm.get("flip_vertical", True)
            ),
            "flip_horizontal_about_center_x": bool(
                phase_slm.get("flip_horizontal", False)
            ),
            "outside_square_windows_phase_rad": 0.0,
            "dark_phase_pixels_are_not_blocked": True,
        },
        "target_field": {
            **dict(raw["target_field"]),
            "physical_span_um": target_span_px * target_pitch_um,
            "phase_span_px": target_span_phase_px,
            "n4_outer_center_spacing_phase_px": target_span_phase_px,
            "n9_adjacent_center_spacing_phase_px": target_span_phase_px / 2.0,
        },
        "lens_window_phase_px": {
            f"n{count}": int(
                raw["lens_window_phase_px"].get(
                    count, raw["lens_window_phase_px"].get(str(count))
                )
            )
            for count in (1, 4, 9)
        },
        "files": files,
        "notes": [
            "P4 outer focus centres are the exact 478x17um ROI vertices.",
            "P9 adds exact edge midpoints and the optical-axis centre.",
            "Square window size is independent of ROI-centre spacing because two 1015.75px windows cannot fit on a 1200-row SLM.",
            "No circle, rejection grating, synthetic cross, or clipped quarter/half lens is used.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# MATLAB 方窗菲涅尔 ROI 标定\n\n"
        "振幅始终播放 `A_WHITE.bmp`（1024×1024，全部为 255）。相位 BMP 中的黑色表示 0 rad，不表示遮光。\n\n"
        "相位严格使用老师 MATLAB 的方窗二次相位：`-k*((p*x)^2+(p*y)^2)/(2*f)`，再对 2π 取模。"
        "P1 用于寻找 10 cm 焦面；P4 的四个中心对应 478×17 µm ROI 的四个物理顶点；"
        "P9 增加边中点和中心，用于检查翻转、畸变与 ROI。\n\n"
        "当前 n4/n9 方窗为 160×160 相位像素，这是在 y=590、1200 行面板上保证四角完整、不裁切的安全尺寸。"
        "区域外保持 0 rad；没有圆形孔径、外围光栅或人为 CROSS mask。\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = generate(Path(args.config).expanduser().resolve())
    print(
        f"[fresnel_matlab_roi] files={len(manifest['files'])} "
        f"span_phase_px={manifest['target_field']['phase_span_px']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
