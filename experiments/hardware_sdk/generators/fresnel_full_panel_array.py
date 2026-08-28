"""Generate full-panel phase-only Fresnel point and cross calibration masks.

The amplitude SLM is deliberately all white (255).  The phase SLM is never
used as an amplitude aperture: all pixels that can survive the calibrated
vertical reflection about ``center_y`` carry a phase value.  Ordinary
spherical Fresnel phase produces point foci; a coherent superposition of two
orthogonal cylindrical Fresnel waves produces a visibly extended cross.
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

from .fresnel_square_aperture_array import reflect_vertical_about_edge_center


SCHEMA_VERSION = 4


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


def _axis_targets(active_px: int, pitch_um: float, grid: int) -> list[float]:
    half_span = float(active_px) * float(pitch_um) / 2.0
    if grid == 1:
        return [0.0]
    if grid == 2:
        return [-half_span, half_span]
    if grid == 3:
        return [-half_span, 0.0, half_span]
    raise ValueError("grid must be 1, 2, or 3")


def _tile_edges(length: int, grid: int) -> list[tuple[int, int]]:
    edges = np.rint(np.linspace(0, length, grid + 1)).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(grid)]


def build_phase(
    *,
    mode: str,
    grid: int,
    size_wh: tuple[int, int],
    active_height: int,
    center_xy: tuple[float, float],
    phase_pitch_um: float,
    wavelength_nm: float,
    propagation_cm: float,
    target_span_px: int,
    target_pitch_um: float,
    cross_relative_phase_rad: float = math.pi / 2.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build logical (unflipped) full-panel phase and focus metadata."""

    width, height = map(int, size_wh)
    if not 0 < active_height <= height:
        raise ValueError("active_height must fit inside the phase panel")
    if mode not in {"point", "cross"}:
        raise ValueError("mode must be point or cross")
    wavelength_um = float(wavelength_nm) * 1.0e-3
    distance_um = float(propagation_cm) * 1.0e4
    x_um = (
        np.arange(width, dtype=np.float64) + 0.5 - float(center_xy[0])
    ) * float(phase_pitch_um)
    y_um = (
        np.arange(active_height, dtype=np.float64) + 0.5 - float(center_xy[1])
    ) * float(phase_pitch_um)
    targets = _axis_targets(target_span_px, target_pitch_um, grid)
    # Outside the alias-safe lens supports use a 0/pi rejection grating.  This
    # region is phase-modulated (not black/blocked) and sends otherwise
    # unfocused full-white illumination away from the calibration ROI.
    iy, ix = np.indices((active_height, width))
    phase = np.zeros((height, width), dtype=np.float64)
    phase[:active_height] = np.pi * (((ix // 2) + (iy // 2)) % 2)
    records: list[dict[str, Any]] = []
    x_tiles = _tile_edges(width, grid)
    y_tiles = _tile_edges(active_height, grid)
    # At 10 cm an 8 um pixel can represent a quadratic lens only while its
    # local spatial frequency remains below Nyquist.  Keep a 10% safety margin:
    # |coordinate-target| <= 0.9 * lambda*z/(2*pitch).
    alias_safe_half_um = (
        0.9 * wavelength_um * distance_um / (2.0 * float(phase_pitch_um))
    )

    for row, ((y0, y1), target_y) in enumerate(zip(y_tiles, targets)):
        yy = y_um[y0:y1, None]
        for column, ((x0, x1), target_x) in enumerate(zip(x_tiles, targets)):
            xx = x_um[None, x0:x1]
            linear = 2.0 * np.pi * (xx * target_x + yy * target_y) / (
                wavelength_um * distance_um
            )
            if mode == "point":
                local = -np.pi * (xx * xx + yy * yy) / (
                    wavelength_um * distance_um
                ) + linear
            else:
                # Each component focuses only one axis.  Their phase-only
                # coherent superposition yields one vertical and one horizontal
                # line crossing at the requested physical target.
                vertical = (
                    -np.pi * xx * xx / (wavelength_um * distance_um) + linear
                )
                horizontal = (
                    -np.pi * yy * yy / (wavelength_um * distance_um) + linear
                )
                local = np.angle(
                    np.exp(1j * vertical)
                    + np.exp(1j * (horizontal + float(cross_relative_phase_rad)))
                )
            support = (np.abs(xx - target_x) <= alias_safe_half_um) & (
                np.abs(yy - target_y) <= alias_safe_half_um
            )
            current = phase[y0:y1, x0:x1]
            current[support] = local[support]
            phase[y0:y1, x0:x1] = current
            ys, xs = np.nonzero(support)
            support_bounds = [
                int(x0 + xs.min()),
                int(y0 + ys.min()),
                int(x0 + xs.max() + 1),
                int(y0 + ys.max() + 1),
            ]
            records.append(
                {
                    "grid": grid,
                    "mode": mode,
                    "row": row,
                    "column": column,
                    "phase_tile_xyxy": [x0, y0, x1, y1],
                    "alias_safe_lens_support_xyxy": support_bounds,
                    "alias_safe_half_width_um": float(alias_safe_half_um),
                    "target_xy_um": [float(target_x), float(target_y)],
                    "target_xy_phase_px_from_center": [
                        float(target_x / phase_pitch_um),
                        float(target_y / phase_pitch_um),
                    ],
                }
            )
    return _phase_to_u8(phase), records


def _simulate(
    logical_phase_u8: np.ndarray,
    *,
    active_height: int,
    phase_pitch_um: float,
    wavelength_nm: float,
    propagation_cm: float,
    pad_size: int,
) -> np.ndarray:
    height, width = logical_phase_u8.shape
    if pad_size < max(height, width):
        raise ValueError("simulation pad_size must contain the phase panel")
    field = np.zeros((pad_size, pad_size), dtype=np.complex64)
    y0 = (pad_size - height) // 2
    x0 = (pad_size - width) // 2
    phase = logical_phase_u8.astype(np.float32) * (2.0 * np.pi / 256.0)
    illuminated = np.zeros_like(phase, dtype=np.float32)
    illuminated[:active_height] = 1.0
    field[y0 : y0 + height, x0 : x0 + width] = illuminated * np.exp(1j * phase)

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


def generate(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(raw["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    amp = raw["amplitude_slm"]
    phase = raw["phase_slm"]
    simulation = raw["simulation"]
    amp_size = tuple(map(int, amp["size_wh"]))
    phase_size = tuple(map(int, phase["size_wh"]))
    center_xy = tuple(map(float, phase["center_xy"]))
    # Reflection about y=590 maps rows [0,1180) onto themselves.  The final 20
    # rows cannot be populated without clipping; they remain zero phase, not a
    # blocked aperture.
    active_height = min(phase_size[1], int(round(2.0 * center_xy[1])))
    amplitude = np.full((amp_size[1], amp_size[0]), 255, dtype=np.uint8)
    amplitude_record = _save_u8(amplitude, output_dir / "A_WHITE.bmp", "BMP")

    files: list[dict[str, Any]] = [amplitude_record]
    targets: list[dict[str, Any]] = []
    for grid in (1, 2, 3):
        for mode in ("point", "cross"):
            logical, records = build_phase(
                mode=mode,
                grid=grid,
                size_wh=phase_size,
                active_height=active_height,
                center_xy=center_xy,
                phase_pitch_um=float(phase["pixel_pitch_um"]),
                wavelength_nm=float(raw["wavelength_nm"]),
                propagation_cm=float(raw["propagation_cm"]),
                target_span_px=int(raw["target_field"]["size_px"]),
                target_pitch_um=float(raw["target_field"]["pixel_pitch_um"]),
                cross_relative_phase_rad=float(raw.get("cross_relative_phase_rad", math.pi / 2)),
            )
            exported = (
                reflect_vertical_about_edge_center(logical, center_xy[1])
                if bool(phase.get("flip_vertical", True))
                else logical
            )
            if bool(phase.get("flip_horizontal", False)):
                exported = np.fliplr(exported)
            name = f"P{grid * grid}_{mode.upper()}.bmp"
            record = _save_u8(exported, output_dir / name, "BMP")
            record.update({"mode": mode, "grid": grid, "active_phase_rows": active_height})
            files.append(record)
            targets.extend(records)

            intensity = _simulate(
                logical,
                active_height=active_height,
                phase_pitch_um=float(phase["pixel_pitch_um"]),
                wavelength_nm=float(raw["wavelength_nm"]),
                propagation_cm=float(raw["propagation_cm"]),
                pad_size=int(simulation["pad_size"]),
            )
            linear = np.rint(np.clip(intensity, 0.0, 1.0) * 255.0).astype(np.uint8)
            floor_db = float(simulation.get("log_floor_db", -50.0))
            db = 10.0 * np.log10(np.maximum(intensity, 10.0 ** (floor_db / 10.0)))
            log_view = np.rint((db - floor_db) / -floor_db * 255.0).astype(np.uint8)
            _save_u8(linear, output_dir / "ideal" / f"I{grid * grid}_{mode.upper()}_linear.png", "PNG")
            _save_u8(log_view, output_dir / "ideal" / f"I{grid * grid}_{mode.upper()}_log.png", "PNG")
            np.save(output_dir / "ideal" / f"I{grid * grid}_{mode.upper()}.npy", intensity)

    with (output_dir / "targets.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": "full_panel_phase_only_fresnel_calibration",
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
            "pixel_pitch_um": float(phase["pixel_pitch_um"]),
            "center_xy": list(center_xy),
            "active_rows": [0, active_height],
            "inactive_rows_are_zero_phase_not_an_amplitude_stop": True,
            "flip_vertical_about_center_y": bool(phase.get("flip_vertical", True)),
        },
        "target_field": dict(raw["target_field"]),
        "files": files,
        "notes": [
            "POINT masks use an ordinary spherical Fresnel phase.",
            "CROSS masks use a phase-only superposition of orthogonal cylindrical Fresnel waves.",
            "Dark phase pixels encode zero phase; only A_WHITE controls transmission.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Fresnel 标定（当前唯一版本）\n\n"
        "振幅始终播放 `A_WHITE.bmp`（1024×1024，所有像素均为 255）。"
        "相位中的灰度 0 是 0 rad，不是黑色遮光。\n\n"
        "先用 `P1_POINT.bmp` 移动 CCD 找 10 cm 焦面；再用 `P4_POINT.bmp` "
        "确定四点和 ROI，用 `P9_POINT.bmp` 检查畸变。若普通焦点太小不便观察，"
        "使用对应 `*_CROSS.bmp`；十字来自正交柱面菲涅尔相位，不是画在振幅上的十字。\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = generate(Path(args.config).expanduser().resolve())
    print(
        f"[fresnel_full_panel] files={len(manifest['files'])} "
        f"active_rows={manifest['phase_contract']['active_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
