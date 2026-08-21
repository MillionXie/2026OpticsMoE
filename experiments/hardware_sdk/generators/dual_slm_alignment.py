"""Generate paired 17 um amplitude / 8 um phase SLM alignment BMPs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    physical_pitch_nearest,
    place_at_center,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checker(size: int, block: int, high: int = 255) -> np.ndarray:
    y, x = np.indices((size, size))
    return (((x // block + y // block) % 2) * high).astype(np.uint8)


def _crosshair(size: int, width: int = 3, value: int = 255) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    result[center - width : center + width, :] = value
    result[:, center - width : center + width] = value
    # An asymmetric square makes rotations and flips observable.
    marker = max(5, size // 32)
    result[marker : 2 * marker, 2 * marker : 4 * marker] = value
    return result


def _outline(size: int, width: int = 3, value: int = 255) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.uint8)
    result[:width] = value
    result[-width:] = value
    result[:, :width] = value
    result[:, -width:] = value
    return result


def _blazed(size: int, period: int, axis: str) -> np.ndarray:
    coordinate = np.arange(size, dtype=np.int64)
    ramp = np.floor((coordinate % period) * 256.0 / period).astype(np.uint8)
    return (
        np.broadcast_to(ramp[None, :], (size, size)).copy()
        if axis == "x"
        else np.broadcast_to(ramp[:, None], (size, size)).copy()
    )


def _binary_grating(
    height: int,
    width: int,
    period: int,
    axis: str,
    *,
    high: int = 128,
) -> np.ndarray:
    """Return a 0/pi binary phase grating in logical 17 um coordinates."""
    if period < 2 or period % 2:
        raise ValueError("Binary grating period must be an even integer >= 2")
    coordinate = np.arange(width if axis == "x" else height, dtype=np.int64)
    stripe = ((coordinate % period) >= period // 2).astype(np.uint8) * high
    return (
        np.broadcast_to(stripe[None, :], (height, width)).copy()
        if axis == "x"
        else np.broadcast_to(stripe[:, None], (height, width)).copy()
    )


def _registered_checker_grating(
    size: int,
    cell_size: int,
    grating_period: int,
) -> np.ndarray:
    """Alternate x/y 0-pi gratings on the same grid as an amplitude checker.

    The phase-cell edges and amplitude-cell edges are generated from identical
    logical coordinates.  A focused 1x 4F relay therefore shows grating lines
    clipped exactly by the bright amplitude squares when both SLMs are aligned.
    """
    y, x = np.indices((size, size))
    cell_row = y // cell_size
    cell_column = x // cell_size
    vertical = _binary_grating(size, size, grating_period, "x")
    horizontal = _binary_grating(size, size, grating_period, "y")
    # Keep grating orientation independent from checker parity.  Otherwise all
    # bright cells in one checker exposure would accidentally show the same
    # direction.  Alternating by row produces both x/y gratings in one focused
    # image, matching the laboratory acceptance example.
    phase = np.where(cell_row % 2 == 0, vertical, horizontal)
    # A one-logical-pixel zero-phase frame makes every phase-cell boundary
    # explicit without changing the corresponding checker geometry.
    on_boundary = (x % cell_size == 0) | (y % cell_size == 0)
    phase[on_boundary] = 0
    return phase.astype(np.uint8)


def _ideal_registration_preview(
    amplitude: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    """Idealized focused overlay for file selection, not a propagation model."""
    open_pixels = amplitude.astype(np.float32) / 255.0
    phase_lines = np.where(phase >= 64, 245.0, 55.0)
    return np.round(open_pixels * phase_lines).clip(0, 255).astype(np.uint8)


def _spot_grid(size: int, count: int = 7) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.uint8)
    positions = np.linspace(size * 0.1, size * 0.9, count).round().astype(int)
    radius = max(2, size // 120)
    for y in positions:
        for x in positions:
            result[y - radius : y + radius + 1, x - radius : x + radius + 1] = 255
    # Unique orientation marker in the upper-left lattice cell.
    x, y = positions[0], positions[0]
    result[y - 2 * radius : y + 2 * radius + 1, x - 2 * radius : x + 2 * radius + 1] = 128
    return result


def _moe_boxes(size: int, *, value: int = 255) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.uint8)
    for y0 in (0, 254):
        for x0 in (0, 254):
            result[y0 : y0 + 3, x0 : x0 + 224] = value
            result[y0 + 221 : y0 + 224, x0 : x0 + 224] = value
            result[y0 : y0 + 224, x0 : x0 + 3] = value
            result[y0 : y0 + 224, x0 + 221 : x0 + 224] = value
    return result


def _expert_window(size: int, index: int, pattern: np.ndarray) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.uint8)
    row, col = divmod(index, 2)
    y0, x0 = row * 254, col * 254
    result[y0 : y0 + 224, x0 : x0 + 224] = pattern[:224, :224]
    return result


def _moe_unique_gratings(size: int) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.uint8)
    specifications = (("x", 8), ("y", 8), ("x", 16), ("y", 16))
    for index, (axis, period) in enumerate(specifications):
        row, col = divmod(index, 2)
        y0, x0 = row * 254, col * 254
        result[y0 : y0 + 224, x0 : x0 + 224] = _blazed(224, period, axis)
    return result


def _save_full(
    active: np.ndarray,
    path: Path,
    *,
    slm_size_wh: tuple[int, int],
    center_xy: tuple[float, float],
) -> dict[str, Any]:
    canvas, bounds, actual_center = place_at_center(
        Image.fromarray(active, mode="L"),
        slm_size_wh=slm_size_wh,
        center_xy=center_xy,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="BMP")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "active_size_wh": [active.shape[1], active.shape[0]],
        "active_bounds_xyxy": list(bounds),
        "actual_center_xy": list(actual_center),
    }


def generate(
    config_path: Path,
    *,
    phase_center_override: tuple[float, float] | None = None,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(raw["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    logical = raw["logical"]
    amplitude = raw["amplitude_slm"]
    phase = raw["phase_slm"]
    active_size = int(logical["active_size"])
    usable_size = int(logical["usable_size"])
    logical_pitch = float(logical["pixel_pitch_um"])
    amplitude_size = tuple(map(int, amplitude["size_wh"]))
    phase_size = tuple(map(int, phase["size_wh"]))
    amplitude_center = tuple(map(float, amplitude["center_xy"]))
    phase_center = (
        tuple(map(float, phase_center_override))
        if phase_center_override is not None
        else tuple(map(float, phase["center_xy"]))
    )
    phase_pitch = float(phase["pixel_pitch_um"])
    flip_vertical = bool(phase.get("flip_vertical", True))
    flip_horizontal = bool(phase.get("flip_horizontal", False))

    amplitude_patterns: dict[str, np.ndarray] = {
        "flat_black": np.zeros((active_size, active_size), dtype=np.uint8),
        "flat_white": np.full((active_size, active_size), 255, dtype=np.uint8),
        "active_outline": _outline(active_size),
        "asymmetric_crosshair": _crosshair(active_size),
        "checker_logical_b8": _checker(active_size, 8),
        "checker_logical_b16": _checker(active_size, 16),
        "checker_logical_b32": _checker(active_size, 32),
        "spot_grid_7x7": _spot_grid(active_size),
        "moe4_outline": _moe_boxes(active_size),
    }
    for index in range(4):
        amplitude_patterns[f"expert_{index}_only"] = _expert_window(
            active_size, index, np.full((224, 224), 255, dtype=np.uint8)
        )
    for aperture_size in (384, 416, 448, 478, 496, usable_size):
        field = np.zeros((usable_size, usable_size), dtype=np.uint8)
        start = (usable_size - aperture_size) // 2
        field[start : start + aperture_size, start : start + aperture_size] = 255
        amplitude_patterns[f"aperture_{aperture_size}"] = field

    registration_specs = ((64, 8), (80, 8), (96, 8))
    registration_logical: dict[str, dict[str, Any]] = {}
    for cell_size, period in registration_specs:
        checker = 255 - _checker(active_size, cell_size)
        complement = 255 - checker
        grating = _registered_checker_grating(active_size, cell_size, period)
        amplitude_patterns[f"registration_checker_c{cell_size}"] = checker
        amplitude_patterns[f"registration_checker_c{cell_size}_complement"] = (
            complement
        )
        registration_logical[f"checker_xy_c{cell_size}_p{period}"] = {
            "cell_size": cell_size,
            "period": period,
            "checker": checker,
            "complement": complement,
            "phase": grating,
        }

    phase_patterns: dict[str, np.ndarray] = {
        "flat_0": np.zeros((active_size, active_size), dtype=np.uint8),
        "flat_pi": np.full((active_size, active_size), 128, dtype=np.uint8),
        "asymmetric_crosshair_pi": _crosshair(active_size, value=128),
        "checker_logical_b8_native_b17_0pi": _checker(active_size, 8, 128),
        "checker_logical_b16_native_b34_0pi": _checker(active_size, 16, 128),
        "checker_logical_b32_native_b68_0pi": _checker(active_size, 32, 128),
        "blazed_x_period8_logical": _blazed(active_size, 8, "x"),
        "blazed_y_period8_logical": _blazed(active_size, 8, "y"),
        "blazed_x_period16_logical": _blazed(active_size, 16, "x"),
        "blazed_y_period16_logical": _blazed(active_size, 16, "y"),
        "moe4_outline_pi": _moe_boxes(active_size, value=128),
        "moe4_unique_gratings": _moe_unique_gratings(active_size),
    }
    for name, values in registration_logical.items():
        phase_patterns[f"registration_{name}"] = values["phase"]
    for index, (axis, period) in enumerate(
        (("x", 8), ("y", 8), ("x", 16), ("y", 16))
    ):
        phase_patterns[f"expert_{index}_{axis}_grating_p{period}"] = _expert_window(
            active_size, index, _blazed(224, period, axis)
        )

    files: dict[str, dict[str, Any]] = {"amplitude": {}, "phase": {}}
    for name, pattern in amplitude_patterns.items():
        files["amplitude"][name] = _save_full(
            pattern,
            output_dir / "amplitude_bmp" / f"amplitude_{name}_1024x1024.bmp",
            slm_size_wh=amplitude_size,
            center_xy=amplitude_center,
        )
    for name, logical_pattern in phase_patterns.items():
        oriented = logical_pattern
        if flip_vertical:
            oriented = np.flipud(oriented)
        if flip_horizontal:
            oriented = np.fliplr(oriented)
        native = physical_pitch_nearest(
            np.ascontiguousarray(oriented),
            logical_pixel_pitch_um=logical_pitch,
            slm_pixel_pitch_um=phase_pitch,
        )
        files["phase"][name] = _save_full(
            native,
            output_dir / "phase_bmp" / f"phase_{name}_1920x1200.bmp",
            slm_size_wh=phase_size,
            center_xy=phase_center,
        )

    registration_pairs: list[dict[str, Any]] = []
    preview_dir = output_dir / "registration_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for name, values in registration_logical.items():
        phase_name = f"registration_{name}"
        for amplitude_suffix, amplitude_key in (
            ("primary", f"registration_checker_c{values['cell_size']}"),
            (
                "complement",
                f"registration_checker_c{values['cell_size']}_complement",
            ),
        ):
            preview = _ideal_registration_preview(
                values["checker"]
                if amplitude_suffix == "primary"
                else values["complement"],
                values["phase"],
            )
            preview_path = preview_dir / f"pair_{name}_{amplitude_suffix}.png"
            Image.fromarray(preview, mode="L").save(preview_path, format="PNG")
            registration_pairs.append(
                {
                    "pair_id": f"{name}_{amplitude_suffix}",
                    "amplitude": files["amplitude"][amplitude_key]["path"],
                    "phase": files["phase"][phase_name]["path"],
                    "idealized_preview": str(preview_path),
                    "preview_sha256": _sha256(preview_path),
                    "logical_cell_size_px": values["cell_size"],
                    "logical_grating_period_px": values["period"],
                    "phase_native_cell_size_px_approx": round(
                        values["cell_size"] * logical_pitch / phase_pitch
                    ),
                    "phase_native_grating_period_px": round(
                        values["period"] * logical_pitch / phase_pitch
                    ),
                }
            )

    report = {
        "schema_version": 1,
        "purpose": "17um amplitude to 8um phase SLM registration without magnification",
        "coordinate_model": "shared 4F optical axis; translation center only",
        "logical": {
            "active_size": active_size,
            "usable_size": usable_size,
            "pixel_pitch_um": logical_pitch,
        },
        "amplitude_slm": {
            "size_wh": list(amplitude_size),
            "pixel_pitch_um": float(amplitude["pixel_pitch_um"]),
            "center_xy": list(amplitude_center),
            "mapping": "one_to_one",
        },
        "phase_slm": {
            "size_wh": list(phase_size),
            "pixel_pitch_um": phase_pitch,
            "center_xy": list(phase_center),
            "physical_ratio": logical_pitch / phase_pitch,
            "mapping": "centered physical-coordinate nearest",
            "flip_vertical_before_raster": flip_vertical,
            "flip_horizontal_before_raster": flip_horizontal,
        },
        "exact_checker_correspondence": {
            "8_logical_px": "17_phase_px",
            "16_logical_px": "34_phase_px",
            "32_logical_px": "68_phase_px",
        },
        "background_subtraction": False,
        "registration_protocol": {
            "target": "focused amplitude edges and phase-cell edges coincide to approximately one camera pixel",
            "pattern": "binary amplitude checker plus alternating x/y 0-pi phase gratings",
            "use_complement": "capture both primary and complement so every phase cell is visible once",
            "preview_is_propagation_simulation": False,
            "pairs": registration_pairs,
        },
        "files": files,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignment_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase-center-x", type=float)
    parser.add_argument("--phase-center-y", type=float)
    args = parser.parse_args()
    if (args.phase_center_x is None) != (args.phase_center_y is None):
        parser.error("--phase-center-x and --phase-center-y must be provided together")
    center_override = (
        (args.phase_center_x, args.phase_center_y)
        if args.phase_center_x is not None
        else None
    )
    report = generate(Path(args.config), phase_center_override=center_override)
    print(
        f"Generated {len(report['files']['amplitude'])} amplitude and "
        f"{len(report['files']['phase'])} phase alignment BMPs; "
        f"{len(report['registration_protocol']['pairs'])} checker/grating pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
