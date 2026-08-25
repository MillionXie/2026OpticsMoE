"""Generate inverted-amplitude dual-SLM registration pairs with a phase scale sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from experiments.hardware_sdk.generators.dual_slm_alignment import (
    _checker,
    _ideal_registration_preview,
    _registered_checker_grating,
)
from experiments.hardware_sdk.workflows.reconstruct_slm import (
    physical_pitch_nearest,
    place_at_center,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_scale_values(max_abs_delta: float, step: float) -> list[float]:
    """Return 1 first, then positive/negative scale corrections outwards."""

    max_abs_delta = float(max_abs_delta)
    step = float(step)
    if not 0.0 < step <= max_abs_delta <= 0.005 + 1.0e-12:
        raise ValueError("require 0 < step <= max_abs_delta <= 0.005")
    count = int(round(max_abs_delta / step))
    if not np.isclose(count * step, max_abs_delta, atol=1.0e-12):
        raise ValueError("max_abs_delta must be an integer multiple of step")
    values = [1.0]
    for index in range(1, count + 1):
        delta = index * step
        values.extend((1.0 + delta, 1.0 - delta))
    return [round(value, 7) for value in values]


def dense_tetromino_mask(size: int, cell_size: int) -> tuple[np.ndarray, int]:
    """Return 25 separated white tetrominoes on a black logical background.

    A 5x5 arrangement of 3x3 piece tiles with a one-cell gutter gives many
    asymmetric landmarks while keeping individual tetromino boundaries easy to
    inspect through the focused 4F relay.
    """

    full_cells = int(size) // int(cell_size)
    if full_cells < 19:
        raise ValueError("dense tetromino layout requires at least 19 cells per axis")
    shapes = (
        ((0, 0), (0, 1), (0, 2), (1, 1)),  # T
        ((0, 0), (0, 1), (1, 0), (1, 1)),  # O
        ((0, 0), (1, 0), (2, 0), (2, 1)),  # L
        ((0, 1), (1, 1), (2, 0), (2, 1)),  # J
        ((0, 1), (0, 2), (1, 0), (1, 1)),  # S
        ((0, 0), (0, 1), (1, 1), (1, 2)),  # Z
        ((0, 0), (1, 0), (1, 1), (2, 0)),  # rotated T
        ((0, 0), (0, 1), (0, 2), (1, 2)),  # rotated L
    )
    cells = np.zeros((full_cells, full_cells), dtype=np.uint8)
    piece_count = 0
    for tile_row in range(5):
        for tile_column in range(5):
            shape = shapes[(tile_row * 5 + tile_column) % len(shapes)]
            origin_row, origin_column = tile_row * 4, tile_column * 4
            for row_offset, column_offset in shape:
                cells[origin_row + row_offset, origin_column + column_offset] = 255
            piece_count += 1
    result = np.zeros((size, size), dtype=np.uint8)
    logical_extent = full_cells * cell_size
    result[:logical_extent, :logical_extent] = np.repeat(
        np.repeat(cells, cell_size, axis=0), cell_size, axis=1
    )
    return result, piece_count


def _save_inverted_amplitude(
    intended_open_mask: np.ndarray,
    path: Path,
    *,
    slm_size_wh: tuple[int, int],
    center_xy: tuple[float, float],
) -> dict[str, Any]:
    intended_canvas, bounds, actual_center = place_at_center(
        Image.fromarray(intended_open_mask, mode="L"),
        slm_size_wh=slm_size_wh,
        center_xy=center_xy,
    )
    # The new amplitude optical path is observed to reverse black and white.
    # Invert the full command canvas, including the region outside the active
    # logical aperture, so the optical field reproduces intended_open_mask.
    commanded = 255 - np.asarray(intended_canvas, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(commanded, mode="L").save(path, format="BMP")
    _validate_bmp(path, slm_size_wh, {0, 255})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "encoding": "full_canvas_black_white_inverse",
        "active_bounds_xyxy": list(bounds),
        "actual_center_xy": list(actual_center),
        "commanded_black_fraction": float(np.mean(commanded == 0)),
        "commanded_white_fraction": float(np.mean(commanded == 255)),
    }


def _save_phase(
    native_phase: np.ndarray,
    path: Path,
    *,
    slm_size_wh: tuple[int, int],
    center_xy: tuple[float, float],
) -> dict[str, Any]:
    canvas, bounds, actual_center = place_at_center(
        Image.fromarray(native_phase, mode="L"),
        slm_size_wh=slm_size_wh,
        center_xy=center_xy,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="BMP")
    _validate_bmp(path, slm_size_wh, {0, 128})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "native_active_size_wh": [native_phase.shape[1], native_phase.shape[0]],
        "active_bounds_xyxy": list(bounds),
        "actual_center_xy": list(actual_center),
    }


def _validate_bmp(
    path: Path,
    expected_size_wh: tuple[int, int],
    allowed_values: set[int],
) -> None:
    with Image.open(path) as image:
        if image.format != "BMP" or image.mode != "L":
            raise RuntimeError(
                f"{path.name} must be native 8-bit grayscale BMP; "
                f"got format={image.format} mode={image.mode}"
            )
        if tuple(image.size) != tuple(expected_size_wh):
            raise RuntimeError(
                f"{path.name} size={image.size}, expected={expected_size_wh}"
            )
        values = set(int(value) for value in np.unique(np.asarray(image)))
    if not values.issubset(allowed_values):
        raise RuntimeError(
            f"{path.name} contains gray values {sorted(values - allowed_values)}"
        )


def _k_tag(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def generate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(raw["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()

    logical = raw["logical"]
    amplitude_config = raw["amplitude_slm"]
    phase_config = raw["phase_slm"]
    sweep = raw["phase_scale_sweep"]
    active_size = int(logical["active_size"])
    logical_pitch_um = float(logical["pixel_pitch_um"])
    grating_period = int(logical["grating_period_px"])
    amplitude_size = tuple(int(value) for value in amplitude_config["size_wh"])
    amplitude_center = tuple(float(value) for value in amplitude_config["center_xy"])
    phase_size = tuple(int(value) for value in phase_config["size_wh"])
    phase_center = tuple(float(value) for value in phase_config["center_xy"])
    phase_pitch_um = float(phase_config["pixel_pitch_um"])
    flip_vertical = bool(phase_config.get("flip_vertical", True))
    flip_horizontal = bool(phase_config.get("flip_horizontal", False))
    k_values = phase_scale_values(
        float(sweep["max_abs_delta"]), float(sweep["step"])
    )

    regular_open = _checker(active_size, 64)
    dense_open, tetromino_count = dense_tetromino_mask(active_size, 24)
    patterns: list[dict[str, Any]] = [
        {
            "order": 1,
            "name": "regular_checker_c64",
            "folder": "01_checker_c64_inv",
            "cell_size": 64,
            "intended_open": regular_open,
            "orientation_mode": "visible_checker_cells",
            "layout": "strict checker; command BMP is the exact full-canvas inverse",
            "landmark_count": None,
        },
        {
            "order": 2,
            "name": "dense_tetromino_c24_25pieces",
            "folder": "02_tetromino_c24_inv",
            "cell_size": 24,
            "intended_open": dense_open,
            "orientation_mode": "neighbor_cells",
            "layout": "5x5 separated T/O/L/J/S/Z tetromino field",
            "landmark_count": tetromino_count,
        },
    ]

    manifest_patterns: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for pattern in patterns:
        pair_dir = output_dir / pattern["folder"]
        intended_open = pattern["intended_open"]
        logical_phase = _registered_checker_grating(
            active_size,
            int(pattern["cell_size"]),
            grating_period,
            intended_open,
            orientation_mode=str(pattern["orientation_mode"]),
        )
        amplitude_path = (
            pair_dir
            / "amplitude_bmp"
            / (
                "amplitude_checker_c64_inv_1024x1024.bmp"
                if pattern["order"] == 1
                else "amplitude_tetromino_c24_inv_1024x1024.bmp"
            )
        )
        amplitude_report = _save_inverted_amplitude(
            intended_open,
            amplitude_path,
            slm_size_wh=amplitude_size,
            center_xy=amplitude_center,
        )

        preview_dir = pair_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(intended_open, mode="L").save(
            preview_dir / "intended_optical_white_regions.png", format="PNG"
        )
        Image.fromarray(
            _ideal_registration_preview(intended_open, logical_phase), mode="L"
        ).save(preview_dir / "ideal_focused_overlay_k1p0000.png", format="PNG")

        oriented_phase = logical_phase
        if flip_vertical:
            oriented_phase = np.flipud(oriented_phase)
        if flip_horizontal:
            oriented_phase = np.fliplr(oriented_phase)
        oriented_phase = np.ascontiguousarray(oriented_phase)

        phase_reports: list[dict[str, Any]] = []
        for sweep_index, k_value in enumerate(k_values):
            # Teacher's relation: n=(8/17)k*m, therefore m=17*n/(8*k).
            effective_logical_pitch_um = logical_pitch_um / k_value
            native_phase = physical_pitch_nearest(
                oriented_phase,
                logical_pixel_pitch_um=effective_logical_pitch_um,
                slm_pixel_pitch_um=phase_pitch_um,
            )
            phase_path = (
                pair_dir
                / "phase_bmp_scale_sweep"
                / (
                    f"phase_{sweep_index:02d}_k{_k_tag(k_value)}_"
                    f"{'checker_c64' if pattern['order'] == 1 else 'tetromino_c24'}_"
                    f"p{grating_period}_1920x1200.bmp"
                )
            )
            phase_report = _save_phase(
                native_phase,
                phase_path,
                slm_size_wh=phase_size,
                center_xy=phase_center,
            )
            native_width = int(native_phase.shape[1])
            realized_k = (
                logical_pitch_um * active_size / (phase_pitch_um * native_width)
            )
            target_cell_px = (
                logical_pitch_um * int(pattern["cell_size"])
                / (phase_pitch_um * k_value)
            )
            phase_report.update(
                {
                    "sweep_index": sweep_index,
                    "k": k_value,
                    "delta_from_1": k_value - 1.0,
                    "formula": "n=(8/17)*k*m; m=17*n/(8*k)",
                    "effective_logical_pitch_um": effective_logical_pitch_um,
                    "target_phase_pixels_per_amplitude_cell": target_cell_px,
                    "realized_k_from_full_active_width": realized_k,
                    "realized_k_error": realized_k - k_value,
                }
            )
            phase_reports.append(phase_report)
            csv_rows.append(
                {
                    "pattern_order": pattern["order"],
                    "pattern": pattern["name"],
                    "amplitude_bmp": str(amplitude_path),
                    "phase_bmp": str(phase_path),
                    "sweep_index": sweep_index,
                    "k": f"{k_value:.4f}",
                    "delta_from_1": f"{k_value - 1.0:+.4f}",
                    "native_phase_active_width_px": native_width,
                    "native_phase_active_height_px": int(native_phase.shape[0]),
                    "target_phase_cell_px": f"{target_cell_px:.6f}",
                    "realized_k": f"{realized_k:.8f}",
                    "realized_k_error": f"{realized_k - k_value:+.8f}",
                    "amplitude_sha256": amplitude_report["sha256"],
                    "phase_sha256": phase_report["sha256"],
                }
            )
        manifest_patterns.append(
            {
                "order": pattern["order"],
                "name": pattern["name"],
                "folder": str(pair_dir),
                "cell_size_amplitude_px": pattern["cell_size"],
                "layout": pattern["layout"],
                "landmark_count": pattern["landmark_count"],
                "amplitude": amplitude_report,
                "phase_scale_sweep": phase_reports,
                "phase_rule": (
                    "phase=0 in intended optical black regions; alternating "
                    "x/y 0-pi gratings in intended optical white regions"
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scale_sweep_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = {
        "schema_version": 1,
        "purpose": "inverted amplitude and phase magnification sweep for dual-SLM 4F registration",
        "output_dir": str(output_dir),
        "amplitude_black_white_inverted": True,
        "phase_nominal_unchanged_from_regular_c64_algorithm": True,
        "scale_formula": {
            "teacher_relation": "n=(8/17)*k*m",
            "implemented_relation": "m=17*n/(8*k)",
            "interpretation": "k>1 shrinks the native phase pattern; k<1 enlarges it",
            "values_in_play_order": k_values,
        },
        "logical": {
            "active_size": active_size,
            "amplitude_pitch_um": logical_pitch_um,
            "phase_pitch_um": phase_pitch_um,
            "binary_grating_period_logical_px": grating_period,
        },
        "amplitude_slm": {
            "size_wh": list(amplitude_size),
            "center_xy": list(amplitude_center),
        },
        "phase_slm": {
            "size_wh": list(phase_size),
            "center_xy": list(phase_center),
            "flip_vertical_before_raster": flip_vertical,
            "flip_horizontal_before_raster": flip_horizontal,
        },
        "bmp_contract": {
            "amplitude": "1024x1024 mode-L BMP; binary 0/255",
            "phase": "1920x1200 mode-L BMP; binary 0/128",
        },
        "background_subtraction": False,
        "patterns": manifest_patterns,
    }
    (output_dir / "alignment_scale_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        """# 双 SLM 反相振幅与相位倍率扫描

只使用本目录中的两个编号文件夹。每个文件夹内先固定播放唯一的
`amplitude_bmp/*.bmp`，再按 `phase_bmp_scale_sweep/phase_00...phase_20` 的编号顺序
逐张测试相位。

- `phase_00_k1p0000`：无倍率修正，规则棋盘相位与旧版逐像素相同。
- 后续顺序：`+0.0005, -0.0005, +0.0010, -0.0010, ...`，直到 `±0.0050`。
- 振幅命令图已经整画布黑白取反；不要在播放软件中再次反相。
- 相位已经沿用旧版纵向翻转；不要在播放软件中再次翻转。
- 相位中心为 `(980,590)`。
- `01_checker_c64_inv` 和 `02_tetromino_c24_inv` 的振幅/相位不能交叉配对。

老师给出的关系按 `n=(8/17)×k×m` 实现，即 `m=17n/(8k)`。每一档实际尺寸、
量化误差和 SHA256 见 `scale_sweep_manifest.csv` 与
`alignment_scale_manifest.json`。
""",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = generate(Path(args.config))
    phase_count = sum(
        len(pattern["phase_scale_sweep"]) for pattern in report["patterns"]
    )
    print(
        f"Generated {len(report['patterns'])} inverted amplitude BMPs and "
        f"{phase_count} phase scale-sweep BMPs under {report['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
