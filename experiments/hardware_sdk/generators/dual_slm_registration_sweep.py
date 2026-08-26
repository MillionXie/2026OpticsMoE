"""Generate polarity-explicit dual-SLM registration pairs and phase scale sweeps."""

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
    _binary_grating,
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


def phase_scale_values(
    max_abs_delta: float | None = None,
    step: float | None = None,
    *,
    absolute_deltas: list[float] | tuple[float, ...] | None = None,
) -> list[float]:
    """Return 1 first, then positive/negative scale corrections outwards."""

    if absolute_deltas is not None:
        if max_abs_delta is not None or step is not None:
            raise ValueError("absolute_deltas cannot be combined with max_abs_delta/step")
        deltas = [round(float(value), 7) for value in absolute_deltas]
        if (
            not deltas
            or any(value <= 0.0 or value > 0.1 + 1.0e-12 for value in deltas)
            or deltas != sorted(set(deltas))
        ):
            raise ValueError(
                "absolute_deltas must be unique, increasing, and within (0, 0.1]"
            )
    else:
        if max_abs_delta is None or step is None:
            raise ValueError("provide absolute_deltas or max_abs_delta and step")
        max_abs_delta = float(max_abs_delta)
        step = float(step)
        if not 0.0 < step <= max_abs_delta <= 0.1 + 1.0e-12:
            raise ValueError("require 0 < step <= max_abs_delta <= 0.1")
        count = int(round(max_abs_delta / step))
        if not np.isclose(count * step, max_abs_delta, atol=1.0e-12):
            raise ValueError("max_abs_delta must be an integer multiple of step")
        deltas = [round(index * step, 7) for index in range(1, count + 1)]
    values = [1.0]
    for delta in deltas:
        values.extend((1.0 + delta, 1.0 - delta))
    return [round(value, 7) for value in values]


def large_block_mask(size: int, cell_size: int) -> tuple[np.ndarray, list[int]]:
    """Return six plain, large connected regions made from 4--9 cells."""

    full_cells = 9
    logical_extent = full_cells * int(cell_size)
    if logical_extent > int(size):
        raise ValueError("large block layout exceeds the logical active area")
    shapes = (
        ((0, 0), (0, 1), (1, 0), (1, 1)),  # 2x2 square: 4 cells
        ((0, 4), (0, 5), (0, 6), (1, 4), (1, 5), (1, 6)),  # 2x3: 6
        ((3, 0), (4, 0), (5, 0), (5, 1), (5, 2)),  # plain L: 5
        tuple((row, column) for row in range(3, 6) for column in range(4, 7)),
        ((7, 0), (7, 1), (7, 2), (8, 0), (8, 1), (8, 2)),  # 2x3: 6
        ((7, 5), (7, 6), (8, 5), (8, 6)),  # 2x2 square: 4
    )
    cells = np.zeros((full_cells, full_cells), dtype=np.uint8)
    cell_counts: list[int] = []
    for shape in shapes:
        for row, column in shape:
            cells[row, column] = 255
        cell_counts.append(len(shape))
    expanded = np.repeat(np.repeat(cells, cell_size, axis=0), cell_size, axis=1)
    result = np.zeros((size, size), dtype=np.uint8)
    offset = (size - logical_extent) // 2
    result[offset : offset + logical_extent, offset : offset + logical_extent] = expanded
    return result, cell_counts


def single_axis_masked_grating(
    intended_open_mask: np.ndarray,
    period: int,
    axis: str,
) -> np.ndarray:
    """Use one continuous grating direction across every open region."""

    if intended_open_mask.ndim != 2 or not np.isin(
        intended_open_mask, (0, 255)
    ).all():
        raise ValueError("intended_open_mask must be a binary 2-D uint8 array")
    height, width = intended_open_mask.shape
    phase = _binary_grating(height, width, period, axis)
    phase[intended_open_mask == 0] = 0
    return phase.astype(np.uint8)


def _save_amplitude(
    intended_open_mask: np.ndarray,
    path: Path,
    *,
    slm_size_wh: tuple[int, int],
    center_xy: tuple[float, float],
    invert_before_export: bool,
) -> dict[str, Any]:
    intended_canvas, bounds, actual_center = place_at_center(
        Image.fromarray(intended_open_mask, mode="L"),
        slm_size_wh=slm_size_wh,
        center_xy=center_xy,
    )
    intended = np.asarray(intended_canvas, dtype=np.uint8)
    commanded = 255 - intended if invert_before_export else intended
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(commanded, mode="L").save(path, format="BMP")
    _validate_bmp(path, slm_size_wh, {0, 255})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "encoding": (
            "full_canvas_black_white_inverse"
            if invert_before_export
            else "direct_uint8_255_bright_0_dark"
        ),
        "invert_before_export": bool(invert_before_export),
        "bright_value_uint8": 0 if invert_before_export else 255,
        "dark_value_uint8": 255 if invert_before_export else 0,
        "active_bounds_xyxy": list(bounds),
        "actual_center_xy": list(actual_center),
        "commanded_black_fraction": float(np.mean(commanded == 0)),
        "commanded_white_fraction": float(np.mean(commanded == 255)),
    }


def _save_inverted_amplitude(
    intended_open_mask: np.ndarray,
    path: Path,
    *,
    slm_size_wh: tuple[int, int],
    center_xy: tuple[float, float],
) -> dict[str, Any]:
    """Compatibility wrapper for historical inverted-polarity bundles."""

    return _save_amplitude(
        intended_open_mask,
        path,
        slm_size_wh=slm_size_wh,
        center_xy=center_xy,
        invert_before_export=True,
    )


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
    invert_amplitude = bool(amplitude_config.get("invert_before_export", False))
    polarity_tag = "inv" if invert_amplitude else "normal"
    phase_size = tuple(int(value) for value in phase_config["size_wh"])
    phase_center = tuple(float(value) for value in phase_config["center_xy"])
    phase_pitch_um = float(phase_config["pixel_pitch_um"])
    flip_vertical = bool(phase_config.get("flip_vertical", True))
    flip_horizontal = bool(phase_config.get("flip_horizontal", False))
    if sweep.get("absolute_deltas") is not None:
        k_values = phase_scale_values(
            absolute_deltas=tuple(float(value) for value in sweep["absolute_deltas"])
        )
    else:
        k_values = phase_scale_values(
            float(sweep["max_abs_delta"]), float(sweep["step"])
        )

    regular_open = _checker(active_size, 64)
    large_open, large_cell_counts = large_block_mask(active_size, 48)
    patterns: list[dict[str, Any]] = [
        {
            "order": 1,
            "name": "regular_checker_c64",
            "folder": f"01_checker_c64_{polarity_tag}",
            "cell_size": 64,
            "intended_open": regular_open,
            "orientation_mode": "visible_checker_cells",
            "layout": (
                "strict checker; command BMP is the exact full-canvas inverse"
                if invert_amplitude
                else "strict checker; command 255 is optically open"
            ),
            "landmark_count": None,
            "phase_mode": "legacy_xy",
        },
        {
            "order": 2,
            "name": "large_blocks_c48_4to9cells",
            "folder": f"02_large_blocks_c48_{polarity_tag}",
            "cell_size": 48,
            "intended_open": large_open,
            "layout": "six plain square/rectangle/L regions made from 4--9 cells",
            "landmark_count": len(large_cell_counts),
            "landmark_cell_counts": large_cell_counts,
            "phase_mode": "separate_x_y",
        },
    ]

    manifest_patterns: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for pattern in patterns:
        pair_dir = output_dir / pattern["folder"]
        intended_open = pattern["intended_open"]
        if pattern["phase_mode"] == "legacy_xy":
            phase_variants = {
                "legacy_xy": _registered_checker_grating(
                    active_size,
                    int(pattern["cell_size"]),
                    grating_period,
                    intended_open,
                    orientation_mode="visible_checker_cells",
                )
            }
        else:
            phase_variants = {
                "x": single_axis_masked_grating(
                    intended_open, grating_period, "x"
                ),
                "y": single_axis_masked_grating(
                    intended_open, grating_period, "y"
                ),
            }
        amplitude_path = (
            pair_dir
            / "amplitude_bmp"
            / (
                f"amplitude_checker_c64_{polarity_tag}_1024x1024.bmp"
                if pattern["order"] == 1
                else f"amplitude_large_blocks_c48_{polarity_tag}_1024x1024.bmp"
            )
        )
        amplitude_report = _save_amplitude(
            intended_open,
            amplitude_path,
            slm_size_wh=amplitude_size,
            center_xy=amplitude_center,
            invert_before_export=invert_amplitude,
        )

        preview_dir = pair_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(intended_open, mode="L").save(
            preview_dir / "intended_optical_white_regions.png", format="PNG"
        )
        phase_reports: list[dict[str, Any]] = []
        for grating_axis, logical_phase in phase_variants.items():
            Image.fromarray(
                _ideal_registration_preview(intended_open, logical_phase), mode="L"
            ).save(
                preview_dir / f"ideal_focused_overlay_{grating_axis}_k1p0000.png",
                format="PNG",
            )
            oriented_phase = logical_phase
            if flip_vertical:
                oriented_phase = np.flipud(oriented_phase)
            if flip_horizontal:
                oriented_phase = np.fliplr(oriented_phase)
            oriented_phase = np.ascontiguousarray(oriented_phase)

            phase_subdir = (
                "phase_bmp_scale_sweep"
                if grating_axis == "legacy_xy"
                else f"phase_bmp_scale_sweep_{grating_axis}"
            )
            for sweep_index, k_value in enumerate(k_values):
                # Teacher's relation: n=(8/17)k*m, therefore m=17*n/(8*k).
                effective_logical_pitch_um = logical_pitch_um / k_value
                native_phase = physical_pitch_nearest(
                    oriented_phase,
                    logical_pixel_pitch_um=effective_logical_pitch_um,
                    slm_pixel_pitch_um=phase_pitch_um,
                )
                pattern_tag = (
                    "checker_c64"
                    if pattern["order"] == 1
                    else "large_blocks_c48"
                )
                axis_tag = "" if grating_axis == "legacy_xy" else f"_{grating_axis}"
                phase_path = (
                    pair_dir
                    / phase_subdir
                    / (
                        f"phase_{sweep_index:02d}_k{_k_tag(k_value)}_"
                        f"{pattern_tag}{axis_tag}_p{grating_period}_1920x1200.bmp"
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
                        "grating_axis": grating_axis,
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
                        "grating_axis": grating_axis,
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
                "landmark_cell_counts": pattern.get("landmark_cell_counts"),
                "amplitude": amplitude_report,
                "phase_scale_sweep": phase_reports,
                "phase_rule": (
                    "regular checker preserves the previous phase; large blocks "
                    "use separate globally x-only and y-only 0-pi phase files"
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
        "schema_version": 2,
        "purpose": "polarity-explicit amplitude and phase magnification sweep for dual-SLM 4F registration",
        "output_dir": str(output_dir),
        "amplitude_black_white_inverted": invert_amplitude,
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
            "invert_before_export": invert_amplitude,
            "bright_value_uint8": 0 if invert_amplitude else 255,
            "dark_value_uint8": 255 if invert_amplitude else 0,
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
        f"""# 双 SLM 振幅与相位倍率扫描（{polarity_tag}）

只使用本目录中的两个编号文件夹。每个文件夹内先固定播放唯一的
`amplitude_bmp/*.bmp`，再按 `phase_bmp_scale_sweep/phase_00...phase_40` 的编号顺序
逐张测试相位。

- `phase_00_k1p0000`：无倍率修正，规则棋盘相位与旧版逐像素相同。
- 精细段：`+0.0005, -0.0005, ...`，直到 `±0.0050`。
- 大范围段：`+0.0100, -0.0100, ...`，直到 `±0.1000`。
- 振幅硬件合同：`255=白/透光`、`0=黑/遮光`；本目录
  `invert_before_export={str(invert_amplitude).lower()}`，不要在播放软件中另行反相。
- 相位已经沿用旧版纵向翻转；不要在播放软件中再次翻转。
- 相位中心为 `(980,590)`。
- `02_large_blocks_c48_{polarity_tag}` 的 X/Y 相位分别位于两个目录；单张相位只有一个方向。
- `01_checker_c64_{polarity_tag}` 和 `02_large_blocks_c48_{polarity_tag}` 的振幅/相位不能交叉配对。

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
        f"Generated {len(report['patterns'])} amplitude BMPs and "
        f"{phase_count} phase scale-sweep BMPs under {report['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
