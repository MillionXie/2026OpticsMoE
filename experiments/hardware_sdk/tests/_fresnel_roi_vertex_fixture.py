"""Small deterministic fixture for laboratory bundle contract tests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from experiments.hardware_sdk.generators.fresnel_roi_vertex_contract import (
    LAYOUTS,
    PROPAGATION_DISTANCES_CM,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_bmp(path: Path, size: tuple[int, int], value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=value).save(path, format="BMP")


def make_fresnel_roi_vertex_fixture(root: Path) -> Path:
    calibration = root / "fresnel_roi_vertex_v2"
    full = calibration / "amplitude_bmp" / "amplitude_focus_full_white_1024x1024.bmp"
    roi = calibration / "amplitude_bmp" / "amplitude_roi478_white_black_1024x1024.bmp"
    _save_bmp(full, (1024, 1024), 255)
    roi.parent.mkdir(parents=True, exist_ok=True)
    roi_image = Image.new("L", (1024, 1024), color=0)
    roi_image.paste(255, (273, 273, 751, 751))
    roi_image.save(roi, format="BMP")

    file_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    focus_rows: list[dict[str, object]] = []
    x_axes_by_count = {
        1: [980.0],
        4: [472.125, 1487.875],
        9: [472.125, 980.0, 1487.875],
    }
    y_axes_by_count = {
        1: [590.0],
        4: [82.125, 1097.875],
        9: [82.125, 590.0, 1097.875],
    }
    local_by_count = {
        1: [508.0],
        4: [0.125, 1015.875],
        9: [0.125, 508.0, 1015.875],
    }
    for distance in PROPAGATION_DISTANCES_CM:
        for count, layout in LAYOUTS.items():
            name = (
                f"phase_fresnel_n{count}_{layout}_z{distance}cm_"
                "532nm_8um_1920x1200.bmp"
            )
            phase = calibration / "phase_bmp" / name
            _save_bmp(phase, (1920, 1200), value=(count + distance) % 256)
            preview = calibration / "preview" / Path(name).with_suffix(".png").name
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_bytes(b"preview")
            phase_sha = _sha256(phase)
            file_rows.append(
                {
                    "array_count": count,
                    "grid_size": int(count**0.5),
                    "layout": layout,
                    "propagation_cm": float(distance),
                    "file": {"path": str(phase), "sha256": phase_sha},
                    "numerical_validation_passed": True,
                }
            )
            validation_rows.append(
                {
                    "phase_bmp": name,
                    "passed": True,
                    "max_abs_position_error_phase_px": 0.5,
                    "unique_peak_assignment": True,
                    "minimum_target_peak_to_max_outside_targets": 100.0,
                    "acceptance_minimum_target_peak_to_max_outside_targets": 50.0,
                    "phase_sha256": phase_sha,
                }
            )
            x_axes = x_axes_by_count[count]
            y_axes = y_axes_by_count[count]
            locals_ = local_by_count[count]
            for row_index, (phase_y, local_y) in enumerate(
                zip(y_axes, locals_, strict=True)
            ):
                for column_index, (phase_x, local_x) in enumerate(
                    zip(x_axes, locals_, strict=True)
                ):
                    focus_rows.append(
                        {
                            "phase_bmp": name,
                            "array_count": count,
                            "propagation_cm": float(distance),
                            "logical_row": row_index,
                            "logical_column": column_index,
                            "logical_target_phase_edge_x": phase_x,
                            "logical_target_phase_edge_y": phase_y,
                            "exported_target_bmp_edge_x": phase_x,
                            # The established export flips the logical active
                            # support about its configured y centre (590), not
                            # about the 1200-row full-frame centre.
                            "exported_target_bmp_edge_y": 1180.0 - phase_y,
                            "logical_target_local_edge_x": local_x,
                            "logical_target_local_edge_y": local_y,
                        }
                    )

    focus_path = calibration / "fresnel_focus_targets.csv"
    focus_path.parent.mkdir(parents=True, exist_ok=True)
    with focus_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(focus_rows[0]))
        writer.writeheader()
        writer.writerows(focus_rows)
    (calibration / "numerical_focus_validation.json").write_text(
        json.dumps(validation_rows, indent=2), encoding="utf-8"
    )
    with (calibration / "numerical_focus_validation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)
    (calibration / "README.md").write_text("corrected ROI vertices\n", encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "purpose": "exact physical ROI-vertex Fresnel focus/ROI calibration",
        "amplitude": {
            "polarity": "255=white/bright/transmissive; 0=black/dark/blocked",
            "focus_full_white": {"sha256": _sha256(full)},
            "roi_window": {
                "sha256": _sha256(roi),
                "active_bounds_edge_xyxy": [273, 273, 751, 751],
            },
        },
        "shared_optical_support": {
            "amplitude_active_size_px": 478,
            "amplitude_pixel_pitch_um": 17.0,
            "exact_physical_width_um": 8126.0,
            "phase_pixel_pitch_um": 8.0,
            "exact_phase_support_width_px": 1015.75,
            "exact_phase_support_bounds_edge_xyxy": [
                472.125,
                82.125,
                1487.875,
                1097.875,
            ],
            "quantized_phase_active_bounds_half_open_xyxy": [472, 82, 1488, 1098],
            "phase_center_edge_xy": [980.0, 590.0],
        },
        "phase_slm": {
            "size_wh": [1920, 1200],
            "flip_vertical_before_export": True,
            "flip_horizontal_before_export": False,
        },
        "array_design": {
            "sampling_safe_aperture": {
                "safety_factor": 0.92,
                "shape": "circle intersected with each inward cell",
                "outside_mode": "flat_zero",
            }
        },
        "background_subtraction": False,
        "files": file_rows,
    }
    (calibration / "fresnel_roi_vertex_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return calibration


def make_dual_slm_checker_grating_fixture(root: Path) -> Path:
    pair = root / "dual_slm_checker_grating"
    amplitude = pair / "amplitude_checker_255open_c64_1024x1024.bmp"
    phase = pair / "phase_grating_xy_in_255open_cells_c64_p8_1920x1200.bmp"
    amplitude.parent.mkdir(parents=True, exist_ok=True)
    checker = Image.new("L", (1024, 1024), 0)
    checker.paste(255, (273, 273, 512, 751))
    checker.save(amplitude, format="BMP")
    _save_bmp(phase, (1920, 1200), value=128)
    manifest = {
        "schema_version": 1,
        "pair_id": "recommended_checker_grating_pair",
        "use_only_as_a_pair": True,
        "amplitude_command_contract": {
            "white_open_value_uint8": 255,
            "black_closed_value_uint8": 0,
            "invert_in_player": False,
        },
        "phase_rule": (
            "phase=0 in amplitude-0 black/closed cells; visible amplitude-255 "
            "white/open cells contain alternating x/y gratings"
        ),
        "phase_transform": {
            "center_xy": [980.0, 590.0],
            "flip_vertical_before_raster": True,
            "flip_horizontal_before_raster": False,
        },
        "amplitude": {
            "path": str(amplitude),
            "sha256": _sha256(amplitude),
            "active_bounds_xyxy": [273, 273, 751, 751],
        },
        "phase": {
            "path": str(phase),
            "sha256": _sha256(phase),
            "active_bounds_xyxy": [472, 82, 1488, 1098],
        },
    }
    (pair / "pair_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return pair
