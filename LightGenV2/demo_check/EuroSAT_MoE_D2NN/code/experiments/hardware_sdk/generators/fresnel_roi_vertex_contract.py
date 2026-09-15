"""Validate the corrected Fresnel ROI-vertex calibration payload.

The historical Fresnel array places lenses at quadrant centres.  It is useful
as an internal control-point pattern, but its 508-pixel spacing is not the
physical 478 x 17 um support.  This module defines the stricter transfer
contract used by the Qwen laboratory bundles: n1 at the optical-axis centre,
n4 at the four exact support vertices, and n9 at vertices, edge midpoints and
centre.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


CALIBRATION_DIRECTORY_NAME = "fresnel_roi_vertex_array_532nm_17um_8um_v2"
CALIBRATION_ARCHIVE_ROOT = f"payload/calibration/{CALIBRATION_DIRECTORY_NAME}"
PROPAGATION_DISTANCES_CM = (5, 10, 15)
LAYOUTS = {
    1: "center",
    4: "exact_roi_vertices",
    9: "exact_roi_vertices_edge_midpoints_center",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required corrected Fresnel {label} is missing: {path}")
    return path


def _require_close(observed: Any, expected: float, label: str) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid corrected Fresnel {label}: {observed!r}") from error
    if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(
            f"Corrected Fresnel {label} mismatch: observed={value}, expected={expected}"
        )


def _require_sequence(observed: Any, expected: list[float | int], label: str) -> None:
    if not isinstance(observed, list) or len(observed) != len(expected):
        raise RuntimeError(
            f"Corrected Fresnel {label} mismatch: observed={observed!r}, expected={expected!r}"
        )
    for index, (value, target) in enumerate(zip(observed, expected, strict=True)):
        _require_close(value, float(target), f"{label}[{index}]")


def _validate_bmp(path: Path, expected_size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        if image.format != "BMP" or image.mode != "L" or image.size != expected_size:
            raise RuntimeError(
                f"Invalid corrected Fresnel BMP {path}: format={image.format}, "
                f"mode={image.mode}, size={image.size}, expected={expected_size}"
            )


def _expected_phase_names() -> set[str]:
    return {
        (
            f"phase_fresnel_n{count}_{layout}_z{distance}cm_"
            "532nm_8um_1920x1200.bmp"
        )
        for distance in PROPAGATION_DISTANCES_CM
        for count, layout in LAYOUTS.items()
    }


def _read_focus_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_focus_targets(path: Path, expected_phase_names: set[str]) -> None:
    rows = _read_focus_rows(path)
    if len(rows) != 3 * (1 + 4 + 9):
        raise RuntimeError(
            "Corrected Fresnel focus target CSV must contain 42 rows "
            "(n1+n4+n9 for 5/10/15 cm)"
        )
    observed_names = {row.get("phase_bmp", "") for row in rows}
    if observed_names != expected_phase_names:
        raise RuntimeError("Corrected Fresnel focus target CSV has the wrong phase set")

    logical_vertices = {
        (472.125, 82.125),
        (1487.875, 82.125),
        (472.125, 1097.875),
        (1487.875, 1097.875),
    }
    exported_vertices = {
        (472.125, 1097.875),
        (1487.875, 1097.875),
        (472.125, 82.125),
        (1487.875, 82.125),
    }
    x_axis = {472.125, 980.0, 1487.875}
    y_axis = {82.125, 590.0, 1097.875}
    for distance in PROPAGATION_DISTANCES_CM:
        distance_rows = [
            row for row in rows if float(row["propagation_cm"]) == float(distance)
        ]
        by_count = {
            count: [row for row in distance_rows if int(row["array_count"]) == count]
            for count in LAYOUTS
        }
        if {count: len(items) for count, items in by_count.items()} != {1: 1, 4: 4, 9: 9}:
            raise RuntimeError(f"Corrected Fresnel focus row count mismatch at {distance} cm")
        n1 = by_count[1][0]
        _require_close(n1["logical_target_phase_edge_x"], 980.0, "n1 logical x")
        _require_close(n1["logical_target_phase_edge_y"], 590.0, "n1 logical y")
        _require_close(n1["exported_target_bmp_edge_x"], 980.0, "n1 exported x")
        _require_close(n1["exported_target_bmp_edge_y"], 590.0, "n1 exported y")

        observed_logical = {
            (
                float(row["logical_target_phase_edge_x"]),
                float(row["logical_target_phase_edge_y"]),
            )
            for row in by_count[4]
        }
        observed_exported = {
            (
                float(row["exported_target_bmp_edge_x"]),
                float(row["exported_target_bmp_edge_y"]),
            )
            for row in by_count[4]
        }
        if observed_logical != logical_vertices or observed_exported != exported_vertices:
            raise RuntimeError(
                f"Corrected Fresnel n4 targets are not the exact ROI vertices at {distance} cm"
            )
        n9_logical = {
            (
                float(row["logical_target_phase_edge_x"]),
                float(row["logical_target_phase_edge_y"]),
            )
            for row in by_count[9]
        }
        n9_exported = {
            (
                float(row["exported_target_bmp_edge_x"]),
                float(row["exported_target_bmp_edge_y"]),
            )
            for row in by_count[9]
        }
        if n9_logical != {(x, y) for y in y_axis for x in x_axis}:
            raise RuntimeError(f"Corrected Fresnel n9 logical grid mismatch at {distance} cm")
        if n9_exported != {(x, y) for y in y_axis for x in x_axis}:
            raise RuntimeError(f"Corrected Fresnel n9 exported grid mismatch at {distance} cm")


def validate_fresnel_roi_vertex_calibration(
    calibration_dir: str | Path,
) -> tuple[list[Path], dict[str, Any]]:
    """Return the whitelisted payload files and a verified physical contract."""

    root = Path(calibration_dir).expanduser().resolve()
    manifest_path = _require_file(
        root / "fresnel_roi_vertex_manifest.json", "manifest"
    )
    focus_path = _require_file(root / "fresnel_focus_targets.csv", "focus target CSV")
    validation_json_path = _require_file(
        root / "numerical_focus_validation.json", "numerical validation JSON"
    )
    validation_csv_path = _require_file(
        root / "numerical_focus_validation.csv", "numerical validation CSV"
    )
    readme_path = _require_file(root / "README.md", "README")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) < 2:
        raise RuntimeError("Corrected Fresnel manifest schema must be >=2")
    if manifest.get("purpose") != "exact physical ROI-vertex Fresnel focus/ROI calibration":
        raise RuntimeError("Fresnel payload is not the corrected ROI-vertex calibration")

    support = manifest.get("shared_optical_support", {})
    _require_close(support.get("amplitude_active_size_px"), 478, "active size")
    _require_close(support.get("amplitude_pixel_pitch_um"), 17.0, "amplitude pitch")
    _require_close(support.get("exact_physical_width_um"), 8126.0, "physical width")
    _require_close(support.get("phase_pixel_pitch_um"), 8.0, "phase pitch")
    _require_close(
        support.get("exact_phase_support_width_px"), 1015.75, "phase support width"
    )
    _require_sequence(
        support.get("exact_phase_support_bounds_edge_xyxy"),
        [472.125, 82.125, 1487.875, 1097.875],
        "exact support bounds",
    )
    _require_sequence(
        support.get("quantized_phase_active_bounds_half_open_xyxy"),
        [472, 82, 1488, 1098],
        "quantized support bounds",
    )
    _require_sequence(
        support.get("phase_center_edge_xy"), [980.0, 590.0], "phase centre"
    )
    phase_slm = manifest.get("phase_slm", {})
    _require_sequence(phase_slm.get("size_wh"), [1920, 1200], "phase SLM size")
    if phase_slm.get("flip_vertical_before_export") is not True:
        raise RuntimeError("Corrected Fresnel phase BMP must include the vertical export flip")
    if phase_slm.get("flip_horizontal_before_export") is not False:
        raise RuntimeError("Corrected Fresnel phase BMP must not include a horizontal export flip")
    sampling_aperture = manifest.get("array_design", {}).get(
        "sampling_safe_aperture", {}
    )
    _require_close(
        sampling_aperture.get("safety_factor"), 0.92, "Nyquist safety factor"
    )
    if sampling_aperture.get("shape") != "circle intersected with each inward cell":
        raise RuntimeError("Corrected Fresnel sampling-safe aperture shape mismatch")
    if sampling_aperture.get("outside_mode") != "flat_zero":
        raise RuntimeError("Corrected Fresnel outside-aperture phase must be flat zero")
    if manifest.get("background_subtraction") is not False:
        raise RuntimeError("Corrected Fresnel calibration must not claim background subtraction")

    expected_phase_names = _expected_phase_names()
    phase_rows = manifest.get("files")
    if not isinstance(phase_rows, list) or len(phase_rows) != 9:
        raise RuntimeError("Corrected Fresnel manifest must declare exactly nine phase BMPs")
    records = {Path(row.get("file", {}).get("path", "")).name: row for row in phase_rows}
    if set(records) != expected_phase_names:
        raise RuntimeError("Corrected Fresnel manifest phase filenames do not match n1/n4/n9")
    phase_paths: list[Path] = []
    for name in sorted(expected_phase_names):
        path = _require_file(root / "phase_bmp" / name, "phase BMP")
        _validate_bmp(path, (1920, 1200))
        record = records[name]
        if record.get("numerical_validation_passed") is not True:
            raise RuntimeError(f"Numerical validation is not passed for {name}")
        declared_sha = str(record.get("file", {}).get("sha256", "")).lower()
        if _sha256(path) != declared_sha:
            raise RuntimeError(f"Corrected Fresnel phase SHA-256 mismatch: {name}")
        phase_paths.append(path)

    amplitude = manifest.get("amplitude", {})
    if amplitude.get("polarity") != "255=white/bright/transmissive; 0=black/dark/blocked":
        raise RuntimeError("Corrected Fresnel amplitude polarity is not the normal SLM polarity")
    full_path = _require_file(
        root / "amplitude_bmp" / "amplitude_focus_full_white_1024x1024.bmp",
        "full-white amplitude BMP",
    )
    roi_path = _require_file(
        root / "amplitude_bmp" / "amplitude_roi478_white_black_1024x1024.bmp",
        "478-white-window amplitude BMP",
    )
    for label, path in (("focus_full_white", full_path), ("roi_window", roi_path)):
        _validate_bmp(path, (1024, 1024))
        declared_sha = str(amplitude.get(label, {}).get("sha256", "")).lower()
        if _sha256(path) != declared_sha:
            raise RuntimeError(f"Corrected Fresnel amplitude SHA-256 mismatch: {path.name}")
    with Image.open(full_path) as image:
        histogram = image.histogram()
        if histogram[255] != 1024 * 1024 or sum(histogram[:255]) != 0:
            raise RuntimeError("Focus amplitude BMP is not full-frame 255")
    with Image.open(roi_path) as image:
        histogram = image.histogram()
        bright = 478 * 478
        if histogram[255] != bright or histogram[0] != 1024 * 1024 - bright:
            raise RuntimeError("ROI amplitude BMP is not exactly one central 478x478 white window")
        if sum(histogram[1:255]) != 0:
            raise RuntimeError("ROI amplitude BMP contains non-binary pixel values")
    _require_sequence(
        amplitude.get("roi_window", {}).get("active_bounds_edge_xyxy"),
        [273, 273, 751, 751],
        "amplitude ROI bounds",
    )

    _validate_focus_targets(focus_path, expected_phase_names)
    validation = json.loads(validation_json_path.read_text(encoding="utf-8"))
    if not isinstance(validation, list) or len(validation) != 9:
        raise RuntimeError("Corrected Fresnel numerical validation must contain nine cases")
    if any(row.get("passed") is not True for row in validation):
        raise RuntimeError("Corrected Fresnel numerical propagation validation failed")
    if {row.get("phase_bmp") for row in validation} != expected_phase_names:
        raise RuntimeError("Corrected Fresnel numerical validation phase set mismatch")
    if max(float(row.get("max_abs_position_error_phase_px", math.inf)) for row in validation) > 0.75:
        raise RuntimeError("Corrected Fresnel focus error exceeds 0.75 phase pixel")
    for row in validation:
        name = str(row["phase_bmp"])
        if row.get("unique_peak_assignment") is not True:
            raise RuntimeError(f"Corrected Fresnel target peaks are not unique: {name}")
        observed_ratio = float(
            row.get("minimum_target_peak_to_max_outside_targets", -math.inf)
        )
        required_ratio = float(
            row.get("acceptance_minimum_target_peak_to_max_outside_targets", math.inf)
        )
        if observed_ratio < required_ratio:
            raise RuntimeError(f"Corrected Fresnel off-target peak validation failed: {name}")
        if str(row.get("phase_sha256", "")).lower() != _sha256(
            root / "phase_bmp" / name
        ):
            raise RuntimeError(f"Corrected Fresnel validation SHA-256 mismatch: {name}")

    preview_paths = [
        _require_file(root / "preview" / Path(name).with_suffix(".png").name, "preview")
        for name in sorted(expected_phase_names)
    ]
    selected = [
        readme_path,
        manifest_path,
        focus_path,
        validation_json_path,
        validation_csv_path,
        full_path,
        roi_path,
        *phase_paths,
        *preview_paths,
    ]
    contract = {
        "version": "roi_vertex_v2",
        "formal_roi_calibration": True,
        "historical_quadrant_center_arrays_formal": False,
        "wavelength_nm": 532.0,
        "available_propagation_distances_cm": list(PROPAGATION_DISTANCES_CM),
        "qwen_nominal_propagation_distance_cm": 10.0,
        "n1_use": "find the focal plane at the shared optical-axis centre",
        "n4_use": "four foci lie directly on the exact physical ROI vertices; do not extrapolate",
        "n9_use": "vertices, edge midpoints, and centre across the full ROI",
        "amplitude_roi": "central 478x478 pixels are 255; outside is 0",
        "phase_export_flip": {"vertical": True, "horizontal": False},
        "phase_center_edge_xy": [980.0, 590.0],
        "logical_roi_vertex_spacing_phase_px": [1015.75, 1015.75],
        "numerical_validation": "passed; maximum centroid/index error <=0.75 phase pixel",
        "nyquist_safe_aperture": {
            "shape": "circle intersected with each inward cell",
            "safety_factor": 0.92,
            "outside_phase": "flat zero",
        },
        "calibration_manifest_sha256": _sha256(manifest_path),
        "vendor_sdk_examples_are_calibration_evidence": False,
        "phase_bmp_count": 9,
    }
    return selected, contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the corrected n1/n4/n9 physical ROI-vertex calibration"
    )
    parser.add_argument("--calibration-dir", required=True)
    args = parser.parse_args(argv)
    files, contract = validate_fresnel_roi_vertex_calibration(args.calibration_dir)
    print(
        json.dumps(
            {
                "status": "passed",
                "calibration_dir": str(
                    Path(args.calibration_dir).expanduser().resolve()
                ),
                "selected_files": len(files),
                "contract": contract,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
