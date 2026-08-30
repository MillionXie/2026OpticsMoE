"""Generate all camera/SLM runtime files from one small laboratory config.

The operator edits only ``experiments/lab_qwen/LAB_CONFIG.yaml``.  Measured
optical corners may be arbitrary CCD pixel coordinates.  This module derives a
TUCam-compatible bounding ROI (left/top/height on four-pixel boundaries and
width on an eight-pixel boundary), creates and hashes the homography contract,
and writes immutable generated runtime configs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.hardware_sdk.workflows.detector_homography import (
    POINT_LABELS,
    build_geometry_contract,
    load_geometry_contract,
    write_geometry_contract,
)


SENSOR_SIZE_WH = (2048, 2048)
TARGET_SIZE_WH = (478, 478)
ROI_OFFSET_ALIGNMENT = 4
ROI_WIDTH_ALIGNMENT = 8
ROI_HEIGHT_ALIGNMENT = 4
ROI_MARGIN_PX = 64
LUT_RELATIVE_DIRECTORY = Path(
    "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark/LUT Files"
)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _validate_lut_filename(value: Any) -> str:
    filename = str(value or "").strip()
    if not filename:
        raise ValueError("LAB_CONFIG.yaml: amplitude_lut_filename cannot be empty")
    if Path(filename).name != filename or any(mark in filename for mark in ("/", "\\", ":")):
        raise ValueError(
            "LAB_CONFIG.yaml: amplitude_lut_filename must contain only the LUT "
            "file name, not a directory. Put the file under the Meadowlark "
            "'LUT Files' directory."
        )
    if Path(filename).suffix.lower() != ".lut":
        raise ValueError("LAB_CONFIG.yaml: amplitude_lut_filename must end in .lut")
    return filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_optional_sha256(value: Any, *, label: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"LAB_CONFIG.yaml: {label} must be one 64-character SHA256")
    return digest


def _parse_corners(value: Any) -> dict[str, list[float]] | None:
    if not isinstance(value, Mapping):
        raise ValueError(
            "LAB_CONFIG.yaml: logical_corners_full_sensor_xy must map TL/TR/BR/BL labels"
        )
    missing = [label for label in POINT_LABELS if label not in value]
    extra = sorted(set(value) - set(POINT_LABELS))
    if missing or extra:
        raise ValueError(
            "LAB_CONFIG.yaml corner labels mismatch: "
            f"missing={missing}, unexpected={extra}"
        )
    null_labels = [label for label in POINT_LABELS if value[label] is None]
    if len(null_labels) == len(POINT_LABELS):
        return None
    if null_labels:
        raise ValueError(
            "Either fill all four logical corners or set all four to null; "
            f"currently null={null_labels}"
        )
    sensor_width, sensor_height = SENSOR_SIZE_WH
    result: dict[str, list[float]] = {}
    for label in POINT_LABELS:
        point = value[label]
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 2
        ):
            raise ValueError(f"corner {label} must be [x,y] or null")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"corner {label} contains NaN/Inf")
        if not (0.0 <= x <= sensor_width - 1 and 0.0 <= y <= sensor_height - 1):
            raise ValueError(
                f"corner {label}={[x, y]} is outside the 2048x2048 full sensor"
            )
        result[label] = [x, y]
    return result


def derive_device_roi(
    corners: Mapping[str, Sequence[float]],
    *,
    sensor_size_wh: tuple[int, int] = SENSOR_SIZE_WH,
    margin_px: int = ROI_MARGIN_PX,
    offset_alignment: int = ROI_OFFSET_ALIGNMENT,
    width_alignment: int = ROI_WIDTH_ALIGNMENT,
    height_alignment: int = ROI_HEIGHT_ALIGNMENT,
) -> list[int]:
    """Return an exact TUCam ROI enclosing arbitrary measured corners.

    The current 2048x2048 TUCam accepts left/top offsets and height in
    four-pixel increments, but its ROI width is quantized to eight pixels.  If
    width is only four-pixel aligned the SDK silently rounds it down (for
    example 1404 -> 1400), which invalidates the detector homography contract.
    We therefore round the bounding width *outward* to eight pixels here.
    """

    alignments = (offset_alignment, width_alignment, height_alignment)
    if margin_px < 0 or any(value <= 0 for value in alignments):
        raise ValueError("margin_px must be non-negative and alignments positive")
    sensor_width, sensor_height = sensor_size_wh
    if sensor_width % width_alignment or sensor_height % height_alignment:
        raise ValueError("sensor dimensions must be divisible by ROI size alignment")
    xs = [float(corners[label][0]) for label in POINT_LABELS]
    ys = [float(corners[label][1]) for label in POINT_LABELS]

    left = max(
        0,
        math.floor((min(xs) - margin_px) / offset_alignment) * offset_alignment,
    )
    top = max(
        0,
        math.floor((min(ys) - margin_px) / offset_alignment) * offset_alignment,
    )
    # +1 makes an integer point on the maximum pixel center part of the ROI.
    required_right = min(
        sensor_width,
        math.ceil(max(xs) + margin_px + 1.0),
    )
    required_bottom = min(
        sensor_height,
        math.ceil(max(ys) + margin_px + 1.0),
    )

    width = math.ceil((required_right - left) / width_alignment) * width_alignment
    height = (
        math.ceil((required_bottom - top) / height_alignment) * height_alignment
    )
    if left + width > sensor_width:
        left = sensor_width - width
    if top + height > sensor_height:
        top = sensor_height - height

    roi = [int(left), int(top), int(width), int(height)]
    invalid_alignment = (
        roi[0] % offset_alignment
        or roi[1] % offset_alignment
        or roi[2] % width_alignment
        or roi[3] % height_alignment
    )
    if min(roi[2:]) <= 0 or invalid_alignment:
        raise RuntimeError(f"derived an invalid TUCam ROI: {roi}")
    if roi[0] + roi[2] > sensor_width or roi[1] + roi[3] > sensor_height:
        raise RuntimeError(f"derived a TUCam ROI outside the sensor: {roi}")
    return roi


def _parse_capture_timing(user: Mapping[str, Any]) -> dict[str, Any]:
    raw = user.get("capture_timing", {})
    if not isinstance(raw, Mapping):
        raise ValueError("LAB_CONFIG.yaml: capture_timing must be a mapping")
    result = {
        "formal_slm_settle_delay_ms": float(
            raw.get("formal_slm_settle_delay_ms", 200.0)
        ),
        "discard_frames_after_display": int(
            raw.get("discard_frames_after_display", 1)
        ),
        "camera_warmup_frames": int(raw.get("camera_warmup_frames", 3)),
    }
    if (
        not math.isfinite(result["formal_slm_settle_delay_ms"])
        or result["formal_slm_settle_delay_ms"] < 0.0
    ):
        raise ValueError(
            "LAB_CONFIG.yaml: formal_slm_settle_delay_ms must be finite and non-negative"
        )
    if min(
        result["discard_frames_after_display"], result["camera_warmup_frames"]
    ) < 0:
        raise ValueError("LAB_CONFIG.yaml: camera frame counts cannot be negative")
    diagnostic = user.get("timing_diagnostic", {})
    if not isinstance(diagnostic, Mapping):
        raise ValueError("LAB_CONFIG.yaml: timing_diagnostic must be a mapping")
    result["timing_diagnostic"] = copy.deepcopy(dict(diagnostic))
    return result


def _parse_lut_calibration(user: Mapping[str, Any]) -> dict[str, Any]:
    raw = user.get("amplitude_lut_calibration", {})
    if not isinstance(raw, Mapping):
        raise ValueError(
            "LAB_CONFIG.yaml: amplitude_lut_calibration must be a mapping"
        )
    value = copy.deepcopy(dict(raw))
    count = int(value.get("gray_point_count", 64))
    frames = int(value.get("frames_per_gray", 3))
    transfer = str(value.get("target_transfer", "field_amplitude"))
    filename = str(
        value.get(
            "output_lut_filename",
            "slm7930_at532_70C_linearized_field_amplitude.lut",
        )
    ).strip()
    if not 32 <= count <= 256:
        raise ValueError(
            "LAB_CONFIG.yaml: LUT gray_point_count must be in 32..256"
        )
    if frames != 3:
        raise ValueError(
            "LAB_CONFIG.yaml: audited LUT calibration requires frames_per_gray=3"
        )
    if transfer not in {"field_amplitude", "linear_intensity"}:
        raise ValueError(
            "LAB_CONFIG.yaml: target_transfer must be field_amplitude or "
            "linear_intensity"
        )
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".lut":
        raise ValueError(
            "LAB_CONFIG.yaml: output_lut_filename must be one plain .lut file name"
        )
    value.update(
        {
            "gray_point_count": count,
            "frames_per_gray": frames,
            "target_transfer": transfer,
            "output_lut_filename": filename,
        }
    )
    return value


def _runtime_base(
    template: Mapping[str, Any],
    lut_filename: str,
    exposure_us: float,
    capture_timing: Mapping[str, Any],
    lut_calibration: Mapping[str, Any],
    lut_sha256: str | None,
) -> dict[str, Any]:
    hardware = copy.deepcopy(dict(template))
    hardware["amplitude_slm"]["lut_file"] = (
        "../../hardware_sdk/vendor_sdk/amplitude_meadowlark/LUT Files/"
        + lut_filename
    )
    hardware["amplitude_slm"]["expected_lut_sha256"] = lut_sha256
    hardware["camera"]["exposure_us"] = exposure_us
    hardware["settle_delay_ms"] = capture_timing[
        "formal_slm_settle_delay_ms"
    ]
    hardware["camera"]["discard_frames_after_display"] = capture_timing[
        "discard_frames_after_display"
    ]
    hardware["camera"]["warmup_frames"] = capture_timing[
        "camera_warmup_frames"
    ]
    hardware["timing_diagnostic"].update(
        capture_timing["timing_diagnostic"]
    )
    hardware["amplitude_lut_calibration"].update(lut_calibration)
    hardware["exposure_calibration"]["exposure_times_us"] = [exposure_us]
    return hardware


def prepare_lab(
    lab_config_path: str | Path,
    *,
    template_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    require_lut_file: bool = True,
) -> dict[str, Any]:
    lab_config_file = Path(lab_config_path).expanduser().resolve()
    template_file = Path(template_path).expanduser().resolve()
    generated = Path(output_dir).expanduser().resolve()
    repository = Path(repo_root).expanduser().resolve()
    generated.mkdir(parents=True, exist_ok=True)

    user = _read_yaml(lab_config_file)
    template = _read_yaml(template_file)
    lut_filename = _validate_lut_filename(user.get("amplitude_lut_filename"))
    lut_file = repository / LUT_RELATIVE_DIRECTORY / lut_filename
    if require_lut_file and not lut_file.is_file():
        raise FileNotFoundError(
            "The selected Meadowlark LUT does not exist:\n"
            f"  {lut_file}\n"
            "Copy that exact .lut file into the directory above, then rerun the "
            "same prepare_lab command."
        )
    expected_lut_sha256 = _validate_optional_sha256(
        user.get("amplitude_lut_expected_sha256"),
        label="amplitude_lut_expected_sha256",
    )
    actual_lut_sha256 = _sha256(lut_file) if lut_file.is_file() else None
    if (
        expected_lut_sha256 is not None
        and actual_lut_sha256 != expected_lut_sha256
    ):
        raise RuntimeError(
            "Selected Meadowlark LUT hash mismatch; refusing to generate runtime "
            f"configs. file={lut_file}, actual={actual_lut_sha256}, "
            f"expected={expected_lut_sha256}"
        )
    exposure_us = float(user.get("camera_exposure_us", 0.0))
    if not math.isfinite(exposure_us) or exposure_us <= 0.0:
        raise ValueError("LAB_CONFIG.yaml: camera_exposure_us must be positive")
    capture_timing = _parse_capture_timing(user)
    lut_calibration = _parse_lut_calibration(user)
    corners = _parse_corners(user.get("logical_corners_full_sensor_xy"))

    bootstrap = _runtime_base(
        template,
        lut_filename,
        exposure_us,
        capture_timing,
        lut_calibration,
        expected_lut_sha256,
    )
    bootstrap_path = generated / "bootstrap_hardware.yaml"
    _write_yaml(bootstrap_path, bootstrap)

    result: dict[str, Any] = {
        "status": "bootstrap_ready" if corners is None else "ready",
        "edit_only_this_file": str(lab_config_file),
        "lut_file": str(lut_file),
        "lut_present": lut_file.is_file(),
        "lut_sha256": actual_lut_sha256,
        "lut_expected_sha256": expected_lut_sha256,
        "camera_exposure_us": exposure_us,
        "capture_timing": capture_timing,
        "amplitude_lut_calibration": lut_calibration,
        "bootstrap_config": str(bootstrap_path),
        "corner_coordinates_require_multiple_of_four": False,
        "hardware_roi_alignment_px": {
            "left": ROI_OFFSET_ALIGNMENT,
            "top": ROI_OFFSET_ALIGNMENT,
            "width": ROI_WIDTH_ALIGNMENT,
            "height": ROI_HEIGHT_ALIGNMENT,
        },
    }

    formal_names = (
        "formal_hardware.yaml",
        "detector_homography_478.contract.json",
        "detector_homography_478.contract.json.sha256",
        "geometry_source.generated.yaml",
    )
    if corners is None:
        for name in formal_names:
            (generated / name).unlink(missing_ok=True)
        result["instruction"] = (
            "Use bootstrap_hardware.yaml to capture P4, fill all four logical "
            "corners in LAB_CONFIG.yaml, then rerun prepare_lab."
        )
    else:
        roi = derive_device_roi(corners)
        geometry_source = {
            "device_roi_xywh_full_sensor": roi,
            "source_points_full_sensor_xy": corners,
            "target_size_wh": list(TARGET_SIZE_WH),
            "orientation": {
                "flip_vertical_after_warp": False,
                "flip_horizontal_after_warp": False,
                "downstream_loader_flip_vertical": False,
                "downstream_loader_flip_horizontal": False,
            },
            "validation_max_rms_error_px": 1.5,
            "validation_max_error_px": 3.0,
        }
        geometry_source_path = generated / "geometry_source.generated.yaml"
        _write_yaml(geometry_source_path, geometry_source)
        contract_path = generated / "detector_homography_478.contract.json"
        contract_report = write_geometry_contract(
            build_geometry_contract(geometry_source), contract_path
        )
        _, verified = load_geometry_contract(
            contract_path, expected_file_sha256=contract_report["file_sha256"]
        )

        formal = _runtime_base(
            template,
            lut_filename,
            exposure_us,
            capture_timing,
            lut_calibration,
            expected_lut_sha256,
        )
        formal_camera = formal["camera"]
        formal_camera["require_device_roi"] = True
        formal_camera["device_roi_xywh"] = roi
        formal_camera["saved_frame_size_wh"] = list(TARGET_SIZE_WH)
        formal_camera["saved_frame_resize_mode"] = "auto"
        formal_camera["detector_geometry"] = {
            "enabled": True,
            "contract_file": contract_path.name,
            "expected_file_sha256": verified["file_sha256"],
        }
        formal_path = generated / "formal_hardware.yaml"
        _write_yaml(formal_path, formal)
        # Parse the just-written files one more time as the acquisition code will.
        load_geometry_contract(
            contract_path,
            expected_file_sha256=formal_camera["detector_geometry"][
                "expected_file_sha256"
            ],
        )
        result.update(
            {
                "logical_corners_full_sensor_xy": corners,
                "derived_device_roi_xywh": roi,
                "roi_margin_px": ROI_MARGIN_PX,
                "contract": str(contract_path),
                "contract_sha256": verified["file_sha256"],
                "formal_config": str(formal_path),
                "saved_output_size_wh": list(TARGET_SIZE_WH),
                "saved_output_orientation": "canonical_model_xy",
                "next_command": (
                    "python -m experiments.hardware_sdk.workflows.roi_calibration "
                    "exposure --config "
                    "experiments/lab_qwen/generated/formal_hardware.yaml"
                ),
            }
        )

    report_path = generated / "prepare_report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result["report"] = str(report_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="experiments/lab_qwen/LAB_CONFIG.yaml"
    )
    args = parser.parse_args()
    lab_directory = Path(__file__).resolve().parent
    repository = lab_directory.parents[1]
    report = prepare_lab(
        args.config,
        template_path=lab_directory / "internal/hardware_template.yaml",
        output_dir=lab_directory / "generated",
        repo_root=repository,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
