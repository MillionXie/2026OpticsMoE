"""Create the CCD homography contract and activate it in one command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.hardware_sdk.workflows.detector_homography import (
    build_geometry_contract,
    load_geometry_contract,
    write_geometry_contract,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def setup_geometry(
    geometry_path: str | Path,
    hardware_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    geometry_file = Path(geometry_path).expanduser().resolve()
    hardware_file = Path(hardware_path).expanduser().resolve()
    contract_file = (
        hardware_file.parent / "geometry.json"
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    geometry = _read_yaml(geometry_file)
    report = write_geometry_contract(
        build_geometry_contract(geometry), contract_file
    )
    _, verified = load_geometry_contract(
        contract_file, expected_file_sha256=report["file_sha256"]
    )

    hardware = _read_yaml(hardware_file)
    camera = hardware.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("hardware.yaml has no camera mapping")
    roi = [int(value) for value in geometry["device_roi_xywh_full_sensor"]]
    if len(roi) != 4 or any(value % 4 for value in roi):
        raise ValueError(
            "device_roi_xywh_full_sensor must contain four values divisible by 4"
        )
    camera["require_device_roi"] = True
    camera["device_roi_xywh"] = roi
    camera["saved_frame_size_wh"] = [478, 478]
    camera["saved_frame_resize_mode"] = "auto"
    camera["detector_geometry"] = {
        "enabled": True,
        # Paths inside hardware.yaml resolve relative to its own directory.
        "contract_file": contract_file.relative_to(hardware_file.parent).as_posix(),
        "expected_file_sha256": report["file_sha256"],
    }
    exposure = hardware.get("exposure_calibration")
    if isinstance(exposure, dict):
        exposure["exposure_times_us"] = [float(camera["exposure_us"])]
    hardware_file.write_text(
        yaml.safe_dump(hardware, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    result = {
        "status": "ready",
        "geometry_source": str(geometry_file),
        "contract": str(contract_file),
        "contract_sha256": verified["file_sha256"],
        "hardware_config_updated": str(hardware_file),
        "device_roi_xywh": roi,
        "output_size_wh": [478, 478],
        "orientation": "canonical_model_xy; no downstream flip",
        "next_command": (
            "python -m experiments.hardware_sdk.workflows.roi_calibration "
            "exposure --config experiments/lab_qwen/config/hardware.yaml"
        ),
    }
    summary_path = hardware_file.parent / "geometry_setup_report.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        default="experiments/lab_qwen/config/geometry.yaml",
    )
    parser.add_argument(
        "--hardware",
        default="experiments/lab_qwen/config/hardware.yaml",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = setup_geometry(args.geometry, args.hardware, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
