from __future__ import annotations

import json
from pathlib import Path
import hashlib

import yaml

from experiments.lab_qwen.prepare_lab import derive_device_roi, prepare_lab


POINTS = {
    "top_left": [1626, 281],
    "top_right": [358, 285],
    "bottom_right": [363, 1547],
    "bottom_left": [1631, 1545],
}


def _lab_config(path: Path, corners: dict | None = POINTS) -> Path:
    lut_digest = hashlib.sha256(b"test LUT").hexdigest()
    value = {
        "amplitude_lut_filename": "selected-70c.lut",
        "amplitude_lut_expected_sha256": lut_digest,
        "camera_exposure_us": 4321.0,
        "capture_timing": {
            "formal_slm_settle_delay_ms": 175,
            "discard_frames_after_display": 2,
            "camera_warmup_frames": 4,
        },
        "timing_diagnostic": {
            "settle_delay_ms_values": [0, 75, 175, 350],
            "discard_frames_after_display": 2,
        },
        "amplitude_lut_calibration": {
            "gray_point_count": 96,
            "frames_per_gray": 3,
            "target_transfer": "field_amplitude",
            "output_lut_filename": "selected-linearized.lut",
        },
        "logical_corners_full_sensor_xy": (
            {label: None for label in POINTS} if corners is None else corners
        ),
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _fake_repo(tmp_path: Path) -> Path:
    lut = (
        tmp_path
        / "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark/LUT Files"
        / "selected-70c.lut"
    )
    lut.parent.mkdir(parents=True)
    lut.write_bytes(b"test LUT")
    return tmp_path


def test_arbitrary_points_produce_aligned_padded_roi() -> None:
    assert derive_device_roi(POINTS) == [292, 216, 1408, 1396]


def test_prepare_lab_generates_pinned_formal_runtime(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    generated = tmp_path / "generated"
    report = prepare_lab(
        _lab_config(tmp_path / "LAB_CONFIG.yaml"),
        template_path=root / "internal/hardware_template.yaml",
        output_dir=generated,
        repo_root=_fake_repo(tmp_path),
    )

    assert report["status"] == "ready"
    assert report["derived_device_roi_xywh"] == [292, 216, 1408, 1396]
    assert len(report["contract_sha256"]) == 64
    assert (generated / "detector_homography_478.contract.json.sha256").is_file()
    contract = json.loads(
        (generated / "detector_homography_478.contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["orientation_canonicalized"] is True
    formal = yaml.safe_load(
        (generated / "formal_hardware.yaml").read_text(encoding="utf-8")
    )
    assert formal["camera"]["device_roi_xywh"] == [292, 216, 1408, 1396]
    assert formal["camera"]["exposure_us"] == 4321.0
    assert formal["settle_delay_ms"] == 175.0
    assert formal["camera"]["discard_frames_after_display"] == 2
    assert formal["camera"]["warmup_frames"] == 4
    assert formal["timing_diagnostic"]["settle_delay_ms_values"] == [
        0,
        75,
        175,
        350,
    ]
    assert formal["amplitude_lut_calibration"]["gray_point_count"] == 96
    assert formal["amplitude_lut_calibration"]["target_transfer"] == (
        "field_amplitude"
    )
    assert formal["amplitude_lut_calibration"]["output_lut_filename"] == (
        "selected-linearized.lut"
    )
    assert formal["camera"]["detector_geometry"] == {
        "enabled": True,
        "contract_file": "detector_homography_478.contract.json",
        "expected_file_sha256": report["contract_sha256"],
    }
    assert formal["amplitude_slm"]["lut_file"].endswith("selected-70c.lut")
    assert formal["amplitude_slm"]["expected_lut_sha256"] == hashlib.sha256(
        b"test LUT"
    ).hexdigest()
    assert report["lut_sha256"] == formal["amplitude_slm"]["expected_lut_sha256"]


def test_prepare_lab_rejects_changed_pinned_lut(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = _lab_config(tmp_path / "LAB_CONFIG.yaml")
    repository = _fake_repo(tmp_path)
    lut = (
        repository
        / "experiments/hardware_sdk/vendor_sdk/amplitude_meadowlark/LUT Files"
        / "selected-70c.lut"
    )
    lut.write_bytes(b"changed LUT")

    import pytest

    with pytest.raises(RuntimeError, match="LUT hash mismatch"):
        prepare_lab(
            config,
            template_path=root / "internal/hardware_template.yaml",
            output_dir=tmp_path / "generated",
            repo_root=repository,
        )


def test_prepare_lab_allows_bootstrap_before_points_are_known(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    generated = tmp_path / "generated"
    report = prepare_lab(
        _lab_config(tmp_path / "LAB_CONFIG.yaml", corners=None),
        template_path=root / "internal/hardware_template.yaml",
        output_dir=generated,
        repo_root=_fake_repo(tmp_path),
    )

    assert report["status"] == "bootstrap_ready"
    assert (generated / "bootstrap_hardware.yaml").is_file()
    assert not (generated / "formal_hardware.yaml").exists()
