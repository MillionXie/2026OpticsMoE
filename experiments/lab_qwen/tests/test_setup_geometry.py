from __future__ import annotations

import json
from pathlib import Path

import yaml

from experiments.lab_qwen.setup_geometry import setup_geometry


def test_setup_geometry_updates_hardware_without_manual_hash(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    hardware = tmp_path / "hardware.yaml"
    hardware.write_bytes((root / "config/hardware.yaml").read_bytes())
    report = setup_geometry(root / "config/geometry.yaml", hardware)

    assert report["status"] == "ready"
    assert len(report["contract_sha256"]) == 64
    assert (tmp_path / "geometry.json.sha256").is_file()
    assert json.loads((tmp_path / "geometry.json").read_text(encoding="utf-8"))[
        "orientation_canonicalized"
    ] is True
    updated = yaml.safe_load(hardware.read_text(encoding="utf-8"))
    assert updated["camera"]["device_roi_xywh"] == [0, 0, 2048, 2048]
    assert updated["camera"]["detector_geometry"] == {
        "enabled": True,
        "contract_file": "geometry.json",
        "expected_file_sha256": report["contract_sha256"],
    }
