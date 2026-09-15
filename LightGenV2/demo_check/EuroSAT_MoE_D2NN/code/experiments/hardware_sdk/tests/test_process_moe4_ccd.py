import json

import numpy as np
import yaml
from PIL import Image

from experiments.hardware_sdk.workflows.process_moe4_ccd import process


def test_processor_resizes_entire_roi_to_uint8_without_flip(tmp_path) -> None:
    source_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    source_dir.mkdir()
    source = np.arange(60, dtype=np.uint16).reshape(6, 10) * 100
    Image.fromarray(source).save(source_dir / "frame.tif")
    config = {
        "input_dir": str(source_dir),
        "output_dir": str(output_dir),
        "roi_xywh": [2, 1, 6, 4],
        "target_size_wh": [956, 956],
        "intensity": {"mode": "fixed_range", "black_level": 0, "white_level": 65535},
        "flip_vertical": False,
        "flip_horizontal": False,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    process(path, None, None)
    with Image.open(output_dir / "frame.png") as image:
        array = np.asarray(image)
        assert image.mode == "L"
        assert image.size == (956, 956)
        assert array.dtype == np.uint8
        assert float(array[:100].mean()) < float(array[-100:].mean())
    report = json.loads((output_dir / "processing_report.json").read_text())
    assert report["flip_applied"] is False
    assert report["resize_rule"] == "resize the entire selected ROI; no center crop"


def test_cli_relative_paths_are_resolved_from_working_directory(
    tmp_path, monkeypatch
) -> None:
    source_dir = tmp_path / "raw_override"
    source_dir.mkdir()
    Image.fromarray(np.arange(16, dtype=np.uint16).reshape(4, 4)).save(
        source_dir / "frame.png"
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = {
        "input_dir": "unused_raw",
        "output_dir": "unused_output",
        "roi_xywh": None,
        "target_size_wh": [956, 956],
        "intensity": {
            "mode": "fixed_range",
            "black_level": 0,
            "white_level": 65535,
        },
        "flip_vertical": False,
        "flip_horizontal": False,
    }
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    process(config_path, "raw_override", "processed_override")
    assert (tmp_path / "processed_override" / "frame.png").is_file()
    assert not (config_dir / "processed_override").exists()
