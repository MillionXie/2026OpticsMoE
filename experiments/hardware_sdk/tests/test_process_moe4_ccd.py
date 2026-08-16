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
