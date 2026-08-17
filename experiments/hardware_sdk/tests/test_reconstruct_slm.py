import csv
import json

import numpy as np
from PIL import Image

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_amplitude_with_metadata,
    reconstruct_directory,
    save_active_png,
)


def test_amplitude_encoding_records_reversible_scale_metadata() -> None:
    value = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    encoded, metadata = encode_active_amplitude_with_metadata(value, percentile=100.0)
    reconstructed = encoded.astype(np.float32) * metadata["scale"] / 255.0
    assert metadata["scale"] == 4.0
    assert np.allclose(reconstructed, value, atol=4.0 / 255.0)


def test_compact_payload_reconstructs_exact_centered_slm(tmp_path) -> None:
    source = tmp_path / "compact"
    output = tmp_path / "full"
    value = np.arange(12, dtype=np.uint8).reshape(3, 4)
    save_active_png(value, source / "sample.png")
    reconstruct_directory(
        source, output, slm_size_wh=(12, 10), scale_factor=2
    )
    with Image.open(output / "sample.bmp") as image:
        actual = np.asarray(image)
    expected = np.zeros((10, 12), dtype=np.uint8)
    expected[2:8, 2:10] = np.repeat(np.repeat(value, 2, axis=0), 2, axis=1)
    assert np.array_equal(actual, expected)
    with (output / "reconstruction_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["basename"] == "sample"
    assert row["active_bounds_xyxy"] == "2,2,10,8"


def test_compact_payload_uses_configured_slm_center(tmp_path) -> None:
    source = tmp_path / "compact"
    output = tmp_path / "shifted"
    value = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    save_active_png(value, source / "phase.png")
    report = reconstruct_directory(
        source,
        output,
        slm_size_wh=(12, 10),
        scale_factor=2,
        center_xy=(8, 3),
    )
    with Image.open(output / "phase.bmp") as image:
        actual = np.asarray(image)
    expected = np.zeros((10, 12), dtype=np.uint8)
    expected[1:5, 6:10] = np.repeat(
        np.repeat(value, 2, axis=0), 2, axis=1
    )
    assert np.array_equal(actual, expected)
    assert report["requested_center_xy"] == [8, 3]
    assert report["active_center_xy"] == [8.0, 3.0]
    assert report["center_offset_xy"] == [2.0, -2.0]
    persisted = json.loads(
        (output / "reconstruction_report.json").read_text(encoding="utf-8")
    )
    assert persisted["coordinate_convention"].startswith("x increases right")
    with (output / "reconstruction_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["active_bounds_xyxy"] == "6,1,10,5"
    assert row["active_center_xy"] == "8,3"
    assert row["canvas_center_offset_xy"] == "2,-2"
