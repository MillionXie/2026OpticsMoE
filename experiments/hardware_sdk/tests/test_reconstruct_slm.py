import csv

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
