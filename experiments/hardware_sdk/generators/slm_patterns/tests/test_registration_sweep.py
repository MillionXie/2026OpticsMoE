from pathlib import Path

import numpy as np
from PIL import Image

from experiments.hardware_sdk.generators.dual_slm_registration_sweep import (
    _save_inverted_amplitude,
    _save_phase,
    dense_tetromino_mask,
    phase_scale_values,
)


def test_scale_sweep_is_center_out_and_bounded_by_half_percent() -> None:
    values = phase_scale_values(0.005, 0.0005)
    assert len(values) == 21
    assert values[:5] == [1.0, 1.0005, 0.9995, 1.001, 0.999]
    assert values[-2:] == [1.005, 0.995]
    assert max(abs(value - 1.0) for value in values) <= 0.005 + 1.0e-12


def test_dense_tetromino_mask_contains_25_four_cell_landmarks() -> None:
    mask, count = dense_tetromino_mask(478, 24)
    assert count == 25
    assert mask.shape == (478, 478)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}
    assert int(np.count_nonzero(mask == 255)) == 25 * 4 * 24 * 24


def test_amplitude_export_inverts_the_entire_slm_canvas(tmp_path: Path) -> None:
    intended = np.zeros((4, 4), dtype=np.uint8)
    intended[:2, :2] = 255
    output = tmp_path / "amplitude.bmp"
    _save_inverted_amplitude(
        intended,
        output,
        slm_size_wh=(8, 6),
        center_xy=(4.0, 3.0),
    )
    with Image.open(output) as image:
        actual = np.asarray(image)
        assert image.format == "BMP"
        assert image.mode == "L"
    expected_intended = np.zeros((6, 8), dtype=np.uint8)
    expected_intended[1:5, 2:6] = intended
    assert np.array_equal(actual, 255 - expected_intended)


def test_phase_export_is_native_8bit_binary_bmp(tmp_path: Path) -> None:
    phase = np.zeros((4, 6), dtype=np.uint8)
    phase[:, 3:] = 128
    output = tmp_path / "phase.bmp"
    report = _save_phase(
        phase,
        output,
        slm_size_wh=(12, 10),
        center_xy=(6.0, 5.0),
    )
    with Image.open(output) as image:
        assert image.format == "BMP"
        assert image.mode == "L"
        assert image.size == (12, 10)
        assert set(np.unique(np.asarray(image))) == {0, 128}
    assert report["native_active_size_wh"] == [6, 4]
