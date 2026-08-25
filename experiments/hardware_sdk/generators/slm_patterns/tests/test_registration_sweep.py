from pathlib import Path

import numpy as np
from PIL import Image

from experiments.hardware_sdk.generators.dual_slm_registration_sweep import (
    _save_inverted_amplitude,
    _save_phase,
    large_block_mask,
    phase_scale_values,
    single_axis_masked_grating,
)


def test_scale_sweep_is_center_out_and_bounded_by_half_percent() -> None:
    values = phase_scale_values(0.005, 0.0005)
    assert len(values) == 21
    assert values[:5] == [1.0, 1.0005, 0.9995, 1.001, 0.999]
    assert values[-2:] == [1.005, 0.995]
    assert max(abs(value - 1.0) for value in values) <= 0.005 + 1.0e-12


def test_multiresolution_scale_sweep_extends_to_plus_minus_point_one() -> None:
    deltas = [0.0005 * index for index in range(1, 11)] + [
        0.01 * index for index in range(1, 11)
    ]
    values = phase_scale_values(absolute_deltas=deltas)
    assert len(values) == 41
    assert values[:5] == [1.0, 1.0005, 0.9995, 1.001, 0.999]
    assert values[-2:] == [1.1, 0.9]


def test_large_blocks_are_plain_regions_made_from_four_to_nine_cells() -> None:
    mask, cell_counts = large_block_mask(478, 48)
    assert cell_counts == [4, 6, 5, 9, 6, 4]
    assert mask.shape == (478, 478)
    assert set(np.unique(mask)) == {0, 255}
    assert int(np.count_nonzero(mask == 255)) == sum(cell_counts) * 48 * 48


def test_masked_grating_uses_only_one_axis_per_phase_image() -> None:
    mask, _ = large_block_mask(478, 48)
    x_phase = single_axis_masked_grating(mask, 8, "x")
    y_phase = single_axis_masked_grating(mask, 8, "y")
    assert set(np.unique(x_phase)) == {0, 128}
    assert set(np.unique(y_phase)) == {0, 128}
    # X grating varies only with column inside an open 2x2 block; Y only row.
    open_y, open_x = np.argwhere(mask == 255)[0]
    assert np.array_equal(
        x_phase[open_y, open_x : open_x + 48],
        x_phase[open_y + 1, open_x : open_x + 48],
    )
    assert np.array_equal(
        y_phase[open_y : open_y + 48, open_x],
        y_phase[open_y : open_y + 48, open_x + 1],
    )


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
