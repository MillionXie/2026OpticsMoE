from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.hardware_sdk.generators.fresnel_full_panel_array import (
    matlab_square_fresnel_phase,
)


COMMON = {
    "size_wh": (1920, 1200),
    "center_xy": (980.0, 590.0),
    "phase_pitch_um": 8.0,
    "wavelength_nm": 532.0,
    "propagation_cm": 10.0,
    "target_span_px": 478,
    "target_pitch_um": 17.0,
    "lens_window_phase_px": {1: 403, 4: 160, 9: 160},
}


def test_matlab_n4_uses_exact_roi_vertex_spacing_and_full_square_windows() -> None:
    phase, records = matlab_square_fresnel_phase(grid=2, **COMMON)

    assert phase.dtype == np.uint8
    assert phase.shape == (1200, 1920)
    assert len(records) == 4
    targets = {
        tuple(record["logical_target_phase_edge_xy"]) for record in records
    }
    assert targets == {
        (472.125, 82.125),
        (1487.875, 82.125),
        (472.125, 1097.875),
        (1487.875, 1097.875),
    }
    assert 1487.875 - 472.125 == pytest.approx(478 * 17 / 8)
    occupied = np.zeros_like(phase, dtype=bool)
    for record in records:
        x0, y0, x1, y1 = record["square_phase_window_xyxy"]
        assert (x1 - x0, y1 - y0) == (160, 160)
        assert 0 <= x0 < x1 <= 1920
        assert 0 <= y0 < y1 <= 1200
        assert not occupied[y0:y1, x0:x1].any()
        occupied[y0:y1, x0:x1] = True
        assert record["window_clipped"] is False
    assert np.count_nonzero(phase[~occupied]) == 0
    assert np.unique(phase[occupied]).size > 200


def test_phase_samples_follow_teacher_quadratic_formula() -> None:
    phase, records = matlab_square_fresnel_phase(grid=1, **COMMON)
    record = records[0]
    x0, y0, _, _ = record["square_phase_window_xyxy"]
    target_x, target_y = record["logical_target_phase_edge_xy"]
    pitch_m = 8.0e-6
    wavelength_m = 532.0e-9
    focal_m = 0.10
    x_m = (x0 + 0.5 - target_x) * pitch_m
    y_m = (y0 + 0.5 - target_y) * pitch_m
    expected_rad = -(2.0 * math.pi / wavelength_m) * (
        x_m * x_m + y_m * y_m
    ) / (2.0 * focal_m)
    expected_u8 = int(
        math.floor((expected_rad % (2.0 * math.pi)) * 256.0 / (2.0 * math.pi))
    )
    assert int(phase[y0, x0]) == expected_u8


def test_impossible_teacher_window_is_rejected_instead_of_clipped() -> None:
    invalid = dict(COMMON)
    invalid["lens_window_phase_px"] = {1: 403, 4: 1016, 9: 160}
    with pytest.raises(ValueError, match="would be clipped"):
        matlab_square_fresnel_phase(grid=2, **invalid)

