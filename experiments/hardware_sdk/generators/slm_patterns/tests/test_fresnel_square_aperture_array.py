from __future__ import annotations

import numpy as np

from experiments.hardware_sdk.generators.fresnel_square_aperture_array import (
    build_matched_fresnel_pair,
    reflect_vertical_about_edge_center,
    roi_axis_targets_um,
    simulate_focus_array,
)


COMMON = {
    "wavelength_nm": 532.0,
    "propagation_cm": 10.0,
    "amplitude_size_wh": (1024, 1024),
    "amplitude_pitch_um": 17.0,
    "amplitude_center_xy": (512.0, 512.0),
    "amplitude_active_size_px": 478,
    "phase_size_wh": (1920, 1200),
    "phase_pitch_um": 8.0,
    "phase_center_xy": (980.0, 590.0),
}


def test_roi_targets_are_exact_physical_vertices_and_midpoints() -> None:
    assert roi_axis_targets_um(
        amplitude_active_size_px=478,
        amplitude_pixel_pitch_um=17.0,
        grid_size=1,
    ) == [0.0]
    assert roi_axis_targets_um(
        amplitude_active_size_px=478,
        amplitude_pixel_pitch_um=17.0,
        grid_size=2,
    ) == [-4063.0, 4063.0]
    assert roi_axis_targets_um(
        amplitude_active_size_px=478,
        amplitude_pixel_pitch_um=17.0,
        grid_size=3,
    ) == [-4063.0, 0.0, 4063.0]


def test_n4_uses_four_complete_independent_matched_square_pupils() -> None:
    amplitude, phase, illumination, lenslets = build_matched_fresnel_pair(
        grid_size=2,
        aperture_width_phase_px=128,
        **COMMON,
    )
    assert amplitude.shape == (1024, 1024)
    assert phase.shape == (1200, 1920)
    assert illumination.shape == phase.shape
    assert set(np.unique(amplitude)) == {0, 255}
    assert len(lenslets) == 4
    assert all(item["aperture_kind"] == "full_independent_square" for item in lenslets)
    assert all(item["full_aperture_not_clipped"] for item in lenslets)
    assert all(item["amplitude_aperture_size_wh_px"] == [60, 60] for item in lenslets)
    assert int(np.count_nonzero(amplitude)) == 4 * 60 * 60

    expected_targets = {
        (472.125, 82.125),
        (1487.875, 82.125),
        (472.125, 1097.875),
        (1487.875, 1097.875),
    }
    observed_targets = {
        tuple(item["target_phase_edge_xy_before_export_flip"])
        for item in lenslets
    }
    assert observed_targets == expected_targets
    for item in lenslets:
        left, top, right, bottom = item[
            "phase_mapped_aperture_bounds_edge_xyxy_before_flip"
        ]
        target_x, target_y = item["target_phase_edge_xy_before_export_flip"]
        assert left < target_x < right
        assert top < target_y < bottom


def test_n9_pupils_do_not_overlap_and_amplitude_quantization_is_recorded() -> None:
    amplitude, _, illumination, lenslets = build_matched_fresnel_pair(
        grid_size=3,
        aperture_width_phase_px=48,
        **COMMON,
    )
    assert len(lenslets) == 9
    assert all(item["amplitude_aperture_size_wh_px"] == [22, 22] for item in lenslets)
    assert all(
        item["amplitude_aperture_actual_size_wh_um"] == [374.0, 374.0]
        for item in lenslets
    )
    assert int(np.count_nonzero(amplitude)) == 9 * 22 * 22
    assert int(np.count_nonzero(illumination)) > 0
    assert int(np.count_nonzero(illumination)) < illumination.size


def test_small_square_pupil_produces_a_wide_clean_sinc_cross() -> None:
    _, phase, illumination, lenslets = build_matched_fresnel_pair(
        grid_size=1,
        aperture_width_phase_px=48,
        **COMMON,
    )
    intensity, metrics = simulate_focus_array(
        phase,
        illumination,
        targets_axis_um=[0.0],
        phase_center_xy=(980.0, 590.0),
        phase_pitch_um=8.0,
        wavelength_nm=532.0,
        propagation_cm=10.0,
        pad_size=2048,
        actual_aperture_width_um=lenslets[0][
            "amplitude_aperture_actual_size_wh_um"
        ][0],
    )
    assert intensity.shape == (2048, 2048)
    assert metrics["passed"] is True
    assert metrics["max_abs_position_error_phase_px"] <= 0.5
    assert metrics["minimum_target_peak_to_max_background"] > 100
    peak = metrics["peaks"][0]
    assert peak["measured_fwhm_xy_phase_px"] == [16, 16]
    assert peak["cross_axis_to_diagonal_mean_energy_ratio"] > 20
    assert peak["axis_extent_at_minus30db_xy_phase_px"][0] >= 40
    assert peak["axis_extent_at_minus30db_xy_phase_px"][1] >= 40


def test_phase_export_flip_keeps_x_and_inverts_y_edge_coordinate() -> None:
    _, phase, _, lenslets = build_matched_fresnel_pair(
        grid_size=2,
        aperture_width_phase_px=64,
        **COMMON,
    )
    exported = reflect_vertical_about_edge_center(phase, 590.0)
    assert exported.shape == (1200, 1920)
    for item in lenslets:
        before_x, before_y = item["target_phase_edge_xy_before_export_flip"]
        exported_x, exported_y = item["target_phase_edge_xy_in_exported_bmp"]
        assert exported_x == before_x
        assert exported_y == 2 * 590.0 - before_y

    # The calibrated raster centre stays at y=590 instead of drifting to 610,
    # which a whole-panel np.flipud would produce on a 1200-row panel.
    marker = np.zeros((1200, 3), dtype=np.uint8)
    marker[589, 1] = 255  # pixel centre y=589.5
    reflected = reflect_vertical_about_edge_center(marker, 590.0)
    assert reflected[590, 1] == 255  # pixel centre y=590.5
    assert np.count_nonzero(reflected) == 1


def test_phase_centered_flip_rejects_nonzero_content_that_would_be_clipped() -> None:
    value = np.zeros((1200, 2), dtype=np.uint8)
    value[1199, 0] = 1
    with np.testing.assert_raises_regex(ValueError, "would be clipped"):
        reflect_vertical_about_edge_center(value, 590.0)
