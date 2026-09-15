from pathlib import Path

import numpy as np
from PIL import Image

from experiments.hardware_sdk.generators.fresnel_phase_array import (
    fresnel_lens_phase,
    fresnel_phase_array,
    symmetric_tile_sizes,
    tile_bounds_and_centers,
)
from experiments.hardware_sdk.generators.fresnel_roi_vertex_array import (
    angular_spectrum_focus_validation,
    exact_support_axis_targets,
    fresnel_roi_vertex_phase,
)
from experiments.hardware_sdk.generators.slm_patterns.generate import lens_phase


def test_shared_1016_support_has_exact_symmetric_array_centers() -> None:
    assert symmetric_tile_sizes(1016, 1) == [1016]
    assert symmetric_tile_sizes(1016, 2) == [508, 508]
    assert symmetric_tile_sizes(1016, 3) == [339, 338, 339]
    _, centers4 = tile_bounds_and_centers(1016, 2, active_origin_xy=(472, 82))
    assert centers4 == [
        (726.0, 336.0),
        (1234.0, 336.0),
        (726.0, 844.0),
        (1234.0, 844.0),
    ]
    _, centers9 = tile_bounds_and_centers(1016, 3, active_origin_xy=(472, 82))
    assert centers9[0] == (641.5, 251.5)
    assert centers9[4] == (980.0, 590.0)
    assert centers9[-1] == (1318.5, 928.5)


def test_fresnel_phase_matches_repository_paraxial_formula() -> None:
    phase = fresnel_lens_phase(
        64,
        64,
        pixel_pitch_um=8.0,
        wavelength_nm=532.0,
        propagation_cm=5.0,
    )
    assert phase.shape == (64, 64)
    assert phase.dtype == np.uint8
    assert int(phase.max()) > int(phase.min())
    assert np.array_equal(phase, lens_phase(64, 8.0, 532.0, 5.0))
    # The first 2pi radius is sqrt(2*lambda*z)/pitch = 28.83 pixels.
    expected_radius = np.sqrt(2 * 532e-9 * 0.05) / 8e-6
    np.testing.assert_allclose(expected_radius, 28.8314, rtol=1e-4)


def test_flip_coded_array_keeps_centers_and_reduces_only_top_left_aperture() -> None:
    uniform, uniform_lenses = fresnel_phase_array(
        120,
        2,
        pixel_pitch_um=8.0,
        wavelength_nm=532.0,
        propagation_cm=10.0,
        orientation_coded=False,
        marker_aperture_fraction=0.55,
    )
    coded, coded_lenses = fresnel_phase_array(
        120,
        2,
        pixel_pitch_um=8.0,
        wavelength_nm=532.0,
        propagation_cm=10.0,
        orientation_coded=True,
        marker_aperture_fraction=0.55,
    )
    assert [item["active_center_edge_xy"] for item in uniform_lenses] == [
        item["active_center_edge_xy"] for item in coded_lenses
    ]
    assert [item["aperture_fraction"] for item in coded_lenses] == [
        0.55,
        1.0,
        1.0,
        1.0,
    ]
    assert np.count_nonzero(coded[:60, :60]) < np.count_nonzero(uniform[:60, :60])
    assert np.array_equal(coded[:60, 60:], uniform[:60, 60:])


def test_uniform_white_amplitude_bmp_contract(tmp_path: Path) -> None:
    path = tmp_path / "amplitude_uniform_white.bmp"
    Image.fromarray(np.full((1024, 1024), 255, dtype=np.uint8), mode="L").save(
        path, format="BMP"
    )
    with Image.open(path) as image:
        assert image.format == "BMP"
        assert image.mode == "L"
        assert image.size == (1024, 1024)
        assert np.asarray(image).min() == 255


def test_corrected_array_targets_exact_physical_support_points() -> None:
    exact_width = 478 * 17 / 8
    assert exact_width == 1015.75
    assert exact_support_axis_targets(
        active_size=1016,
        exact_physical_width_px=exact_width,
        grid_size=1,
    ) == [508.0]
    assert exact_support_axis_targets(
        active_size=1016,
        exact_physical_width_px=exact_width,
        grid_size=2,
    ) == [0.125, 1015.875]
    assert exact_support_axis_targets(
        active_size=1016,
        exact_physical_width_px=exact_width,
        grid_size=3,
    ) == [0.125, 508.0, 1015.875]


def test_corrected_array_uses_inward_apertures_and_full_support() -> None:
    phase4, lenses4 = fresnel_roi_vertex_phase(
        1016,
        2,
        exact_physical_width_px=1015.75,
        pixel_pitch_um=8.0,
        wavelength_nm=532.0,
        propagation_cm=10.0,
    )
    assert phase4.shape == (1016, 1016)
    assert phase4.dtype == np.uint8
    assert [item["aperture_kind"] for item in lenses4] == [
        "inward_quarter",
        "inward_quarter",
        "inward_quarter",
        "inward_quarter",
    ]
    assert sum(item["aperture_pixel_count"] for item in lenses4) == 1016**2
    np.testing.assert_allclose(
        lenses4[0]["nyquist_radius_phase_px"], 415.625, rtol=1e-12
    )
    np.testing.assert_allclose(
        lenses4[0]["safe_circular_radius_phase_px"], 382.375, rtol=1e-12
    )
    assert all(
        item["active_lens_phase_pixel_count"] < item["aperture_pixel_count"]
        for item in lenses4
    )
    assert all(item["outside_safe_aperture_phase_uint8"] == 0 for item in lenses4)
    assert lenses4[0]["logical_aperture_bounds_local_edge_xyxy"] == [
        0,
        0,
        508,
        508,
    ]
    assert lenses4[-1]["logical_focus_target_local_edge_xy"] == [
        1015.875,
        1015.875,
    ]

    _, lenses9 = fresnel_roi_vertex_phase(
        1016,
        3,
        exact_physical_width_px=1015.75,
        pixel_pitch_um=8.0,
        wavelength_nm=532.0,
        propagation_cm=10.0,
    )
    assert sum(item["aperture_pixel_count"] for item in lenses9) == 1016**2
    assert lenses9[4]["aperture_kind"] == "full"
    assert [item["logical_focus_target_local_edge_xy"] for item in lenses9] == [
        [0.125, 0.125],
        [508.0, 0.125],
        [1015.875, 0.125],
        [0.125, 508.0],
        [508.0, 508.0],
        [1015.875, 508.0],
        [0.125, 1015.875],
        [508.0, 1015.875],
        [1015.875, 1015.875],
    ]


def test_quantized_corrected_array_numerically_focuses_at_roi_vertices() -> None:
    # Use the real 1016-pixel support and 10 cm propagation.  This checks the
    # phase values themselves, rather than trusting manifest coordinates.
    phase4, _ = fresnel_roi_vertex_phase(
        1016,
        2,
        exact_physical_width_px=1015.75,
        pixel_pitch_um=8.0,
        wavelength_nm=532.0,
        propagation_cm=10.0,
    )
    validation = angular_spectrum_focus_validation(
        phase4,
        target_axis_edge_coordinates=[0.125, 1015.875],
        pixel_pitch_um=8.0,
        wavelength_nm=532.0,
        propagation_cm=10.0,
        pad_size=2048,
    )
    assert validation["passed"] is True
    assert validation["max_abs_position_error_phase_px"] <= 0.375
    assert validation["unique_peak_assignment"] is True
    assert validation["minimum_target_peak_to_global_median"] >= 100.0
    assert validation["minimum_target_peak_to_max_outside_targets"] >= 50.0
    assert len(validation["peaks"]) == 4
