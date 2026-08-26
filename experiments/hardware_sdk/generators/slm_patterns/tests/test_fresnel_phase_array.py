from pathlib import Path

import numpy as np
from PIL import Image

from experiments.hardware_sdk.generators.fresnel_phase_array import (
    fresnel_lens_phase,
    fresnel_phase_array,
    symmetric_tile_sizes,
    tile_bounds_and_centers,
)
from experiments.hardware_sdk.generators.slm_patterns.generate import lens_phase


def test_shared_1016_support_has_exact_symmetric_array_centers() -> None:
    assert symmetric_tile_sizes(1016, 1) == [1016]
    assert symmetric_tile_sizes(1016, 2) == [508, 508]
    assert symmetric_tile_sizes(1016, 3) == [339, 338, 339]
    _, centers4 = tile_bounds_and_centers(
        1016, 2, active_origin_xy=(472, 82)
    )
    assert centers4 == [
        (726.0, 336.0),
        (1234.0, 336.0),
        (726.0, 844.0),
        (1234.0, 844.0),
    ]
    _, centers9 = tile_bounds_and_centers(
        1016, 3, active_origin_xy=(472, 82)
    )
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
