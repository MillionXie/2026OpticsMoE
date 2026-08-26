from pathlib import Path

import pytest

from ..settings import load_settings


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "release"
    / "mnist4_single_layer_17um_10cm_v2_robust_raw.yaml"
)
CORRECTED_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "release"
    / "mnist4_single_layer_17um_10cm_v2_notebook_mse_light_robust.yaml"
)


def test_notebook_detector_geometry_maps_edge_by_edge_to_478() -> None:
    settings = load_settings(CONFIG)
    assert settings.detector_reference_grid_size == 400
    assert settings.detector_reference_intervals == ((75, 125), (275, 325))
    assert settings.detector_bounds() == (
        (90, 90, 149, 149),
        (329, 90, 388, 149),
        (90, 329, 149, 388),
        (329, 329, 388, 388),
    )
    for left, top, right, bottom in settings.detector_bounds():
        assert (right - left, bottom - top) == (59, 59)
    centers = {
        ((left + right) / 2.0, (top + bottom) / 2.0)
        for left, top, right, bottom in settings.detector_bounds()
    }
    assert centers == {
        (119.5, 119.5),
        (358.5, 119.5),
        (119.5, 358.5),
        (358.5, 358.5),
    }


def test_formal_contract_is_10cm_raw_ccd_kspace_and_robust() -> None:
    settings = load_settings(CONFIG)
    assert settings.detector_distance_m == pytest.approx(0.10)
    assert settings.wavelength_nm == pytest.approx(532.0)
    assert settings.logical_pixel_pitch_um == pytest.approx(17.0)
    assert settings.input_content_size == 336
    assert settings.input_size == 400
    assert settings.ccd_target_size == 478
    assert settings.k_space_enabled is True
    assert settings.k_space_theta_max_deg == pytest.approx(0.65)
    assert settings.robustness_enabled is True
    assert settings.input_shift_max_px == 2
    assert settings.phase_shift_max_px == 2
    assert settings.pre_ccd_shift_max_px == 2
    assert settings.ccd_postprocess == "none_raw_linear"
    assert settings.phase_parameterization == "sigmoid"
    assert settings.phase_init == "zeros"


def test_incorrect_detector_size_is_rejected(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8").replace("size: 59", "size: 49", 1)
    config = tmp_path / "bad.yaml"
    config.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="proportionally mapped"):
        load_settings(config)


def test_corrected_contract_uses_notebook_loss_and_light_delayed_jitter() -> None:
    settings = load_settings(CORRECTED_CONFIG)
    assert settings.loss_mode == "notebook_full_plane_mse"
    assert settings.notebook_full_plane_mse_scale == pytest.approx(100.0)
    assert settings.k_space_theta_max_deg == pytest.approx(0.80)
    assert settings.robustness_probability == pytest.approx(0.50)
    assert settings.robustness_warmup_epochs == 8
    assert settings.input_shift_max_px == 1
    assert settings.phase_shift_max_px == 1
    assert settings.pre_ccd_shift_max_px == 1
    assert settings.phase_learning_rate == pytest.approx(0.01)
    assert settings.min_learning_rate == pytest.approx(0.01)
