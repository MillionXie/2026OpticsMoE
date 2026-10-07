from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "temporal_16x4_s163.yaml").read_text(encoding="utf-8")
    )


def _public_contract():
    sys.path.insert(0, str(ROOT / "runtime"))
    from lgvq_temporal.contracts import load_reference_contract

    return load_reference_contract(ROOT / "configs" / "temporal_16x4_s163.yaml")


def test_frozen_08044_optical_contract_is_10_cm() -> None:
    config = _config()
    optics = config["optics"]
    geometry = config["geometry"]

    assert optics["wavelength_nm"] == pytest.approx(532.0)
    assert optics["pixel_pitch_um"] == pytest.approx(17.0)
    assert optics["modulator_pixel_pitch_um"] == pytest.approx(8.0)
    assert optics["distance_m"] == pytest.approx(0.10)
    assert optics["theta_max_deg"] == pytest.approx(0.5)
    assert optics["phase_quantization_levels"] == 0
    assert geometry["canvas_size"] == 518
    assert geometry["active_size"] == 478


def test_config_loads_with_the_multivideo_settings_contract() -> None:
    sys.path.insert(0, str(ROOT / "runtime"))
    from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import (
        load_settings,
    )

    settings = load_settings(ROOT / "configs" / "temporal_16x4_s163.yaml")
    assert settings.architecture_label == (
        "lightgenv2_t06_temporal_multivideo16x4_visualrouter_o6_top2_no_attention_v3"
    )
    assert settings.distance_m == pytest.approx(0.10)
    assert settings.propagation_canvas_size == 1101
    assert settings.propagation_active_size == 1016
    assert settings.phase_quantization_levels == 0
    assert settings.ccd_noise_enabled is False
    assert settings.ccd_noise_distribution == "truncated_biased_gaussian"


def test_17um_model_to_8um_device_preserves_physical_aperture() -> None:
    contract = _public_contract()
    raster = contract.device_raster(8.0)
    assert contract.active_width_mm == pytest.approx(8.126)
    assert raster.pixels == 1016
    assert raster.width_mm == pytest.approx(8.128)
    assert abs(raster.relative_width_error) < 0.00025


def test_public_entrypoint_uses_clean_runtime_facade() -> None:
    source = (ROOT / "simulate.py").read_text(encoding="utf-8")
    assert "from lgvq_temporal.fixed_weight import" in source
    assert "from LightGenV2.tasks" not in source
    assert "from experiments." not in source


def test_reference_eval_keeps_optional_ccd_noise_disabled() -> None:
    config = _config()
    robustness = config["robustness"]
    optics = config["optics"]

    assert robustness["input_shift_pixels"] == 4
    assert robustness["phase_shift_pixels"] == 4
    assert robustness["ccd_shift_pixels"] == 4
    assert robustness["phase_dropout_p"] == pytest.approx(0.05)
    assert robustness["phase_dropout_cell_size"] == 4
    ccd_noise = robustness["ccd_noise"]
    assert ccd_noise["enabled"] is False
    assert ccd_noise["biased_gaussian"]["distribution"] == (
        "truncated_biased_gaussian"
    )
    assert ccd_noise["biased_gaussian"]["mean_fraction"] == pytest.approx(0.03)
    assert ccd_noise["shot_noise"]["photons_per_mean"] == pytest.approx(4096.0)
    assert optics["unmodulated_power_fraction_eval"] == pytest.approx(0.20)
    # The optional CCD perturbation is disabled for the reported 0.8044 result.
    # Other calibrated sensor and device effects remain outside this profile.
    assert optics["phase_quantization_levels"] == 0
    assert "sensor" not in config
