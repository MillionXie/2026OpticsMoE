from __future__ import annotations

import json
from pathlib import Path

import pytest

from LightGenV2.common.optical_contract import (
    OpticalContract,
    REFERENCE_532NM_17UM_10CM,
)
from LightGenV2.tasks.t12_text_to_image.settings import load_settings


ROOT = Path(__file__).resolve().parents[2]


def test_reference_contract_and_8um_raster() -> None:
    propagation = REFERENCE_532NM_17UM_10CM
    assert propagation.propagation_distance_cm == pytest.approx(10.0)
    assert propagation.active_pixels is None

    contract = propagation.with_aperture(478, canvas_pixels=518)
    assert contract.propagation_distance_cm == pytest.approx(10.0)
    assert contract.active_width_mm == pytest.approx(8.126)
    raster = contract.device_raster(8.0)
    assert raster.pixels == 1016
    assert raster.width_mm == pytest.approx(8.128)
    assert abs(raster.relative_width_error) < 0.00025


def test_t10_study_uses_the_reference_physics() -> None:
    values = json.loads(
        (ROOT / "tasks" / "t10_expert_scaling" / "configs" / "study.json").read_text(
            encoding="utf-8"
        )
    )["geometry"]
    contract = OpticalContract(
        wavelength_nm=values["wavelength_nm"],
        logical_pixel_pitch_um=values["pixel_pitch_um"],
        propagation_distance_m=values["propagation_distance_m"],
    )
    contract.require_same_physics(REFERENCE_532NM_17UM_10CM)


def test_t12_distinguishes_physical_and_dimensionless_backends() -> None:
    config_dir = ROOT / "tasks" / "t12_text_to_image" / "configs"
    formal = load_settings(config_dir / "lightgen_parallel.yaml")
    smoke = load_settings(config_dir / "smoke.yaml")
    baseline = load_settings(config_dir / "qwen_vae_baseline.yaml")
    assert formal.optical_contract == REFERENCE_532NM_17UM_10CM
    assert smoke.optical_contract is None
    assert baseline.optical_contract is None
