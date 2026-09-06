from pathlib import Path

import pytest

from LightGenV2.tasks.t03_saliency.settings import load_settings


TASK = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("moe_optical_router_scale_matched_dc20.yaml", "optical_router_scale_matched_moe"),
        ("d2nn_active_expert_matched_dc20.yaml", "d2nn_active_expert_matched"),
    ],
)
def test_formal_profiles(name: str, variant: str) -> None:
    settings = load_settings(TASK / "configs" / name)
    assert settings.lightgen_model_variant == variant
    assert settings.router_backend == "optical"
    assert settings.top_k == 2
    assert settings.gradient_clip_norm == 1.0
    assert settings.pixel_pitch_um == pytest.approx(17.0)
    assert settings.global_to_detector_distance_m == pytest.approx(0.10)
    assert settings.language_optical_zero_order_enabled is True
    assert settings.language_optical_amplitude_zero_order_intensity_min >= 0.20
    assert settings.validation_limit is None


def test_active_d2nn_phase_budget_matches_top2() -> None:
    settings = load_settings(
        TASK / "configs" / "d2nn_active_expert_matched_dc20.yaml"
    )
    assert settings.d2nn_phase_layers * settings.d2nn_phase_size**2 == 2 * 224**2
