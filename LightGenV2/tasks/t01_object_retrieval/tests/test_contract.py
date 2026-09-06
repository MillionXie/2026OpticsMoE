from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t01_object_retrieval.modeling import (
    DensePhasePlane,
    parameter_fairness_contract,
)
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings


TASK_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "variant"),
    (
        ("moe_optical_router_scale_matched.yaml", "optical_router_scale_matched_moe"),
        ("d2nn_active_expert_matched.yaml", "d2nn_active_expert_matched"),
        ("qwen_frozen_embedding.yaml", "frozen_qwen_embedding"),
    ),
)
def test_formal_profiles_share_the_audited_contract(
    filename: str, variant: str
) -> None:
    settings = load_settings(TASK_DIR / "configs" / filename)
    assert settings.lightgen_model_variant == variant
    assert settings.router_backend == "optical"
    assert settings.top_k == 2
    assert settings.fusion_mode == "scale_matched_convex"
    assert settings.evaluate_test_each_epoch is True
    assert settings.test_evaluation_interval_epochs == 5
    assert settings.optimizer_steps_per_epoch is None
    assert settings.language_optical_distance_m == pytest.approx(0.10)
    assert settings.language_optical_pixel_pitch_um == pytest.approx(17.0)


def test_d2nn_exactly_matches_top2_activated_expert_phase_budget() -> None:
    settings = load_settings(
        TASK_DIR / "configs" / "d2nn_active_expert_matched.yaml"
    )
    report = parameter_fairness_contract(settings)
    assert report["parameters_per_expert"] == 224 * 224
    assert report["moe_active_expert_parameters_per_modality"] == 2 * 224 * 224
    assert report["d2nn_phase_parameters_per_modality"] == 2 * 224 * 224
    assert report["active_expert_parameter_match_exact"] is True


def test_dense_phase_is_2pi_sigmoid_and_receives_gradient() -> None:
    settings = SimpleNamespace(
        language_optical_phase_init_std=0.1,
        language_optical_phase_dropout_p=0.0,
        language_optical_phase_dropout_block_size=8,
    )
    plane = DensePhasePlane(16, settings)
    phase = plane.physical_phase()
    assert torch.all(phase > 0.0)
    assert torch.all(phase < 2.0 * torch.pi)
    loss = torch.cos(phase).mean()
    loss.backward()
    assert plane.raw_phase.grad is not None
    assert torch.isfinite(plane.raw_phase.grad).all()
