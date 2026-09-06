from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t01_object_retrieval.modeling import (
    DensePhasePlane,
    hard_topk_load_balance_loss,
    parameter_fairness_contract,
)
from LightGenV2.tasks.t01_object_retrieval.report import _aggregate
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings
from LightGenV2.tasks.t01_object_retrieval.visualize import render
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import phase_dc_loss


TASK_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "variant"),
    (
        ("moe_optical_router_scale_matched.yaml", "optical_router_scale_matched_moe"),
        ("d2nn_active_expert_matched.yaml", "d2nn_active_expert_matched"),
        ("moe_optical_router_scale_matched_dc20.yaml", "optical_router_scale_matched_moe"),
        ("d2nn_active_expert_matched_dc20.yaml", "d2nn_active_expert_matched"),
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


@pytest.mark.parametrize(
    "filename",
    (
        "moe_optical_router_scale_matched_dc20.yaml",
        "d2nn_active_expert_matched_dc20.yaml",
    ),
)
def test_dc20_profiles_model_coherent_leakage_and_robustness(filename: str) -> None:
    settings = load_settings(TASK_DIR / "configs" / filename)
    assert settings.language_optical_zero_order_enabled is True
    assert settings.language_optical_amplitude_zero_order_intensity_min == pytest.approx(0.20)
    assert settings.language_optical_amplitude_zero_order_intensity_max == pytest.approx(0.30)
    assert settings.language_optical_phase_zero_order_intensity_min == pytest.approx(0.20)
    assert settings.language_optical_phase_zero_order_intensity_max == pytest.approx(0.30)
    assert settings.language_optical_ccd_noise_distribution == "truncated_biased_gaussian"
    assert settings.phase_dc_enabled is True
    assert settings.lambda_phase_dc == pytest.approx(0.005)
    if filename.startswith("moe_"):
        assert settings.lambda_router_hard_load_balance == pytest.approx(0.08)


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
    assert torch.isfinite(phase_dc_loss(plane))


def test_optical_router_hard_load_loss_uses_hard_value_and_soft_gradient() -> None:
    logits = torch.tensor(
        [[0.0, 0.1, 0.2, -0.1]] * 8, requires_grad=True
    )
    probabilities = torch.softmax(logits, dim=-1)
    selected = torch.zeros_like(probabilities, dtype=torch.bool)
    selected[:, 1:3] = True
    loss = hard_topk_load_balance_loss(
        {"probabilities": probabilities, "selected_mask": selected},
        num_experts=4,
        top_k=2,
    )
    assert float(loss.detach()) == pytest.approx(1.0)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_report_uses_sample_standard_deviation_for_repeated_runs() -> None:
    rows = []
    for key, values in (
        ("main", [0.80, 0.82, 0.84]),
        ("d2nn", [0.70, 0.72, 0.74]),
        ("qwen", [0.995]),
    ):
        rows.extend(
            {"key": key, "top1": value, "top3": value, "mrr": value}
            for value in values
        )
    output = {row["key"]: row for row in _aggregate(rows)}
    assert output["main"]["top1_mean"] == pytest.approx(0.82)
    assert output["main"]["top1_std"] == pytest.approx(0.02)
    assert output["qwen"]["top1_std"] == 0.0


def test_checkpoint_phase_renderer_is_direct_and_self_describing(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_checkpoint.pt"
    torch.save(
        {
            "epoch": 7,
            "metadata": {"weight_variant": "ema"},
            "vision_optical": {
                "core.optical_branch.phase1.raw_phase": torch.zeros(8, 8)
            },
            "language_optical": {
                "core.optical_branch.phase1.raw_phase": torch.ones(8, 8)
            },
        },
        checkpoint,
    )
    output = tmp_path / "visualization"
    report = render(checkpoint, output)
    assert report["checkpoint_epoch"] == 7
    assert report["plane_count"] == 2
    assert (output / "best_phase_overview.png").is_file()
    assert (output / "best_phase_overview.pdf").is_file()
    assert (output / "phase_statistics.json").is_file()
