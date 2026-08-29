from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import (
    _sample_truncated_normal_like,
)

from ..modeling import SOURCE_ARCHITECTURE
from ..settings import load_settings


PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "mean", "std", "phase_lr", "dc_weight"),
    [
        ("noise_mild.yaml", 0.01, 0.01, 0.03, 0.01),
        ("noise_medium.yaml", 0.03, 0.025, 0.03, 0.01),
        ("noise_strong.yaml", 0.06, 0.05, 0.03, 0.015),
        ("noise_medium_extreme_phase.yaml", 0.03, 0.025, 0.08, 0.03),
    ],
)
def test_release_matrix_is_controlled(
    name: str, mean: float, std: float, phase_lr: float, dc_weight: float
) -> None:
    settings = load_settings(PROJECT / "configs" / "release" / name)
    assert settings.optical_fusion_minimum == pytest.approx(0.01)
    assert settings.optical_fusion_initial == pytest.approx(0.015)
    assert settings.language_optical_ccd_noise_distribution == (
        "truncated_biased_gaussian"
    )
    assert settings.language_optical_ccd_noise_mean_fraction == pytest.approx(mean)
    assert settings.language_optical_ccd_noise_std_fraction == pytest.approx(std)
    assert settings.phase_learning_rate == pytest.approx(phase_lr)
    assert settings.lambda_phase_dc == pytest.approx(dc_weight)
    assert settings.epochs == 8
    assert settings.optimizer_steps_per_epoch == 12
    assert not settings.evaluate_test_each_epoch
    assert not settings.resume_optimizer_state
    assert len(settings.continuation_sha256) == 64
    assert SOURCE_ARCHITECTURE.endswith("warmstart5_stage_b_v1")


def test_truncated_gaussian_has_bounds_bias_and_no_boundary_atoms() -> None:
    torch.manual_seed(7)
    sample = _sample_truncated_normal_like(
        torch.empty(200_000),
        mean=0.03,
        std=0.025,
        minimum=-0.02,
        maximum=0.08,
    )
    assert float(sample.min()) >= -0.02
    assert float(sample.max()) <= 0.08
    assert float(sample.mean()) == pytest.approx(0.03, abs=5.0e-4)
    assert not bool((sample == -0.02).any())
    assert not bool((sample == 0.08).any())


def test_truncated_gaussian_retains_intensity_gradient() -> None:
    intensity = torch.ones(2, 4, 4, requires_grad=True)
    noise = _sample_truncated_normal_like(
        intensity,
        mean=0.01,
        std=0.01,
        minimum=-0.01,
        maximum=0.03,
    )
    output = intensity + noise * intensity.mean().detach()
    output.sum().backward()
    torch.testing.assert_close(intensity.grad, torch.ones_like(intensity))

