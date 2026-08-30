from __future__ import annotations

from pathlib import Path

import pytest

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff.modeling import (
    EarlyRobustTradeoffReplacement,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_early_robust_tradeoff.settings import (
    load_settings,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.modeling import (
    STAGE_ARCHITECTURES,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "floor", "ccd_mean", "amplitude_range", "phase_range"),
    [
        (
            "accuracy_first_floor0p1.yaml",
            0.001,
            0.02,
            (0.0, 0.05),
            (0.0, 0.05),
        ),
        (
            "balanced_floor0p5.yaml",
            0.005,
            0.04,
            (0.02, 0.08),
            (0.02, 0.08),
        ),
    ],
)
def test_release_tradeoff_contracts(
    filename: str,
    floor: float,
    ccd_mean: float,
    amplitude_range: tuple[float, float],
    phase_range: tuple[float, float],
) -> None:
    settings = load_settings(ROOT / "configs" / "release" / filename)
    assert settings.optical_fusion_minimum == pytest.approx(floor)
    assert settings.language_optical_ccd_noise_mean_fraction == pytest.approx(ccd_mean)
    assert (
        settings.language_optical_amplitude_zero_order_intensity_min,
        settings.language_optical_amplitude_zero_order_intensity_max,
    ) == pytest.approx(amplitude_range)
    assert (
        settings.language_optical_phase_zero_order_intensity_min,
        settings.language_optical_phase_zero_order_intensity_max,
    ) == pytest.approx(phase_range)
    assert settings.language_optical_zero_order_random_relative_phase is True
    assert settings.evaluate_test_each_epoch is False
    assert settings.resume_optimizer_state is False
    assert settings.epochs == 32
    assert settings.optimizer_steps_per_epoch == 12
    assert settings.continuation_checkpoint.name == (
        "ema_best_train_loss_checkpoint.pt"
    )


def test_checkpoint_topology_is_the_audited_stage_a_contract() -> None:
    assert EarlyRobustTradeoffReplacement.checkpoint_architecture == (
        STAGE_ARCHITECTURES["optical_calibration"]
    )


@pytest.mark.parametrize(
    "filename", ["accuracy_first_quick210.yaml", "balanced_quick210.yaml"]
)
def test_hardware_configs_are_exact_quick210_contracts(filename: str) -> None:
    settings = load_settings(ROOT / "configs" / "hardware" / filename)
    assert settings.gallery_images_per_sku == 1
    assert settings.train_limit_per_sku == 10
    assert settings.test_limit_per_sku == 10
    assert settings.reserve_test_before_train is True
    assert settings.hardware_ccd_flip_vertical is False
    assert settings.hardware_ccd_flip_horizontal is False
