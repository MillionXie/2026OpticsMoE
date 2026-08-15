from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ..modeling import RobustOpticalRetrievalReadout
from ..optics.hybrid import (
    LearnableResidualFusion,
    RobustHybridOpticalCore,
    translate_zero_fill,
)
from ..settings import load_settings


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
CONFIG = EXPERIMENT_DIR / "configs" / "release" / "robust_hybrid_moe4.yaml"
REGULARIZED_CONFIG = (
    EXPERIMENT_DIR
    / "configs"
    / "continuation"
    / "regularized_electronic_finetune.yaml"
)
STAGE1_RETRAIN_CONFIG = (
    EXPERIMENT_DIR / "configs" / "retrain" / "stage1_grocery31_pretrain.yaml"
)
STAGE2_RETRAIN_CONFIG = (
    EXPERIMENT_DIR / "configs" / "retrain" / "stage2_grocery10_finetune.yaml"
)


def test_robust_configuration_is_deliberate() -> None:
    settings = load_settings(CONFIG)
    assert settings.phase_learning_rate == pytest.approx(1.0e-4)
    assert settings.k_space_constraint_enabled
    assert settings.phase_dropout_mode == "block_phase_bypass"
    assert settings.phase_dropout_p == pytest.approx(0.05)
    assert settings.hybrid_residual_initial_weight == pytest.approx(0.8)
    assert settings.input_shift_max_px == 12
    assert settings.phase_shift_max_px == 12
    assert settings.ccd_shift_max_px == 12
    assert not settings.transformer_residual_enabled


def test_regularized_continuation_reduces_memorization_pressure() -> None:
    settings = load_settings(REGULARIZED_CONFIG)
    assert settings.optimizer_steps_per_epoch is None
    assert settings.weight_decay == pytest.approx(1.0e-3)
    assert settings.phase_learning_rate == 0.0
    assert settings.router_learning_rate == 0.0
    assert settings.adapter_learning_rate == pytest.approx(1.0e-5)
    assert settings.readout_learning_rate == pytest.approx(2.0e-5)
    assert settings.lambda_kd > settings.lambda_ret
    assert settings.lambda_teacher_gallery > settings.lambda_gallery
    assert settings.crop_scale_min == pytest.approx(0.75)
    assert settings.readout_dropout == pytest.approx(0.25)


def test_two_stage_retraining_uses_natural_epochs_and_normal_lrs() -> None:
    stage1 = load_settings(STAGE1_RETRAIN_CONFIG)
    stage2 = load_settings(STAGE2_RETRAIN_CONFIG)
    assert len(stage1.selected_skus) == 31
    assert len(stage2.selected_skus) == 10
    assert stage1.optimizer_steps_per_epoch is None
    assert stage2.optimizer_steps_per_epoch is None
    assert stage1.learning_rate == pytest.approx(1.0e-4)
    assert stage1.adapter_learning_rate == pytest.approx(1.0e-4)
    assert stage1.readout_learning_rate == pytest.approx(1.0e-4)
    assert stage1.router_learning_rate == pytest.approx(5.0e-5)
    assert stage1.phase_learning_rate == pytest.approx(5.0e-5)
    assert stage2.learning_rate == pytest.approx(5.0e-5)
    assert stage2.phase_learning_rate == pytest.approx(1.0e-5)
    assert stage1.weight_decay == pytest.approx(1.0e-4)
    assert stage2.weight_decay == pytest.approx(1.0e-4)
    assert not stage1.evaluate_test_each_epoch
    assert not stage2.evaluate_test_each_epoch


def test_zero_fill_translation_does_not_wrap() -> None:
    value = torch.zeros(1, 4, 4)
    value[0, 0, 0] = 1.0
    translated = translate_zero_fill(value, 1, 2)
    assert translated[0, 1, 2] == 1.0
    assert translated[0, 0, 0] == 0.0
    assert translated.sum() == 1.0
    assert translate_zero_fill(value, -1, -1).sum() == 0.0


def test_residual_fusion_has_learnable_convex_gate_and_small_refiner() -> None:
    module = LearnableResidualFusion(0.8, width=16, dilation=2, dropout=0.0)
    assert float(module.input_weight()) == pytest.approx(0.8)
    optical = torch.rand(2, 16, 16)
    residual = torch.rand(2, 16, 16)
    output = module(optical, residual)
    assert output.shape == optical.shape
    assert torch.all(output >= 0)
    assert sum(parameter.numel() for parameter in module.parameters()) < 1_000
    output.mean().backward()
    assert module.input_weight_logit.grad is not None


def test_enhanced_readout_is_normalized_and_parameter_bounded() -> None:
    module = RobustOpticalRetrievalReadout(
        detector_dim=224, embedding_dim=64, bottleneck_dim=96, dropout=0.0
    )
    output = module(torch.rand(3, 224))
    assert output.shape == (3, 64)
    assert torch.allclose(output.norm(dim=-1), torch.ones(3), atol=1.0e-5)
    assert sum(parameter.numel() for parameter in module.parameters()) < 100_000


def test_full_robust_core_forward_and_backward() -> None:
    settings = load_settings(CONFIG)
    core = RobustHybridOpticalCore(hidden_size=32, max_tokens=224, settings=settings)
    core.train()
    core.set_phase_dropout_active(True)
    inputs = torch.rand(2, 224, 224, requires_grad=True)
    field, routing = core.begin(inputs)
    field = core.run_stage(0, field, routing)
    hidden = core.read_hidden(field, [3, 4], torch.float32, final=True)
    assert hidden.shape == (7, 32)
    assert core.current_detector_readout is not None
    assert core.current_detector_readout.shape == (2, 224, 224)
    hidden.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert core.expert_fusions[0].input_weight_logit.grad is not None
    assert core.detector_fusion.input_weight_logit.grad is not None
    gradients = [
        parameter.grad
        for parameter in core.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
