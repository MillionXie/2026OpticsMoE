from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from experiments.d2nn_cifar10_high_performance_optical_backbone.datasets import _split
from experiments.d2nn_cifar10_high_performance_optical_backbone.formal_settings import load_formal_settings
from experiments.d2nn_cifar10_high_performance_optical_backbone.model import OpticalClassifier
from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import (
    OpticalOEOStage,
    ResidualMixer,
    physical_phase,
)
from experiments.d2nn_cifar10_high_performance_optical_backbone.settings import load_settings


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def test_settings_and_stratified_split() -> None:
    settings = load_settings(CONFIG)
    assert settings.num_classes == 10
    train, validation = _split([0] * 10 + [1] * 10, validation_per_class=2, seed=42)
    assert len(train) == 16
    assert len(validation) == 4
    assert not set(train) & set(validation)


def test_constrained_residual_initialization_and_bound() -> None:
    mixer = ResidualMixer("constrained", main_init=0.5, main_min=0.35)
    assert torch.isclose(mixer.main_weight(), torch.tensor(0.5), atol=1e-5)
    mixer.logit.data.fill_(-100.0)
    assert float(mixer.main_weight()) >= 0.35 - 1e-6


def test_phase_parameterization_is_physical() -> None:
    raw = torch.tensor([-100.0, 0.0, 100.0])
    phase = physical_phase(raw)
    assert torch.all(phase >= 0.0)
    assert torch.all(phase <= 2.0 * torch.pi)


def test_forward_backward_and_ablation_shapes() -> None:
    settings = load_settings(CONFIG)
    optical = replace(settings.optical, canvas_size=16, pool_size=2, hidden_dim=16)
    model = OpticalClassifier(optical, num_classes=10)
    images = torch.rand(2, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert all(stage.raw_phase.grad is not None for stage in model.stages)
    assert model(images, ablation="optical_off").shape == (2, 10)
    assert model(images, ablation="phase_random").shape == (2, 10)
    assert model(images, ablation="phase_shuffle").shape == (2, 10)


def test_optical_off_does_not_depend_on_phase() -> None:
    settings = load_settings(CONFIG)
    optical = replace(settings.optical, canvas_size=16, pool_size=2, hidden_dim=16, dropout=0.0)
    model = OpticalClassifier(optical, num_classes=10).eval()
    images = torch.rand(2, 3, 32, 32)
    before = model(images, ablation="optical_off")
    for stage in model.stages:
        stage.raw_phase.data.normal_()
    after = model(images, ablation="optical_off")
    torch.testing.assert_close(before, after)


def _stage() -> OpticalOEOStage:
    return OpticalOEOStage(
        size=8,
        channels=2,
        wavelength_m=5.32e-7,
        pixel_size_m=1.6e-5,
        distance_m=0.05,
        phase_init_std=0.05,
        layernorm_eps=1e-5,
        residual_mode="constrained",
        residual_main_init=0.5,
        residual_main_min=0.35,
        normalize_branch_rms=True,
        random_seed=17,
    )


def test_fixed_feedback_matches_bp_when_connector_matches_current_phase() -> None:
    bp = _stage()
    fixed = _stage()
    fixed.load_state_dict(bp.state_dict())
    fixed.set_feedback("fa_pretrained", bp.phase().detach())
    bp_input = torch.rand(2, 2, 8, 8, requires_grad=True)
    fixed_input = bp_input.detach().clone().requires_grad_(True)
    bp(bp_input).square().mean().backward()
    fixed(fixed_input).square().mean().backward()
    torch.testing.assert_close(bp_input.grad, fixed_input.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(bp.raw_phase.grad, fixed.raw_phase.grad, rtol=2e-4, atol=2e-5)


def test_random_feedback_changes_connector_but_not_forward() -> None:
    bp = _stage()
    fixed = _stage()
    fixed.load_state_dict(bp.state_dict())
    random_phase = 2.0 * torch.pi * torch.rand_like(bp.phase())
    fixed.set_feedback("fa_random", random_phase)
    bp_input = torch.rand(2, 2, 8, 8, requires_grad=True)
    fixed_input = bp_input.detach().clone().requires_grad_(True)
    bp_output = bp(bp_input)
    fixed_output = fixed(fixed_input)
    torch.testing.assert_close(bp_output, fixed_output)
    bp_output.square().mean().backward()
    fixed_output.square().mean().backward()
    assert not torch.allclose(bp_input.grad, fixed_input.grad)
    torch.testing.assert_close(bp.raw_phase.grad, fixed.raw_phase.grad, rtol=2e-4, atol=2e-5)


def test_formal_pilot_has_only_the_expected_contract() -> None:
    formal = load_formal_settings(CONFIG.parent / "formal_pilot.yaml")
    assert formal.formal.finetune_seeds == (2026, 2027, 2028)
    assert formal.formal.head_warmup_epochs == 10
    assert formal.formal.finetune_epochs == 20
    assert len(formal.formal.source_checkpoint_sha256) == 64


def test_a07_changes_only_the_optical_floor_contract() -> None:
    baseline = load_settings(CONFIG.parent / "a04_cifar100_to_cifar10.yaml")
    candidate = load_settings(CONFIG.parent / "a07_high_optical_cifar100_to_cifar10.yaml")
    assert baseline.optical.residual_main_min == 0.35
    assert candidate.optical.residual_main_min == 0.50
    assert baseline.optical.residual_main_init == candidate.optical.residual_main_init == 0.50
    assert baseline.optimizer == candidate.optimizer
    assert baseline.training == candidate.training
