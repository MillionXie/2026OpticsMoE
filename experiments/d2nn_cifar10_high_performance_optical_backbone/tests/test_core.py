from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from experiments.d2nn_cifar10_high_performance_optical_backbone.datasets import _split
from experiments.d2nn_cifar10_high_performance_optical_backbone.model import OpticalClassifier
from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import ResidualMixer, physical_phase
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
    assert float(mixer.main_weight()) >= 0.35


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
