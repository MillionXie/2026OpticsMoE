from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from experiments.d2nn_cifar100c10_fixed_feedback_20stage400.datasets import epoch_order
from experiments.d2nn_cifar100c10_fixed_feedback_20stage400.model import OpticalClassifier
from experiments.d2nn_cifar100c10_fixed_feedback_20stage400.settings import OpticalConfig


def config(stages: int = 3, size: int = 16) -> OpticalConfig:
    return OpticalConfig(
        canvas_size=size,
        num_stages=stages,
        wavelength_m=532e-9,
        pixel_size_m=16e-6,
        propagation_distance_m=0.05,
        phase_parameterization="sigmoid",
        phase_init="zeros",
        layernorm_eps=1e-5,
        residual_main_init=0.1,
        residual_skip_init=0.9,
        readout_pool_size=4,
        readout_hidden_dim=8,
        num_output_classes=10,
    )


def gradients(model: OpticalClassifier, images: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(images), labels).backward()
    return [stage.raw_phase.grad.detach().clone() for stage in model.stages]


def test_formal_phase_parameter_count_and_shapes() -> None:
    formal = config(stages=20, size=400)
    model = OpticalClassifier(formal)
    assert model.parameter_report()["phase"] == 3_200_000
    assert len(model.stages) == 20
    assert all(tuple(stage.raw_phase.shape) == (400, 400) for stage in model.stages)
    assert all(tuple(stage.propagator.transfer_function.shape) == (400, 400) for stage in model.stages)


def test_raw_zero_means_pi_and_residual_has_two_trainable_weights() -> None:
    model = OpticalClassifier(config())
    assert torch.allclose(model.phase_stack(), torch.full_like(model.phase_stack(), math.pi))
    weights = model.residual_weights()
    assert tuple(weights.shape) == (3, 2)
    assert torch.allclose(weights[:, 0], torch.full((3,), 0.1))
    assert torch.allclose(weights[:, 1], torch.full((3,), 0.9))
    assert all(stage.residual.logits.requires_grad for stage in model.stages)


def test_twenty_oeo_outputs_are_nonnegative_and_return_ccd_details() -> None:
    model = OpticalClassifier(config(stages=3))
    logits, details = model(torch.rand(2, 1, 16, 16), return_intermediates=True)
    assert tuple(logits.shape) == (2, 10)
    assert len(details["stages"]) == 3
    assert all(torch.all(stage["intensity"] >= 0) for stage in details["stages"])
    assert all(torch.all(stage["activated"] >= 0) for stage in details["stages"])
    assert all(torch.all(stage["reloaded"] >= 0) for stage in details["stages"])


def test_feedback_mode_never_changes_forward() -> None:
    model = OpticalClassifier(config())
    images = torch.rand(2, 1, 16, 16)
    pretrained = model.snapshot_feedback_phases()
    model.configure_feedback("bp")
    bp = model(images)
    model.configure_feedback("fa_pretrained", pretrained_phases=pretrained)
    fixed = model(images)
    model.configure_feedback("fa_random", random_seed=7)
    random = model(images)
    assert torch.equal(bp, fixed)
    assert torch.equal(bp, random)


def test_fa_pretrained_matches_bp_at_finetuning_start() -> None:
    torch.manual_seed(3)
    model = OpticalClassifier(config())
    images = torch.rand(2, 1, 16, 16)
    labels = torch.tensor([2, 7])
    pretrained = model.snapshot_feedback_phases()
    model.configure_feedback("bp")
    bp = gradients(model, images, labels)
    model.configure_feedback("fa_pretrained", pretrained_phases=pretrained)
    fixed = gradients(model, images, labels)
    for exact, approximate in zip(bp, fixed, strict=True):
        assert torch.allclose(exact, approximate, rtol=3e-4, atol=3e-6)


def test_random_feedback_changes_upstream_gradients_but_not_current_error() -> None:
    torch.manual_seed(4)
    model = OpticalClassifier(config())
    images = torch.rand(2, 1, 16, 16)
    labels = torch.tensor([1, 6])
    model.configure_feedback("bp")
    bp = gradients(model, images, labels)
    model.configure_feedback("fa_random", random_seed=99)
    random = gradients(model, images, labels)
    assert torch.allclose(bp[-1], random[-1], rtol=3e-4, atol=3e-6)
    assert not torch.allclose(bp[0], random[0], rtol=1e-3, atol=1e-5)


def test_fixed_operator_does_not_cache_a_sample_error() -> None:
    torch.manual_seed(5)
    model = OpticalClassifier(config())
    images = torch.rand(2, 1, 16, 16)
    pretrained = model.snapshot_feedback_phases()
    model.configure_feedback("fa_pretrained", pretrained_phases=pretrained)
    first = gradients(model, images, torch.tensor([0, 1]))
    second = gradients(model, images, torch.tensor([8, 9]))
    assert not torch.allclose(first[0], second[0])


def test_feedback_is_buffer_not_parameter() -> None:
    model = OpticalClassifier(config())
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    assert not any("feedback_phase" in name for name in parameter_names)
    assert sum("feedback_phase" in name for name in buffer_names) == 3


def test_rotating_subset_covers_full_dataset_without_duplication() -> None:
    first = epoch_order(45, epoch=1, seed=42, limit=15)
    second = epoch_order(45, epoch=2, seed=42, limit=15)
    third = epoch_order(45, epoch=3, seed=42, limit=15)
    assert len(set(first + second + third)) == 45
    assert epoch_order(45, epoch=1, seed=42, limit=15) == first


@pytest.mark.parametrize("bad", [(0.8, 0.3), (0.0, 1.0)])
def test_invalid_residual_initialization_is_visible(bad: tuple[float, float]) -> None:
    model_config = config()
    object.__setattr__(model_config, "residual_main_init", bad[0])
    object.__setattr__(model_config, "residual_skip_init", bad[1])
    with pytest.raises(ValueError):
        OpticalClassifier(model_config)
