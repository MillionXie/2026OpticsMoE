from __future__ import annotations

import math

import torch

from experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400.datasets import (
    BalancedClassBatchSampler,
)
from experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400.losses import (
    contrastive_transfer_loss,
    supervised_contrastive_loss,
)
from experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400.model import (
    OpticalEmbeddingNetwork,
)
from experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400.settings import (
    BalancedBatchConfig,
    OpticalConfig,
)


def config() -> OpticalConfig:
    return OpticalConfig(
        canvas_size=20,
        num_stages=3,
        wavelength_m=532e-9,
        pixel_size_m=16e-6,
        propagation_distance_m=0.05,
        phase_parameterization="sigmoid",
        phase_init="zeros",
        layernorm_eps=1e-5,
        residual_main_init=0.35,
        residual_skip_init=0.65,
        readout_pool_size=4,
        embedding_dim=12,
        embedding_dropout=0.1,
    )


def test_embedding_shape_norm_and_signed_values() -> None:
    model = OpticalEmbeddingNetwork(config()).eval()
    embedding = model(torch.rand(5, 1, 20, 20))
    assert tuple(embedding.shape) == (5, 12)
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.any(embedding < 0) and torch.any(embedding > 0)


def test_residual_is_adaptive_and_initialized_035_065() -> None:
    model = OpticalEmbeddingNetwork(config())
    weights = model.residual_weights()
    assert torch.allclose(weights[:, 0], torch.full((3,), 0.35))
    assert torch.allclose(weights[:, 1], torch.full((3,), 0.65))
    assert all(stage.residual.logits.requires_grad for stage in model.stages)


def test_phase_zero_still_means_pi() -> None:
    model = OpticalEmbeddingNetwork(config())
    assert torch.allclose(model.phase_stack(), torch.full_like(model.phase_stack(), math.pi))


def test_dropout_is_train_only() -> None:
    model = OpticalEmbeddingNetwork(config())
    images = torch.rand(4, 1, 20, 20)
    model.eval()
    assert torch.equal(model(images), model(images))
    model.train()
    assert not torch.equal(model(images), model(images))


def test_supcon_and_prototype_loss_are_finite_with_gradients() -> None:
    torch.manual_seed(7)
    features = torch.nn.functional.normalize(torch.randn(4, 2, 12), dim=-1).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1])
    total, parts = contrastive_transfer_loss(
        features,
        labels,
        contrastive_temperature=0.1,
        prototype_temperature=0.1,
        supcon_weight=1.0,
        prototype_weight=0.5,
    )
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(parts["supcon"])
    assert torch.isfinite(parts["prototype"])
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_supcon_rejects_missing_positive() -> None:
    features = torch.nn.functional.normalize(torch.randn(3, 1, 8), dim=-1)
    try:
        supervised_contrastive_loss(features, torch.tensor([0, 1, 2]), 0.1)
    except ValueError as exc:
        assert "two views" in str(exc)
    else:
        raise AssertionError("Expected missing-view validation error")


def test_balanced_sampler_has_exact_pk_composition_and_is_deterministic() -> None:
    targets = [label for label in range(5) for _ in range(10)]
    batch_config = BalancedBatchConfig(3, 4, 2, 6)
    first = list(BalancedClassBatchSampler(targets, batch_config, seed=42, epoch=1))
    second = list(BalancedClassBatchSampler(targets, batch_config, seed=42, epoch=1))
    assert first == second
    for batch in first:
        counts: dict[int, int] = {}
        for index in batch:
            counts[targets[index]] = counts.get(targets[index], 0) + 1
        assert len(counts) == 3
        assert set(counts.values()) == {4}


def test_feedback_mode_changes_backward_not_forward() -> None:
    torch.manual_seed(11)
    model = OpticalEmbeddingNetwork(config()).eval()
    images = torch.rand(4, 1, 20, 20)
    phases = model.snapshot_feedback_phases()
    model.configure_feedback("bp")
    bp_output = model(images)
    model.configure_feedback("fa_pretrained", pretrained_phases=phases)
    fixed_output = model(images)
    model.configure_feedback("fa_random", random_seed=9)
    random_output = model(images)
    assert torch.equal(bp_output, fixed_output)
    assert torch.equal(bp_output, random_output)


def test_formal_phase_parameter_count() -> None:
    formal = config()
    object.__setattr__(formal, "canvas_size", 400)
    object.__setattr__(formal, "num_stages", 20)
    model = OpticalEmbeddingNetwork(formal)
    assert model.parameter_report()["phase"] == 3_200_000
