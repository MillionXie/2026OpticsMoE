from __future__ import annotations

import math

import torch
from torch import nn

from ..large_scale_continue import (
    LargeScaleP11Model,
    TrainableEMA,
    build_layerwise_optimizer,
    classification_loss,
    validate_config,
)
from ..model import QwenStemSeparableOpticalImageNetBackbone
from .test_model import fake_stem, model_config


def test_large_recipe_model_keeps_deployable_p11_state_dict(tmp_path) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    config = {**model_config(), "stage_drop_path_rate": 0.05}
    torch.manual_seed(9)
    base = QwenStemSeparableOpticalImageNetBackbone(stem, config)
    torch.manual_seed(9)
    recipe = LargeScaleP11Model(stem, config)
    assert recipe.state_dict().keys() == base.state_dict().keys()
    assert recipe.stage_drop_probabilities()[0] == 0.0
    assert math.isclose(recipe.stage_drop_probabilities()[-1], 0.05)


def test_layerwise_optimizer_partitions_parameters_and_decays_early_lr(
    tmp_path,
) -> None:
    config = {**model_config(), "stage_drop_path_rate": 0.05}
    model = LargeScaleP11Model(fake_stem(tmp_path / "stem.pt"), config)
    optimizer, schema = build_layerwise_optimizer(
        model,
        {
            "optimizer": {
                "phase_learning_rate": 0.007,
                "electronic_learning_rate": 0.00025,
                "adapter_learning_rate": 0.00025,
                "head_learning_rate": 0.0005,
                "layer_decay": 0.92,
                "weight_decay": 0.02,
            }
        },
    )
    assigned = sum(group["parameter_elements"] for group in schema)
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert assigned == expected
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert math.isclose(rates["stage07.phase"], 0.007)
    assert rates["stage00.phase"] < rates["stage07.phase"]
    assert rates["stage00.phase"] > 0.003


def test_soft_target_bce_accepts_mixup_targets_and_backpropagates() -> None:
    logits = torch.randn(4, 10, requires_grad=True)
    labels_a = torch.tensor([0, 1, 2, 3])
    labels_b = torch.tensor([4, 5, 6, 7])
    loss = classification_loss(
        logits,
        labels_a,
        labels_b,
        0.25,
        {"loss": {"mode": "bce_soft_targets", "label_smoothing": 0.0}},
    )
    assert loss.ndim == 0
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())


def test_soft_target_bce_keeps_standard_mean_scale() -> None:
    logits = torch.zeros(4, 1000)
    labels = torch.tensor([0, 1, 2, 3])
    loss = classification_loss(
        logits,
        labels,
        labels,
        1.0,
        {"loss": {"mode": "bce_soft_targets", "label_smoothing": 0.0}},
    )
    assert math.isclose(float(loss), math.log(2.0), rel_tol=1.0e-6)


def test_trainable_ema_round_trip_and_scope_restores_live_weights() -> None:
    model = nn.Linear(3, 2)
    ema = TrainableEMA(model, decay=0.9, warmup_updates=0)
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2.0)
    live = model.weight.detach().clone()
    ema.update(model)
    with ema.apply(model):
        assert not torch.equal(model.weight, live)
    torch.testing.assert_close(model.weight, live)
    state = ema.state_dict()
    restored = TrainableEMA(model, decay=0.9, warmup_updates=0)
    restored.load_state_dict(state)
    assert restored.updates == 1
    assert not torch.equal(original, live)


def test_validate_config_accepts_single_and_multi_rank_world_sizes() -> None:
    class FakeContext:
        world_size = 5

    validate_config(
        {
            "training": {"expected_world_size": 5},
            "objective": {"mode": "supervised_imagenet1k"},
        },
        FakeContext(),  # type: ignore[arg-type]
    )
