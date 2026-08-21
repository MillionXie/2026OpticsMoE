from __future__ import annotations

from pathlib import Path

import torch

from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.datasets import (
    CLIP_MEAN,
    CLIP_STD,
)

from ..general_backbone_pretraining import (
    CompactOpticalImageNetStudent,
    SubsetEpochViewSampler,
    batch_contrastive_loss,
    load_p06_settings,
    stratified_base_indices,
)
from ..settings import OpticalConfig


def _tiny_optical() -> OpticalConfig:
    return OpticalConfig(
        canvas_size=16,
        input_channels=3,
        num_stages=2,
        wavelength_m=5.32e-7,
        pixel_size_m=1.6e-5,
        propagation_distance_m=0.05,
        phase_init_std=0.05,
        layernorm_eps=1e-5,
        residual_mode="constrained",
        residual_main_init=0.5,
        residual_main_min=0.5,
        normalize_branch_rms=True,
        electronic_skip_mode="lowres",
        electronic_skip_hidden_channels=8,
        electronic_skip_downsample_factor=4,
        electronic_skip_scale_init=0.1,
        electronic_skip_scale_max=0.25,
        long_skip_enabled=False,
        long_skip_weight_init=0.0,
        long_skip_weight_max=0.0,
        readout_mode="mlp",
        pool_size=2,
        hidden_dim=8,
        conv_channels=4,
        dropout=0.0,
    )


def test_clip_denormalization_and_feature_contract() -> None:
    model = CompactOpticalImageNetStudent(
        _tiny_optical(),
        selected_stage_indices=(0, 1),
        pool_size=2,
        projection_dim=7,
        num_classes=5,
    )
    raw = torch.rand(2, 3, 20, 20)
    mean = torch.tensor(CLIP_MEAN).reshape(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).reshape(1, 3, 1, 1)
    normalized = (raw - mean) / std
    assert torch.allclose(model.denormalize_clip_input(normalized), raw, atol=1e-6)
    logits, embedding, descriptor = model(normalized)
    assert logits.shape == (2, 5)
    assert embedding.shape == (2, 7)
    assert descriptor.shape == (2, 48)
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_stratified_sampler_preserves_full_cache_indices_and_views() -> None:
    targets = [0, 0, 0, 1, 1, 1]
    selected = stratified_base_indices(targets, per_class=2, seed=9)
    assert len(selected) == 4
    assert [targets[index] for index in selected].count(0) == 2
    assert [targets[index] for index in selected].count(1) == 2

    class Dataset:
        views = 4

    rank_zero = SubsetEpochViewSampler(
        Dataset(), selected, shuffle=False, seed=9, rank=0, world_size=2
    )
    rank_one = SubsetEpochViewSampler(
        Dataset(), selected, shuffle=False, seed=9, rank=1, world_size=2
    )
    composite = list(rank_zero) + list(rank_one)
    assert sorted(index // 4 for index in composite) == selected
    assert all(0 <= index % 4 < 4 for index in composite)


def test_contrastive_distillation_penalizes_collapsed_or_mismatched_features() -> None:
    teacher = torch.eye(4)
    matched = batch_contrastive_loss(teacher, teacher, temperature=0.07)
    collapsed = batch_contrastive_loss(
        torch.ones_like(teacher), teacher, temperature=0.07
    )
    mismatched = batch_contrastive_loss(
        teacher.roll(1, dims=0), teacher, temperature=0.07
    )
    assert matched < collapsed
    assert matched < mismatched


def test_full_imagenet_config_uses_every_base_sample() -> None:
    config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "p06_imagenet_full_stage1.yaml"
    )
    settings = load_p06_settings(config)
    assert settings.training.train_samples_per_class is None
    assert settings.training.validation_samples_per_class == 50
    assert settings.training.epochs == 10
