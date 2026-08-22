from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill.datasets import (
    CLIP_MEAN,
    CLIP_STD,
)

from ..general_backbone_pretraining import (
    CompactOpticalImageNetStudent,
    SubsetEpochViewSampler,
    batch_contrastive_loss,
    load_p06_settings,
    sha256_file,
    stratified_base_indices,
)
from ..formal_settings import load_formal_settings
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


def test_descriptor_mlp_decouples_classifier_from_clip_projection() -> None:
    model = CompactOpticalImageNetStudent(
        _tiny_optical(),
        selected_stage_indices=(0, 1),
        pool_size=2,
        projection_dim=7,
        num_classes=5,
        classifier_mode="descriptor_mlp",
        classifier_hidden_dim=11,
        classifier_dropout=0.1,
    )
    logits, embedding, descriptor = model(torch.rand(2, 3, 20, 20))
    assert logits.shape == (2, 5)
    assert embedding.shape == (2, 7)
    assert descriptor.shape == (2, 48)
    assert model.classifier[0].in_features == 48
    assert model.classifier[3].out_features == 5


def test_compatible_checkpoint_load_restores_encoder_but_reinitializes_head(
    tmp_path: Path,
) -> None:
    source = CompactOpticalImageNetStudent(
        _tiny_optical(),
        selected_stage_indices=(0, 1),
        pool_size=2,
        projection_dim=7,
        num_classes=5,
    )
    checkpoint = tmp_path / "source.pt"
    torch.save({"model": source.state_dict(), "epoch": 3}, checkpoint)
    target = CompactOpticalImageNetStudent(
        _tiny_optical(),
        selected_stage_indices=(0, 1),
        pool_size=2,
        projection_dim=7,
        num_classes=5,
        classifier_mode="descriptor_mlp",
        classifier_hidden_dim=11,
    )
    report = target.load_pretraining_checkpoint(
        checkpoint,
        sha256_file(checkpoint),
        load_mode="compatible",
    )
    assert report["load"]["mode"] == "compatible"
    assert not any(
        key.startswith("encoder.") for key in report["load"]["missing_keys"]
    )
    assert "classifier.weight" in report["load"]["unexpected_keys"]
    assert "classifier.0.weight" in report["load"]["missing_keys"]
    for key, value in source.encoder.state_dict().items():
        assert torch.equal(value, target.encoder.state_dict()[key])


def test_expanded_checkpoint_load_interpolates_depth_and_phase_resolution(
    tmp_path: Path,
) -> None:
    source = CompactOpticalImageNetStudent(
        _tiny_optical(),
        selected_stage_indices=(0, 1),
        pool_size=2,
        projection_dim=7,
        num_classes=5,
    )
    checkpoint = tmp_path / "compact.pt"
    torch.save({"model": source.state_dict(), "epoch": 10}, checkpoint)
    expanded_optical = replace(_tiny_optical(), canvas_size=20, num_stages=4)
    target = CompactOpticalImageNetStudent(
        expanded_optical,
        selected_stage_indices=(0, 3),
        pool_size=2,
        projection_dim=7,
        num_classes=5,
    )
    report = target.load_pretraining_checkpoint(
        checkpoint,
        sha256_file(checkpoint),
        load_mode="expanded",
    )
    load = report["load"]
    assert load["source_stage_count"] == 2
    assert load["target_stage_count"] == 4
    assert len(load["stage_mapping"]) == 4
    expected_first = F.interpolate(
        source.encoder.stages[0].raw_phase.detach().unsqueeze(0),
        size=(20, 20),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)
    expected_last = F.interpolate(
        source.encoder.stages[1].raw_phase.detach().unsqueeze(0),
        size=(20, 20),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)
    assert torch.allclose(target.encoder.stages[0].raw_phase, expected_first)
    assert torch.allclose(target.encoder.stages[-1].raw_phase, expected_last)
    assert sum(parameter.numel() for parameter in target.encoder.phase_parameters()) == 4800


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


def test_block_shuffle_preserves_coverage_locality_and_epoch_view_cycle() -> None:
    class Dataset:
        views = 4

    base_indices = list(range(24))
    epoch_zero = SubsetEpochViewSampler(
        Dataset(),
        base_indices,
        shuffle=True,
        seed=17,
        rank=0,
        world_size=1,
        shuffle_block_size=4,
    )
    epoch_zero.set_epoch(0)
    first = list(epoch_zero)
    epoch_zero.set_epoch(0)
    assert list(epoch_zero) == first

    first_base = [index // 4 for index in first]
    assert sorted(first_base) == base_indices
    # Each shuffled unit must still be a forward-contiguous source block.
    chunks = [first_base[start : start + 4] for start in range(0, 24, 4)]
    assert all(chunk == list(range(chunk[0], chunk[0] + len(chunk))) for chunk in chunks)

    epoch_zero.set_epoch(1)
    second = list(epoch_zero)
    assert [index // 4 for index in second] != first_base
    by_sample_first = {index // 4: index % 4 for index in first}
    by_sample_second = {index // 4: index % 4 for index in second}
    assert all(
        by_sample_second[sample] == (by_sample_first[sample] + 1) % 4
        for sample in base_indices
    )


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
    assert settings.training.shuffle_block_size == 4096


def test_capacity_expansion_is_million_scale_and_electronically_bounded() -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "p06_imagenet_capacity_12x192.yaml"
    settings = load_p06_settings(config)
    assert settings.source_checkpoint_load_mode == "integrity_only"
    assert settings.initial_checkpoint_load_mode == "expanded"
    assert settings.model.selected_stage_indices == (2, 5, 8, 11)
    architecture = load_formal_settings(settings.architecture_config).base
    model = CompactOpticalImageNetStudent(
        architecture.optical,
        selected_stage_indices=settings.model.selected_stage_indices,
        pool_size=settings.model.pool_size,
        projection_dim=settings.model.projection_dim,
        num_classes=settings.model.num_classes,
        classifier_mode=settings.model.classifier_mode,
        classifier_hidden_dim=settings.model.classifier_hidden_dim,
        classifier_dropout=settings.model.classifier_dropout,
    )
    report = model.parameter_report()
    assert architecture.optical.canvas_size == 192
    assert architecture.optical.num_stages == 12
    assert report["phase_parameters"] == 12 * 3 * 192 * 192
    assert 1_000_000 <= report["phase_parameters"] <= 2_000_000
    assert report["residual_electronic_parameters"] < 1_000_000
    assert (
        report["residual_electronic_parameters"]
        + report["pretraining_head_parameters"]
        < 2_000_000
    )

def test_eight_stage_224_capacity_keeps_pixel_size_and_joint_bp() -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "p06_imagenet_8x224_screen_projected.yaml"
    settings = load_p06_settings(config)
    architecture = load_formal_settings(settings.architecture_config).base
    model = CompactOpticalImageNetStudent(
        architecture.optical,
        selected_stage_indices=settings.model.selected_stage_indices,
        pool_size=settings.model.pool_size,
        projection_dim=settings.model.projection_dim,
        num_classes=settings.model.num_classes,
        classifier_mode=settings.model.classifier_mode,
        classifier_hidden_dim=settings.model.classifier_hidden_dim,
        classifier_dropout=settings.model.classifier_dropout,
    )
    report = model.parameter_report()
    assert architecture.optical.canvas_size == 224
    assert architecture.optical.num_stages == 8
    assert architecture.optical.pixel_size_m == 1.6e-5
    assert architecture.optical.electronic_skip_downsample_factor == 7
    assert settings.training.head_warmup_epochs == 0
    assert settings.training.joint_epochs == 3
    assert settings.initial_checkpoint_load_mode == "expanded"
    assert settings.model.selected_stage_indices == (1, 3, 5, 7)
    assert report["phase_parameters"] == 8 * 3 * 224 * 224
    assert report["phase_parameters"] == 1_204_224
    assert report["residual_electronic_parameters"] < 1_000_000
    assert (
        report["residual_electronic_parameters"]
        + report["pretraining_head_parameters"]
        < 2_000_000
    )

    spatial_settings = load_p06_settings(
        root / "configs" / "p06_imagenet_8x224_screen_spatial.yaml"
    )
    spatial_model = CompactOpticalImageNetStudent(
        architecture.optical,
        selected_stage_indices=spatial_settings.model.selected_stage_indices,
        pool_size=spatial_settings.model.pool_size,
        projection_dim=spatial_settings.model.projection_dim,
        num_classes=spatial_settings.model.num_classes,
        classifier_mode=spatial_settings.model.classifier_mode,
    )
    spatial_report = spatial_model.parameter_report()
    assert spatial_settings.model.pool_size == 8
    assert spatial_report["descriptor_dim"] == 1536
    assert (
        spatial_report["residual_electronic_parameters"]
        + spatial_report["pretraining_head_parameters"]
        < 2_000_000
    )

    full = load_p06_settings(
        root / "configs" / "p06_imagenet_8x224_full_spatial.yaml"
    )
    assert full.source_checkpoint_load_mode == "integrity_only"
    assert full.initial_checkpoint_load_mode == "strict"
    assert full.training.head_warmup_epochs == 0
    assert full.training.joint_epochs == 12
    assert full.training.train_samples_per_class is None
    assert full.training.validation_samples_per_class == 50
    assert full.training.run_final_ablations
