from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ..modeling import (
    LGVQSingleMetricOEO16,
    TrainableQualityFrameStem,
    _phase,
    _phase_modulation,
    _routing_statistics,
)
from ..run import _apply_trainable_scope, _load_compatible_initialization
from ..settings import TARGET_PROMPTS, ExperimentSettings, Geometry
from ..train_cached_deep_readout import _batches as cached_readout_batches
from ..training import (
    MosStratifiedBatchSampler,
    _training_stage_factors,
    curriculum_values,
    soft_spearman_loss,
    train,
    weighted_level_distribution_loss,
)


def _small_settings(tmp_path: Path, *, target_name: str = "spatial") -> ExperimentSettings:
    """Return a fast, non-overlapping optical geometry for CPU contract tests."""

    spatial = target_name == "spatial"
    settings = ExperimentSettings(
        config_path=tmp_path / f"{target_name}.yaml",
        output_dir=tmp_path / f"runs_{target_name}",
        dataset_root=None,
        manifest_path=None,
        vision_cache_path=None,
        language_cache_path=None,
        target_name=target_name,
        prompt=TARGET_PROMPTS[target_name],
        device="cpu",
        random_seed=7,
        geometry=Geometry(
            canvas_size=96,
            active_size=88,
            lane_grid=2 if spatial else 4,
            lane_size=42 if spatial else 18,
            lane_pitch=46 if spatial else 22,
            lane_offset=0 if spatial else 2,
            parallel_expert_size=18 if spatial else 8,
            parallel_expert_pitch=24 if spatial else 10,
            serial_expert_size=24,
            serial_expert_pitch=32,
        ),
        frame_count=4 if spatial else 16,
        maximum_language_tokens=24,
        detector_projection_size=8,
        serial_router_input_size=24,
        parallel_router_intervals=((4, 8), (10, 14)),
        serial_router_intervals=((20, 28), (60, 68)),
        input_shift_pixels=0,
        phase_shift_pixels=0,
        ccd_shift_pixels=0,
        phase_dropout_p=0.0,
        router_noise_std=0.0,
        head_width=16,
        dropout=0.0,
        epochs=1,
        batch_size=2,
        num_workers=0,
        test_interval_epochs=1,
        soft_target_weight=0.0,
        k_space_enabled=False,
        synthetic=True,
    )
    settings.validate()
    return settings


def _inputs(batch: int = 2, frame_count: int = 16) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(11)
    return (
        torch.randn(batch, frame_count, 49, 1024, generator=generator),
        torch.randn(batch, frame_count, 49, 14, generator=generator),
        torch.randn(batch, 4, 2048, generator=generator),
        torch.ones(batch, 4, dtype=torch.bool),
    )


def test_soft_spearman_loss_tracks_rank_order_and_backpropagates() -> None:
    target = torch.tensor([-1.0, -0.2, 0.4, 1.2])
    ordered = target.clone().requires_grad_(True)
    reversed_prediction = target.flip(0)
    good = soft_spearman_loss(ordered, target, temperature=0.05)
    bad = soft_spearman_loss(reversed_prediction, target, temperature=0.05)
    assert float(good) < 0.02
    assert float(bad) > 1.9
    good.backward()
    assert ordered.grad is not None
    assert bool(torch.isfinite(ordered.grad).all())


def test_weighted_level_distribution_prefers_nearby_ordered_level() -> None:
    scores = torch.linspace(-1.0, 1.0, 5)
    base = torch.zeros(2)
    target = torch.tensor([-0.9, 0.9])
    correct = torch.tensor([[8.0, 2.0, 0.0, -2.0, -8.0], [-8.0, -2.0, 0.0, 2.0, 8.0]])
    reversed_logits = correct.flip(-1)
    good = weighted_level_distribution_loss(correct, scores, base, target)
    bad = weighted_level_distribution_loss(reversed_logits, scores, base, target)
    assert float(good) < float(bad)


def test_cached_readout_mos_strata_interleave_score_range() -> None:
    targets = torch.arange(16, dtype=torch.float32)
    payload = {
        "vision": torch.zeros(16, 1, 1, 1),
        "language": torch.zeros(16, 1, 1),
        "mask": torch.ones(16, 1, dtype=torch.bool),
        "targets_normalized": targets,
        "teacher_normalized": targets.clone(),
    }
    batches = list(
        cached_readout_batches(
            payload,
            torch.arange(16),
            batch_size=8,
            generator=torch.Generator().manual_seed(7),
            mos_strata=4,
        )
    )
    first_targets = batches[0][3]
    assert first_targets.min() <= 3
    assert first_targets.max() >= 12
    visited = torch.cat([batch[-1] for batch in batches]).sort().values
    assert torch.equal(visited, torch.arange(16))


def test_mos_stratified_sampler_uses_each_item_once_and_spans_scores() -> None:
    targets = torch.arange(64, dtype=torch.float32)
    sampler = MosStratifiedBatchSampler(
        targets, batch_size=16, strata=8, seed=91
    )
    batches = list(iter(sampler))
    flat = [index for batch in batches for index in batch]
    assert sorted(flat) == list(range(64))
    assert all(max(batch) - min(batch) >= 49 for batch in batches)


def test_curriculum_interpolates_training_only_weights(tmp_path: Path) -> None:
    settings = replace(
        _small_settings(tmp_path),
        epochs=100,
        training_soft_targets_path=tmp_path / "teacher.pt",
        curriculum_enabled=True,
        curriculum_start_epoch=10,
        curriculum_end_epoch=90,
        soft_target_weight=3.0,
        curriculum_soft_target_weight_final=0.25,
        soft_spearman_weight=0.0,
        curriculum_soft_spearman_weight_final=0.12,
        router_noise_std=0.08,
        curriculum_router_noise_std_final=0.01,
        unmodulated_power_fraction_max=0.35,
        curriculum_unmodulated_power_fraction_max_initial=0.20,
    )
    settings.validate()
    start = curriculum_values(settings, 10)
    middle = curriculum_values(settings, 50)
    final = curriculum_values(settings, 90)
    assert start["soft_target_weight"] == pytest.approx(3.0)
    assert middle["soft_target_weight"] == pytest.approx(1.625)
    assert final["soft_target_weight"] == pytest.approx(0.25)
    assert start["soft_spearman_weight"] == pytest.approx(0.0)
    assert final["soft_spearman_weight"] == pytest.approx(0.12)
    assert start["unmodulated_power_fraction_max"] == pytest.approx(0.20)
    assert final["unmodulated_power_fraction_max"] == pytest.approx(0.35)


def test_trainable_frame_stem_exactly_matches_cache_source_implementation() -> None:
    from experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.modeling import (
        FrameStem,
    )

    torch.manual_seed(19)
    source = FrameStem(192).eval()
    destination = TrainableQualityFrameStem().eval()
    destination.load_state_dict(source.state_dict(), strict=True)
    frames = torch.randint(0, 256, (1, 4, 3, 224, 224), dtype=torch.uint8)
    with torch.no_grad():
        expected = source(frames)
        actual = destination(frames)
    assert torch.equal(expected, actual)
    assert actual.shape == (1, 4, 196, 192)


def test_formal_geometry_is_exact_4x4_frame_and_2x2_expert_layout() -> None:
    geometry = Geometry()
    geometry.validate(formal=True)
    assert (geometry.canvas_size, geometry.active_size, geometry.active_margin) == (
        518,
        478,
        20,
    )
    assert (geometry.lane_grid, geometry.lane_size, geometry.lane_pitch) == (4, 114, 120)
    assert geometry.lane_origins == tuple(
        (top, left)
        for top in (2, 122, 242, 362)
        for left in (2, 122, 242, 362)
    )
    assert (geometry.parallel_expert_size, geometry.parallel_expert_pitch) == (54, 60)
    assert geometry.parallel_expert_origins == ((0, 0), (0, 60), (60, 0), (60, 60))
    assert (geometry.serial_expert_size, geometry.serial_expert_pitch) == (109, 123)
    assert geometry.serial_expert_origins == (
        (123, 123),
        (123, 246),
        (246, 123),
        (246, 246),
    )


def test_formal_temporal9_geometry_fills_same_active_field_compactly() -> None:
    geometry = Geometry(
        lane_grid=3,
        lane_size=156,
        lane_pitch=160,
        lane_offset=1,
        parallel_expert_size=77,
        parallel_expert_pitch=79,
    )
    geometry.validate(formal=True)
    assert geometry.lane_origins[0] == (1, 1)
    assert geometry.lane_origins[-1] == (321, 321)
    assert geometry.lane_origins[-1][0] + geometry.lane_size == 477
    assert geometry.parallel_expert_pitch - geometry.parallel_expert_size == 2


def test_formal_temporal36_geometry_fills_same_active_field_compactly() -> None:
    geometry = Geometry(
        lane_grid=6,
        lane_size=77,
        lane_pitch=79,
        lane_offset=3,
        parallel_expert_size=37,
        parallel_expert_pitch=40,
    )
    geometry.validate(formal=True)
    assert len(geometry.lane_origins) == 36
    assert geometry.lane_origins[0] == (3, 3)
    assert geometry.lane_origins[-1] == (398, 398)
    assert geometry.lane_origins[-1][0] + geometry.lane_size == 475
    assert geometry.parallel_expert_pitch - geometry.parallel_expert_size == 3


def test_spatial_and_temporal_prompts_are_exact_and_distinct(tmp_path: Path) -> None:
    spatial = _small_settings(tmp_path, target_name="spatial")
    temporal = _small_settings(tmp_path, target_name="temporal")
    assert spatial.prompt == (
        "Please evaluate the spatial quality of this video and rate it using one "
        "of the following five levels: Excellent, Good, Fair, Poor, or Bad."
    )
    assert temporal.prompt == (
        "Please evaluate the temporal quality of this video and rate it using one "
        "of the following five levels: Excellent, Good, Fair, Poor, or Bad."
    )
    assert spatial.prompt != temporal.prompt
    with pytest.raises(ValueError, match="exact target-specific"):
        spatial.prompt = temporal.prompt
        spatial.validate()


@pytest.mark.parametrize("target_name", ("spatial", "temporal"))
def test_small_optical_on_and_same_checkpoint_bypass_shapes(
    tmp_path: Path, target_name: str
) -> None:
    torch.manual_seed(13)
    model = LGVQSingleMetricOEO16(_small_settings(tmp_path, target_name=target_name)).eval()
    inputs = _inputs(frame_count=model.settings.frame_count)
    with torch.no_grad():
        optical_on = model(*inputs, optical_enabled=True)
        optical_off = model(*inputs, optical_enabled=False)
    assert optical_on["prediction"].shape == optical_off["prediction"].shape == (2,)
    assert optical_on["routing"]["vision"]["weights"].shape == (
        2,
        model.settings.frame_count,
        4,
    )
    assert optical_on["routing"]["language"]["weights"].shape == (2, 4)
    assert optical_off["routing"] == {}
    assert optical_on["optical_enabled"] is True
    assert optical_off["optical_enabled"] is False


def test_text_quality_feature_phase_and_router_receive_gradients(tmp_path: Path) -> None:
    torch.manual_seed(17)
    model = LGVQSingleMetricOEO16(_small_settings(tmp_path)).train()
    vision, quality, language, mask = _inputs(frame_count=model.settings.frame_count)
    quality.requires_grad_()
    language.requires_grad_()
    result = model(vision, quality, language, mask, optical_enabled=True)
    loss = (
        result["normalized_prediction"].sum()
        + 0.05 * result["optical_alignment_loss"]
        + 0.05 * result["router_balance_loss"]
        + 0.01 * result["router_capture_loss"]
    )
    loss.backward()

    gradients = {
        "quality_input": quality.grad,
        "text_input": language.grad,
        "vision_feature_phase": model.parallel_optics.raw_expert_phase.grad,
        "language_feature_phase": model.serial_optics.raw_global_phase.grad,
        "vision_router_phase": model.parallel_router.raw_router_phase.grad,
        "language_router_phase": model.serial_router.raw_router_phase.grad,
    }
    for name, gradient in gradients.items():
        assert gradient is not None, f"{name} has no gradient"
        assert bool(torch.isfinite(gradient).all()), f"{name} gradient is non-finite"
        assert float(gradient.abs().sum()) > 0.0, f"{name} gradient is identically zero"


def test_student_contains_no_attention_or_transformer_module(tmp_path: Path) -> None:
    model = LGVQSingleMetricOEO16(_small_settings(tmp_path))
    forbidden = [
        module.__class__.__name__
        for module in model.modules()
        if "attention" in module.__class__.__name__.lower()
        or "transformer" in module.__class__.__name__.lower()
    ]
    assert forbidden == []


def test_strict_two_branch_profile_has_only_qwen_shared_input_and_two_routes(
    tmp_path: Path,
) -> None:
    settings = replace(
        _small_settings(tmp_path),
        strict_two_branch=True,
        quality_branch_enabled=False,
        qwen_gate_enabled=False,
        electronic_route_variant="residual_conv",
        electronic_route_depth=2,
        phase_snapshot_interval_epochs=0,
    )
    settings.validate()
    model = LGVQSingleMetricOEO16(settings).eval()
    assert model.quality_adapter is None
    assert model.vgg_correction is None
    assert model.frame_stem is None
    assert model.late_input_correction is None
    assert not any(
        key.startswith(("quality_adapter.", "vgg_correction.", "frame_stem."))
        for key in model.state_dict()
    )
    assert not any(
        token in module.__class__.__name__.lower()
        for module in model.modules()
        for token in ("attention", "transformer", "lstm", "gru")
    )
    inputs = list(_inputs(frame_count=4))
    with torch.no_grad():
        first = model(*inputs, optical_enabled=False)["prediction"]
        inputs[1] = torch.randn_like(inputs[1]) * 1000.0
        second = model(*inputs, optical_enabled=False)["prediction"]
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert "strict2branch_v1" in settings.architecture_label


def test_strict_two_branch_contract_rejects_hidden_vgg_side_input(
    tmp_path: Path,
) -> None:
    settings = replace(
        _small_settings(tmp_path),
        strict_two_branch=True,
        quality_branch_enabled=False,
        qwen_gate_enabled=False,
        electronic_route_variant="residual_conv",
        vgg_feature_cache_path=tmp_path / "forbidden.pt",
    )
    with pytest.raises(ValueError, match="strict two-branch"):
        settings.validate()


def test_strict_two_branch_allows_quality_only_inside_electronic_route(
    tmp_path: Path,
) -> None:
    settings = replace(
        _small_settings(tmp_path),
        strict_two_branch=True,
        quality_branch_enabled=False,
        quality_feature_cache_path=tmp_path / "quality_cache.pt",
        quality_input_width=192,
        qwen_gate_enabled=False,
        electronic_route_variant="residual_conv",
        electronic_route_depth=2,
        electronic_quality_residual_enabled=True,
        electronic_quality_residual_initial=0.70,
        phase_snapshot_interval_epochs=0,
    )
    settings.validate()
    model = LGVQSingleMetricOEO16(settings).eval()
    assert model.quality_adapter is None
    assert model.vgg_correction is None
    assert model.electronic_quality_norm is not None
    assert not any(
        token in module.__class__.__name__.lower()
        for module in model.modules()
        for token in ("attention", "transformer", "lstm", "gru")
    )
    generator = torch.Generator().manual_seed(704)
    inputs = [
        torch.randn(2, 4, 49, 1024, generator=generator),
        torch.randn(2, 4, 49, 192, generator=generator),
        torch.randn(2, 4, 2048, generator=generator),
        torch.ones(2, 4, dtype=torch.bool),
    ]
    with torch.no_grad():
        first = model(*inputs, optical_enabled=False)["prediction"]
        inputs[1] = inputs[1] + 3.0 * torch.randn(
            inputs[1].shape, generator=generator
        )
        second = model(*inputs, optical_enabled=False)["prediction"]
    assert not torch.equal(first, second)


def test_strict_e1_quality_refiner_is_zero_start_and_has_no_bypass(
    tmp_path: Path,
) -> None:
    base_settings = replace(
        _small_settings(tmp_path),
        strict_two_branch=True,
        quality_branch_enabled=False,
        quality_feature_cache_path=tmp_path / "quality_cache.pt",
        quality_input_width=192,
        qwen_gate_enabled=False,
        electronic_route_variant="residual_conv",
        electronic_route_depth=2,
        electronic_quality_residual_enabled=True,
        quality_refiner_enabled=False,
        phase_snapshot_interval_epochs=0,
    )
    base_settings.validate()
    base = LGVQSingleMetricOEO16(base_settings).eval()
    checkpoint = tmp_path / "strict_e1_base.pt"
    torch.save({"state_dict": base.state_dict()}, checkpoint)

    refined_settings = replace(
        base_settings,
        quality_refiner_enabled=True,
        initialization_checkpoint=checkpoint,
    )
    refined_settings.validate()
    refined = LGVQSingleMetricOEO16(refined_settings).eval()
    _load_compatible_initialization(refined, refined_settings)
    assert refined.quality_adapter is None
    assert not any(
        token in module.__class__.__name__.lower()
        for module in refined.modules()
        for token in ("attention", "transformer", "lstm", "gru")
    )
    generator = torch.Generator().manual_seed(706)
    inputs = (
        torch.randn(2, 4, 49, 1024, generator=generator),
        torch.randn(2, 4, 49, 192, generator=generator),
        torch.randn(2, 4, 2048, generator=generator),
        torch.ones(2, 4, dtype=torch.bool),
    )
    with torch.no_grad():
        expected = base(*inputs, optical_enabled=False)["prediction"]
        actual = refined(*inputs, optical_enabled=False)["prediction"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_residual_electronic_route_warm_starts_legacy_stem(
    tmp_path: Path,
) -> None:
    source_settings = _small_settings(tmp_path)
    torch.manual_seed(705)
    source = LGVQSingleMetricOEO16(source_settings).eval()
    checkpoint = tmp_path / "legacy_route.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    destination_settings = replace(
        source_settings,
        electronic_route_variant="residual_conv",
        electronic_route_depth=2,
        initialization_checkpoint=checkpoint,
    )
    destination_settings.validate()
    destination = LGVQSingleMetricOEO16(destination_settings).eval()
    report = _load_compatible_initialization(destination, destination_settings)
    assert report["used"] is True
    for route_name in ("vision_routes", "language_routes"):
        for route_index in (0, 1):
            for suffix in (
                "norm.weight",
                "norm.bias",
                "depthwise.weight",
                "pointwise.weight",
                "pointwise.bias",
            ):
                name = f"{route_name}.{route_index}.{suffix}"
                assert torch.equal(source.state_dict()[name], destination.state_dict()[name])


def test_appended_electronic_residual_block_is_exact_identity(
    tmp_path: Path,
) -> None:
    source_settings = replace(
        _small_settings(tmp_path),
        electronic_route_variant="residual_conv",
        electronic_route_depth=2,
    )
    torch.manual_seed(708)
    source = LGVQSingleMetricOEO16(source_settings).eval()
    checkpoint = tmp_path / "depth2_source.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    destination_settings = replace(
        source_settings,
        electronic_route_depth=3,
        initialization_checkpoint=checkpoint,
    )
    destination = LGVQSingleMetricOEO16(destination_settings).eval()
    _load_compatible_initialization(destination, destination_settings)
    inputs = _inputs(frame_count=4)
    with torch.no_grad():
        expected = source(*inputs, optical_enabled=False)["prediction"]
        actual = destination(*inputs, optical_enabled=False)["prediction"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_zero_start_skip_strengthens_same_electronic_route_without_new_branch(
    tmp_path: Path,
) -> None:
    source_settings = replace(
        _small_settings(tmp_path),
        electronic_route_variant="residual_conv",
        electronic_route_depth=2,
    )
    torch.manual_seed(709)
    source = LGVQSingleMetricOEO16(source_settings).eval()
    checkpoint = tmp_path / "no_skip_source.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    destination_settings = replace(
        source_settings,
        initialization_checkpoint=checkpoint,
        electronic_skip_enabled=True,
        electronic_skip_initial=0.0,
        electronic_skip_max=0.75,
    )
    destination = LGVQSingleMetricOEO16(destination_settings).eval()
    _load_compatible_initialization(destination, destination_settings)
    inputs = _inputs(frame_count=4)
    with torch.no_grad():
        expected = source(*inputs, optical_enabled=False)["prediction"]
        actual = destination(*inputs, optical_enabled=False)["prediction"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert all(route.skip is not None for route in destination.vision_routes)
    assert all(route.skip is not None for route in destination.language_routes)


def test_feature_phase_reset_starts_at_raw_zero_and_keeps_router_phase(
    tmp_path: Path,
) -> None:
    source_settings = _small_settings(tmp_path)
    torch.manual_seed(711)
    source = LGVQSingleMetricOEO16(source_settings).eval()
    checkpoint = tmp_path / "trained_phase_source.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)

    destination_settings = replace(
        source_settings,
        initialization_checkpoint=checkpoint,
        reset_feature_phase_on_initialization=True,
        phase_init_std=0.0,
    )
    destination = LGVQSingleMetricOEO16(destination_settings).eval()
    report = _load_compatible_initialization(destination, destination_settings)

    feature_names = (
        "parallel_optics.raw_expert_phase",
        "parallel_optics.raw_global_phase",
        "serial_optics.raw_expert_phase",
        "serial_optics.raw_global_phase",
    )
    for name in feature_names:
        raw = destination.state_dict()[name]
        assert torch.count_nonzero(raw) == 0
        torch.testing.assert_close(_phase(raw), torch.full_like(raw, torch.pi))
        assert name in report["skipped_by_policy"]
    for name in (
        "parallel_router.raw_router_phase",
        "serial_router.raw_router_phase",
    ):
        assert torch.equal(source.state_dict()[name], destination.state_dict()[name])


def test_spatial_grid_readout_preserves_four_frame_contract(tmp_path: Path) -> None:
    settings = _small_settings(tmp_path)
    settings.spatial_readout_mode = "spatial_grid"
    settings.quality_adapter_mode = "spatial_conv"
    settings.quality_gate_initial = 0.70
    settings.validate()
    model = LGVQSingleMetricOEO16(settings).eval()
    with torch.no_grad():
        result = model(
            *_inputs(frame_count=4),
            optical_enabled=False,
        )
    assert result["prediction"].shape == (2,)
    assert settings.architecture_label.endswith("_spatialgrid_v1_qualityconv_v1")


def test_spatial_grid_image_focus_has_exact_zero_start_and_gradient(
    tmp_path: Path,
) -> None:
    source_settings = _small_settings(tmp_path)
    source_settings.spatial_readout_mode = "spatial_grid"
    source = LGVQSingleMetricOEO16(source_settings).eval()
    checkpoint = tmp_path / "grid_source.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    destination_settings = replace(
        source_settings,
        initialization_checkpoint=checkpoint,
        spatial_readout_image_focus_max=1.0,
    )
    destination_settings.validate()
    destination = LGVQSingleMetricOEO16(destination_settings).eval()
    _load_compatible_initialization(destination, destination_settings)
    vision = torch.randn(2, 4, 49, destination_settings.model_width)
    language = torch.randn(2, 10, destination_settings.model_width)
    mask = torch.ones(2, 10, dtype=torch.bool)
    expected = source.readout(vision, language, mask)
    actual = destination.readout(vision, language, mask)
    assert torch.equal(expected, actual)
    actual.sum().backward()
    assert destination.readout.raw_image_focus.grad is not None


def test_weighted_level_readout_is_exact_warm_start_and_trains_levels(
    tmp_path: Path,
) -> None:
    source_settings = _small_settings(tmp_path)
    source_settings.spatial_readout_mode = "spatial_grid"
    torch.manual_seed(811)
    source = LGVQSingleMetricOEO16(source_settings).eval()
    checkpoint = tmp_path / "weighted_source.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    destination_settings = replace(
        source_settings,
        spatial_readout_mode="spatial_weighted_level_residual",
        spatial_residual_max=0.5,
        level_distribution_weight=0.2,
        initialization_checkpoint=checkpoint,
        trainable_scope="residual_only",
    )
    destination_settings.validate()
    destination = LGVQSingleMetricOEO16(destination_settings).eval()
    _load_compatible_initialization(destination, destination_settings)
    vision = torch.randn(2, 4, 49, destination_settings.model_width)
    language = torch.randn(2, 6, destination_settings.model_width)
    mask = torch.ones(2, 6, dtype=torch.bool)
    expected = source.readout(vision, language, mask)
    actual = destination.readout(vision, language, mask)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-7)
    assert destination.readout.last_level_logits.shape == (2, 5)
    actual.sum().backward()
    assert destination.readout.residual_output[-1].weight.grad is not None


def test_late_input_correction_is_zero_start_bounded_and_scope_is_strict(
    tmp_path: Path,
) -> None:
    source_settings = _small_settings(tmp_path)
    torch.manual_seed(303)
    source = LGVQSingleMetricOEO16(source_settings).eval()
    checkpoint = tmp_path / "late_source.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    destination_settings = replace(
        source_settings,
        late_input_correction_enabled=True,
        late_input_correction_max=0.25,
        initialization_checkpoint=checkpoint,
        trainable_scope="late_input_correction_only",
    )
    destination_settings.validate()
    destination = LGVQSingleMetricOEO16(destination_settings).eval()
    _load_compatible_initialization(destination, destination_settings)
    scope = _apply_trainable_scope(destination, destination_settings)
    assert scope["trainable_names"]
    assert all(
        name.startswith("late_input_correction.")
        for name in scope["trainable_names"]
    )
    inputs = _inputs(frame_count=4)
    with torch.no_grad():
        expected = source(*inputs, optical_enabled=True)["normalized_prediction"]
        result = destination(*inputs, optical_enabled=True)
    # Adding an exactly zero correction can still select a different FFT
    # memory/stride path.  Require numerical identity at float32 precision
    # instead of brittle bitwise equality.
    torch.testing.assert_close(
        expected,
        result["normalized_prediction"],
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert torch.count_nonzero(result["late_input_correction"]) == 0
    assert "lateinputcorr025_v1" in destination_settings.architecture_label


def test_plain_vgg_correction_is_zero_start_pre_optical_and_scope_is_strict(
    tmp_path: Path,
) -> None:
    source_settings = replace(_small_settings(tmp_path), token_grid=14)
    torch.manual_seed(401)
    source = LGVQSingleMetricOEO16(source_settings).eval()
    checkpoint = tmp_path / "vgg_source.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    destination_settings = replace(
        source_settings,
        vgg_feature_cache_path=tmp_path / "declared_cache.pt",
        vgg_correction_max=0.5,
        vgg_correction_mode="local",
        initialization_checkpoint=checkpoint,
        trainable_scope="vgg_correction_only",
    )
    destination_settings.validate()
    destination = LGVQSingleMetricOEO16(destination_settings).eval()
    _load_compatible_initialization(destination, destination_settings)
    scope = _apply_trainable_scope(destination, destination_settings)
    assert scope["trainable_names"]
    assert all(name.startswith("vgg_correction.") for name in scope["trainable_names"])

    generator = torch.Generator().manual_seed(402)
    inputs = (
        torch.randn(2, 4, 196, 1024, generator=generator),
        torch.randn(2, 4, 196, 14, generator=generator),
        torch.randn(2, 4, 2048, generator=generator),
        torch.ones(2, 4, dtype=torch.bool),
    )
    vgg = torch.randn(2, 4, 196, 512, generator=generator)
    with torch.no_grad():
        expected = source(*inputs, optical_enabled=True)["normalized_prediction"]
        result = destination(
            *inputs, vgg_tokens=vgg, optical_enabled=True
        )
    # The zero-start branch is functionally identical; tolerate float32 FFT
    # roundoff caused by the extra zero-addition memory/stride path.
    torch.testing.assert_close(
        expected,
        result["normalized_prediction"],
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert torch.count_nonzero(result["vgg_correction_rms"]) == 0
    assert "plainvgg16corr50_local_v1" in destination_settings.architecture_label


@pytest.mark.parametrize(
    "residual_mode", ("spatial_grid_residual", "spatial_pyramid_residual")
)
def test_residual_warm_start_loads_every_shared_tensor_and_preserves_output(
    tmp_path: Path, residual_mode: str,
) -> None:
    source_settings = _small_settings(tmp_path)
    source_settings.spatial_readout_mode = "spatial_grid"
    torch.manual_seed(101)
    source = LGVQSingleMetricOEO16(source_settings).eval()
    checkpoint = tmp_path / "source.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)

    destination_settings = replace(
        source_settings,
        spatial_readout_mode=residual_mode,
        initialization_checkpoint=checkpoint,
        trainable_scope="residual_only",
    )
    torch.manual_seed(202)
    destination = LGVQSingleMetricOEO16(destination_settings).eval()
    report = _load_compatible_initialization(destination, destination_settings)

    assert report["used"] is True
    assert report["loaded_tensors"] == len(source.state_dict())
    for name, value in source.state_dict().items():
        assert torch.equal(value, destination.state_dict()[name]), name

    vision = torch.randn(2, 4, 49, destination_settings.model_width)
    language = torch.randn(2, 6, destination_settings.model_width)
    mask = torch.ones(2, 6, dtype=torch.bool)
    with torch.no_grad():
        expected = source.readout(vision, language, mask)
        actual = destination.readout(vision, language, mask)
    assert torch.equal(expected, actual)


def test_eval_dc_component_is_coherent_and_keeps_phase_gradient(tmp_path: Path) -> None:
    settings = _small_settings(tmp_path)
    settings.unmodulated_power_fraction_min = 0.20
    settings.unmodulated_power_fraction_max = 0.35
    settings.unmodulated_power_fraction_eval = 0.20
    raw = torch.randn(8, 8, requires_grad=True)

    actual = _phase_modulation(raw, settings=settings, training=False)
    expected = (
        (1.0 - 0.20) ** 0.5 * torch.exp(1j * _phase(raw))
        + 0.20**0.5
    )
    assert torch.allclose(actual, expected)

    actual.abs().square().mean().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert float(raw.grad.abs().sum()) > 0.0


def test_phase_quantization_matches_hardware_levels_and_keeps_gradient(
    tmp_path: Path,
) -> None:
    settings = _small_settings(tmp_path)
    settings.phase_quantization_levels = 256
    settings.unmodulated_power_fraction_min = 0.0
    settings.unmodulated_power_fraction_max = 0.0
    settings.unmodulated_power_fraction_eval = 0.0
    raw_leaf = torch.linspace(-3.0, 3.0, 64, requires_grad=True)
    raw = raw_leaf.reshape(8, 8)
    modulation = _phase_modulation(raw, settings=settings, training=True)
    phase_units = torch.remainder(torch.angle(modulation), 2.0 * torch.pi)
    phase_units = phase_units / (2.0 * torch.pi / 255.0)
    assert torch.allclose(phase_units, phase_units.round(), atol=2.0e-4)
    modulation.real.mean().backward()
    assert raw_leaf.grad is not None
    assert float(raw_leaf.grad.abs().sum()) > 0.0


def test_router_diversity_loss_rewards_sample_dependent_choices() -> None:
    fixed = torch.tensor([[0.70, 0.20, 0.08, 0.02]]).repeat(4, 1)
    diverse = torch.tensor(
        [
            [0.70, 0.20, 0.08, 0.02],
            [0.02, 0.70, 0.20, 0.08],
            [0.08, 0.02, 0.70, 0.20],
            [0.20, 0.08, 0.02, 0.70],
        ]
    )
    fixed_selected = fixed >= torch.topk(fixed, 2, -1).values[:, -1:]
    diverse_selected = diverse >= torch.topk(diverse, 2, -1).values[:, -1:]
    fixed_loss = _routing_statistics(fixed, fixed_selected)["diversity_loss"]
    diverse_loss = _routing_statistics(diverse, diverse_selected)["diversity_loss"]
    assert float(diverse_loss) < float(fixed_loss)


def test_three_stage_schedule_freezes_then_refines_electronics(tmp_path: Path) -> None:
    settings = replace(
        _small_settings(tmp_path),
        epochs=12,
        phase_warmup_epochs=3,
        late_refine_start_epoch=9,
        late_refine_electronic_lr_factor=0.1,
        late_refine_readout_lr_factor=0.2,
    )
    settings.validate()
    assert _training_stage_factors(settings, 2)[0] == "optical_phase_warmup"
    assert _training_stage_factors(settings, 5)[0] == "joint"
    name, factors = _training_stage_factors(settings, 10)
    assert name == "late_refine"
    assert factors["electronic"] == pytest.approx(0.1)
    assert factors["readout"] == pytest.approx(0.2)


def test_single_target_training_uses_one_dimensional_target_statistics(
    tmp_path: Path,
) -> None:
    settings = _small_settings(tmp_path)
    model = LGVQSingleMetricOEO16(settings)
    generator = torch.Generator().manual_seed(23)
    payload = {
        "vision_tokens": torch.randn(4, 4, 49, 1024, generator=generator).half(),
        "quality_tokens": torch.randn(4, 4, 49, 14, generator=generator).half(),
        "language_tokens": torch.randn(1, 4, 2048, generator=generator).half(),
        "language_mask": torch.ones(1, 4, dtype=torch.bool),
        "input_ids": torch.arange(4).view(1, 4),
        # Deliberately one-dimensional: this is the single-target contract.
        "targets": torch.tensor([10.0, 14.0, 20.0, 30.0]),
        "sample_ids": ["train_a", "train_b", "test_a", "test_b"],
        "video_paths": ["a.mp4", "b.mp4", "c.mp4", "d.mp4"],
        "splits": ["train", "train", "test", "test"],
        "target_name": "spatial",
    }
    summary = train(model, payload, settings, torch.device("cpu"))
    # Epoch 0 is intentionally evaluated and retained when the single update
    # does not improve SRCC; a warm start must never be silently discarded.
    assert summary["best_epoch"] in (0, 1)
    assert float(model.target_mean) == pytest.approx(12.0)
    assert float(model.target_std) == pytest.approx(2.0)
    assert (settings.output_dir / "best_observed_test_checkpoint.pt").is_file()


def test_training_can_select_a_within_epoch_optimizer_step(tmp_path: Path) -> None:
    settings = replace(
        _small_settings(tmp_path),
        batch_size=1,
        test_interval_steps=1,
    )
    model = LGVQSingleMetricOEO16(settings)
    generator = torch.Generator().manual_seed(29)
    payload = {
        "vision_tokens": torch.randn(4, 4, 49, 1024, generator=generator).half(),
        "quality_tokens": torch.randn(4, 4, 49, 14, generator=generator).half(),
        "language_tokens": torch.randn(1, 4, 2048, generator=generator).half(),
        "language_mask": torch.ones(1, 4, dtype=torch.bool),
        "input_ids": torch.arange(4).view(1, 4),
        "targets": torch.tensor([10.0, 14.0, 20.0, 30.0]),
        "sample_ids": ["train_a", "train_b", "test_a", "test_b"],
        "video_paths": ["a.mp4", "b.mp4", "c.mp4", "d.mp4"],
        "splits": ["train", "train", "test", "test"],
        "target_name": "spatial",
    }
    summary = train(model, payload, settings, torch.device("cpu"))
    history = json.loads(
        (settings.output_dir / "train_history.json").read_text(encoding="utf-8")
    )
    assert len(history[1]["within_epoch_test_evaluations"]) == 1
    assert history[1]["within_epoch_test_evaluations"][0]["optimizer_step"] == 1
    assert summary["periodic_test_interval_optimizer_steps"] == 1
    assert summary["best_optimizer_step"] in (0, 1, 2)
