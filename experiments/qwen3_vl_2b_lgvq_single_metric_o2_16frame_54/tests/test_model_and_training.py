from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ..modeling import (
    LGVQSingleMetricOEO16,
    TrainableQualityFrameStem,
    _phase,
    _phase_modulation,
)
from ..run import _apply_trainable_scope, _load_compatible_initialization
from ..settings import TARGET_PROMPTS, ExperimentSettings, Geometry
from ..training import soft_spearman_loss, train


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
