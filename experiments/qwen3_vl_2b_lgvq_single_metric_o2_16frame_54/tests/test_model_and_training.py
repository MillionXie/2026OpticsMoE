from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ..modeling import LGVQSingleMetricOEO16, _phase, _phase_modulation
from ..settings import TARGET_PROMPTS, ExperimentSettings, Geometry
from ..training import train


def _small_settings(tmp_path: Path, *, target_name: str = "spatial") -> ExperimentSettings:
    """Return a fast, non-overlapping optical geometry for CPU contract tests."""

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
            lane_grid=4,
            lane_size=18,
            lane_pitch=22,
            parallel_expert_size=8,
            parallel_expert_pitch=10,
            serial_expert_size=24,
            serial_expert_pitch=32,
        ),
        maximum_language_tokens=24,
        detector_projection_size=8,
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


def _inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(11)
    return (
        torch.randn(batch, 16, 49, 1024, generator=generator),
        torch.randn(batch, 16, 49, 14, generator=generator),
        torch.randn(batch, 4, 2048, generator=generator),
        torch.ones(batch, 4, dtype=torch.bool),
    )


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
    inputs = _inputs()
    with torch.no_grad():
        optical_on = model(*inputs, optical_enabled=True)
        optical_off = model(*inputs, optical_enabled=False)
    assert optical_on["prediction"].shape == optical_off["prediction"].shape == (2,)
    assert optical_on["routing"]["vision"]["weights"].shape == (2, 16, 4)
    assert optical_on["routing"]["language"]["weights"].shape == (2, 4)
    assert optical_off["routing"] == {}
    assert optical_on["optical_enabled"] is True
    assert optical_off["optical_enabled"] is False


def test_text_quality_feature_phase_and_router_receive_gradients(tmp_path: Path) -> None:
    torch.manual_seed(17)
    model = LGVQSingleMetricOEO16(_small_settings(tmp_path)).train()
    vision, quality, language, mask = _inputs()
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
        "vision_tokens": torch.randn(4, 16, 49, 1024, generator=generator).half(),
        "quality_tokens": torch.randn(4, 16, 49, 14, generator=generator).half(),
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
    assert summary["best_epoch"] == 1
    assert float(model.target_mean) == pytest.approx(12.0)
    assert float(model.target_std) == pytest.approx(2.0)
    assert (settings.output_dir / "best_observed_test_checkpoint.pt").is_file()
