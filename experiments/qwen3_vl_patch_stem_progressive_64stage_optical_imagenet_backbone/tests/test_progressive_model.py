from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.d2nn_cifar10_high_performance_optical_backbone.optics import (
    rms_normalize,
)
from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.stem import (
    STEM_FORMAT,
)
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
    QwenStemSeparableOpticalImageNetBackbone,
)

from ..migration import migrate_strict_p11_checkpoint, sha256_file
from ..model import (
    QwenStemProgressiveOpticalImageNetBackbone,
    anchor_stage_indices,
)


def fake_stem(path: Path) -> Path:
    generator = torch.Generator().manual_seed(117)
    torch.save(
        {
            "format": STEM_FORMAT,
            "conv2d_weight": torch.zeros(1024, 3, 16, 16),
            "conv2d_bias": torch.zeros(1024),
            "position_embedding": torch.randn(196, 1024, generator=generator)
            * 0.01,
            "metadata": {
                "image_size": 224,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            },
        },
        path,
    )
    return path


def common_config() -> dict[str, object]:
    return {
        "canvas_size": 224,
        "optical_channels": 3,
        "token_dim": 224,
        "num_classes": 1000,
        "head_hidden_dim": 448,
        "phase_init_std": 0.10,
        "optical_gate_init": 0.60,
        "optical_gate_min": 0.50,
        "mixer_width": 96,
        "mixer_expansion": 2.0,
        "mixer_kernel_size": 3,
        "mixer_dropout": 0.0,
        "mixer_spatial_gate_init": 0.10,
        "mixer_channel_gate_init": 0.10,
        "residual_scale_init": 0.10,
        "residual_scale_max": 0.25,
        "token_axis_propagation_distance_m": 0.05,
        "channel_axis_propagation_distance_m": 0.05,
        "seed": 2026,
    }


def p11_export(path: Path, stem: Path) -> tuple[Path, QwenStemSeparableOpticalImageNetBackbone]:
    config = common_config()
    config["num_stages"] = 8
    torch.manual_seed(41)
    source = QwenStemSeparableOpticalImageNetBackbone(stem, config)
    torch.save(
        {
            "backbone": source.backbone_state_dict(),
            "best_epoch": 88,
            "config_digest": "unit-test-config",
            "stem_checkpoint_sha256": source.stem.checkpoint_sha256,
            "model_report": source.parameter_report(),
        },
        path,
    )
    return path, source


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (16, (0, 1, 4, 5, 10, 11, 14, 15)),
        (32, (0, 1, 10, 11, 20, 21, 30, 31)),
        (64, (0, 1, 20, 21, 42, 43, 62, 63)),
        (100, (0, 1, 32, 33, 66, 67, 98, 99)),
    ],
)
def test_anchor_schedule(depth: int, expected: tuple[int, ...]) -> None:
    assert anchor_stage_indices(depth) == expected
    assert [index % 2 for index in expected] == [0, 1] * 4


def test_64stage_parameter_and_electronic_budget(tmp_path: Path) -> None:
    config = common_config()
    config.update({"num_stages": 64, "new_stage_alpha_init": 0.0})
    model = QwenStemProgressiveOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), config
    )
    report = model.parameter_report()

    assert report["num_stages"] == 64
    assert report["optical_phase_parameters"] == 9_633_792
    assert report["unique_width96_mixer_instances"] == 8
    assert report["new_stage_count"] == 56
    assert report["new_stage_identity_skip_parameters"] == 0
    assert report["outer_depth_gate_trainable_parameters"] == 0
    assert report["electronic_backbone_parameters"] == 965_176
    assert report["new_stage_electronic_parameters"] == 56
    assert report["optical_fraction_of_backbone_trainable"] > 0.90
    assert report["depth_alpha"]["all_exact_bypass"] is True
    assert [slot.stage.optical_axis for slot in model.slots] == [
        "token",
        "channel",
    ] * 32


def test_strict_p11_migration_is_function_preserving_at_alpha_zero(
    tmp_path: Path,
) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    source_path, source = p11_export(tmp_path / "p11_backbone.pt", stem)
    config = common_config()
    config.update({"num_stages": 64, "new_stage_alpha_init": 0.0})
    target = QwenStemProgressiveOpticalImageNetBackbone(stem, config)
    manifest = migrate_strict_p11_checkpoint(target, source_path)
    source.eval()
    target.eval()

    generator = torch.Generator().manual_seed(613)
    amplitude = rms_normalize(
        torch.rand(1, 3, 224, 224, generator=generator),
        1.0e-5,
    )
    with torch.no_grad():
        expected = amplitude
        for stage in source.stages:
            expected = stage(expected)
        actual, intermediates = target.forward_field(
            amplitude,
            return_intermediates=True,
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert len(intermediates) == 64
    assert manifest["source_checkpoint_sha256"] == sha256_file(source_path)
    assert manifest["source_phase_sequence_sha256"] == manifest[
        "target_anchor_phase_sequence_sha256"
    ]
    assert manifest["source_imagenet_head_migrated"] is False
    feedback_source = manifest["full_depth_feedback_source"]
    assert feedback_source["connector_count"] == 64
    assert feedback_source["internal_interstage_connector_count"] == 63
    assert len(feedback_source["per_connector_phase_sha256"]) == 64
    assert feedback_source["persistent_in_backbone_state_dict"] is True
    torch.testing.assert_close(
        target.feedback_source_snapshot(),
        target.phase_snapshot(),
        rtol=0.0,
        atol=0.0,
    )
    assert target.depth_alpha_report()["all_exact_bypass"] is True


def test_positive_alpha_gives_every_added_phase_a_nonzero_gradient(
    tmp_path: Path,
) -> None:
    config = common_config()
    config.update(
        {
            "num_stages": 16,
            "new_stage_alpha_init": 0.05,
            "activation_checkpointing": True,
        }
    )
    model = QwenStemProgressiveOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), config
    )
    model.train()
    generator = torch.Generator().manual_seed(997)
    amplitude = rms_normalize(
        torch.rand(1, 3, 224, 224, generator=generator),
        1.0e-5,
    ).requires_grad_(True)
    output, _ = model.forward_field(amplitude)
    spatial_weight = torch.linspace(0.2, 1.0, 224).view(1, 1, 224, 1)
    loss = (output.square() * spatial_weight).mean()
    loss.backward()

    gradients = [slot.stage.raw_phase.grad for slot in model.new_slots()]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None)
    assert all(float(gradient.norm()) > 0.0 for gradient in gradients if gradient is not None)


def test_forced_depth_ramp_and_alpha_state_reload(tmp_path: Path) -> None:
    config = common_config()
    config.update(
        {
            "num_stages": 16,
            "new_stage_alpha_init": 0.0,
            "new_stage_alpha_epsilon": 0.02,
            "new_stage_ramp_epochs": 5,
        }
    )
    stem = fake_stem(tmp_path / "stem.pt")
    model = QwenStemProgressiveOpticalImageNetBackbone(stem, config)
    assert model.apply_depth_ramp(0) == 0.0
    assert model.apply_depth_ramp(1) == pytest.approx(0.02)
    assert model.apply_depth_ramp(3) == pytest.approx(0.51)
    assert model.apply_depth_ramp(5) == 1.0

    state = model.state_dict()
    clone = QwenStemProgressiveOpticalImageNetBackbone(stem, config)
    clone.load_state_dict(state, strict=True)
    assert clone.depth_alpha_report()["all_full_depth"] is True
