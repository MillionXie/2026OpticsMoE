from __future__ import annotations

from pathlib import Path

import torch

from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.stem import (
    STEM_FORMAT,
)

from ..model import (
    QwenStemSlimMixerOpticalImageNetBackbone,
    SlimSpatialTokenMixerSkip,
    grid_to_qwen_tokens,
    qwen_tokens_to_grid,
)


def fake_stem(path: Path) -> Path:
    torch.save(
        {
            "format": STEM_FORMAT,
            "conv2d_weight": torch.zeros(1024, 3, 16, 16),
            "conv2d_bias": torch.zeros(1024),
            "position_embedding": torch.randn(196, 1024) * 0.01,
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


def model_config() -> dict[str, object]:
    return {
        "canvas_size": 224,
        "optical_channels": 3,
        "num_stages": 8,
        "token_dim": 224,
        "num_classes": 1000,
        "head_hidden_dim": 448,
        "phase_init_std": 0.10,
        "optical_gate_init": 0.60,
        "optical_gate_min": 0.50,
        "mixer_width": 96,
        "mixer_expansion": 2.0,
        "mixer_kernel_size": 3,
        "mixer_dropout": 0.10,
        "mixer_spatial_gate_init": 0.10,
        "mixer_channel_gate_init": 0.10,
        "residual_scale_init": 0.10,
        "residual_scale_max": 0.25,
    }


def test_qwen_token_grid_round_trip() -> None:
    tokens = torch.arange(2 * 196 * 7, dtype=torch.float32).view(2, 196, 7)
    restored = grid_to_qwen_tokens(qwen_tokens_to_grid(tokens))
    assert torch.equal(restored, tokens)


def test_slim_mixer_has_two_gated_residuals_and_identity_initialization() -> None:
    mixer = SlimSpatialTokenMixerSkip(
        field_size=224,
        token_count=196,
        token_dim=224,
        optical_banks=3,
        width=96,
        expansion=2.0,
        kernel_size=3,
        dropout=0.10,
        spatial_gate_init=0.10,
        channel_gate_init=0.10,
        output_scale_init=0.10,
        output_scale_max=0.25,
        eps=1.0e-5,
    )
    value = torch.rand(1, 3, 224, 224)
    expected = value / value.square().mean(dim=(-2, -1), keepdim=True).add(1.0e-5).sqrt()
    actual = mixer(value)
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-5)
    assert abs(float(mixer.spatial_gate()) - 0.10) < 1.0e-6
    assert abs(float(mixer.channel_gate()) - 0.10) < 1.0e-6
    assert abs(float(mixer.transform_scale()) - 0.10) < 1.0e-6


def test_p09_budget_and_stage_contract(tmp_path: Path) -> None:
    model = QwenStemSlimMixerOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), model_config()
    )
    report = model.parameter_report()
    assert report["optical_phase_parameters"] == 8 * 3 * 224 * 224
    assert report["mixer_width"] == 96
    assert report["mixer_instances"] == 8
    assert report["residual_electronic_parameters"] <= 2_000_000
    assert report["optical_fraction_of_backbone_trainable"] >= 0.50
    assert report["optical_fraction_of_all_trainable"] < report["optical_fraction_of_backbone_trainable"]
    assert report["minimum_optical_gate"] >= 0.50
    assert report["contains_electronic_transformer"] is False
    assert all(stage.electronic_skip.width == 96 for stage in model.stages)
    assert all(not name.startswith("readout.") for name in model.backbone_state_dict())


def test_forward_backward_reaches_phases_and_mixer(tmp_path: Path) -> None:
    model = QwenStemSlimMixerOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), model_config()
    )
    logits = model(torch.zeros(1, 3, 224, 224))
    logits.square().mean().backward()
    assert logits.shape == (1, 1000)
    assert all(stage.raw_phase.grad is not None for stage in model.stages)
    assert all(stage.electronic_skip.output_adapter.weight.grad is not None for stage in model.stages)
