from __future__ import annotations

from pathlib import Path

import torch
import pytest

from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.stem import (
    STEM_FORMAT,
)
from experiments.qwen3_vl_patch_stem_8stage_slim_mixer_imagenet_backbone.model import (
    QwenStemSlimMixerOpticalImageNetBackbone,
    grid_to_qwen_tokens,
)

from ..model import (
    AxisAngularSpectrumPropagator,
    QwenStemSeparableOpticalImageNetBackbone,
    qwen_field_to_row_major,
    row_major_field_to_qwen,
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
        "token_axis_propagation_distance_m": 0.05,
        "channel_axis_propagation_distance_m": 0.05,
    }


def test_qwen_physical_layout_round_trip_and_true_row_major_order() -> None:
    grid = torch.arange(196, dtype=torch.float32).view(1, 1, 14, 14)
    qwen_tokens = grid_to_qwen_tokens(grid).expand(1, 196, 224)
    field = torch.zeros(1, 1, 224, 224)
    field[:, :, :196] = qwen_tokens.view(1, 1, 196, 224)
    field[:, :, 196:] = -7.0
    physical = qwen_field_to_row_major(field)
    assert torch.equal(physical[0, 0, :196, 0], torch.arange(196).float())
    assert torch.equal(physical[:, :, 196:], field[:, :, 196:])
    assert torch.equal(row_major_field_to_qwen(physical), field)


def test_token_axis_does_not_mix_feature_columns() -> None:
    propagator = AxisAngularSpectrumPropagator(
        224, 5.32e-7, 1.6e-5, 0.05, "token"
    )
    field = torch.zeros(1, 1, 224, 224, dtype=torch.complex64)
    field[0, 0, 112, 17] = 1.0
    output = propagator(field)
    outside = output.clone()
    outside[..., 17] = 0.0
    assert float(outside.abs().max()) < 1.0e-5
    assert float(output[..., 17].abs().max()) > 0.0


def test_channel_axis_does_not_mix_token_rows() -> None:
    propagator = AxisAngularSpectrumPropagator(
        224, 5.32e-7, 1.6e-5, 0.05, "channel"
    )
    field = torch.zeros(1, 1, 224, 224, dtype=torch.complex64)
    field[0, 0, 112, 17] = 1.0
    output = propagator(field)
    outside = output.clone()
    outside[..., 112, :] = 0.0
    assert float(outside.abs().max()) < 1.0e-5
    assert float(output[..., 112, :].abs().max()) > 0.0


def test_axis_propagator_matches_direct_one_dimensional_fft() -> None:
    torch.manual_seed(7)
    field = torch.randn(1, 1, 224, 224, dtype=torch.complex64)
    token = AxisAngularSpectrumPropagator(
        224, 5.32e-7, 1.6e-5, 0.05, "token"
    )
    channel = AxisAngularSpectrumPropagator(
        224, 5.32e-7, 1.6e-5, 0.05, "channel"
    )
    token_reference = torch.fft.ifft(
        torch.fft.fft(field, dim=-2, norm="ortho")
        * token.transfer_function[:, 0].view(1, 1, 224, 1),
        dim=-2,
        norm="ortho",
    )
    channel_reference = torch.fft.ifft(
        torch.fft.fft(field, dim=-1, norm="ortho")
        * channel.transfer_function[0].view(1, 1, 1, 224),
        dim=-1,
        norm="ortho",
    )
    torch.testing.assert_close(token(field), token_reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(channel(field), channel_reference, rtol=2e-5, atol=2e-5)


def test_separable_schedule_and_parameter_budget(tmp_path: Path) -> None:
    model = QwenStemSeparableOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), model_config()
    )
    report = model.parameter_report()
    assert [stage.optical_axis for stage in model.stages] == [
        "token",
        "channel",
    ] * 4
    assert report["optical_mixer_variant"] == "separable_token_channel_axis"
    assert report["qwen_token_order_corrected_inside_token_optics"] is True
    assert report["optical_phase_parameters"] == 8 * 3 * 224 * 224
    assert report["residual_electronic_parameters"] == 733_472
    assert report["optical_fraction_of_backbone_trainable"] >= 0.50
    assert report["adds_trainable_parameters_over_p09"] == 0


def test_p11_trainable_initialization_matches_p09(tmp_path: Path) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    torch.manual_seed(123)
    p11 = QwenStemSeparableOpticalImageNetBackbone(stem, model_config())
    torch.manual_seed(123)
    p09 = QwenStemSlimMixerOpticalImageNetBackbone(stem, model_config())
    p11_parameters = dict(p11.named_parameters())
    p09_parameters = dict(p09.named_parameters())
    assert p11_parameters.keys() == p09_parameters.keys()
    for name in p09_parameters:
        torch.testing.assert_close(p11_parameters[name], p09_parameters[name])


def test_token_axis_fixed_feedback_keeps_phase_gradients(tmp_path: Path) -> None:
    model = QwenStemSeparableOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), model_config()
    )
    stage = model.stages[0]
    stage.set_feedback("fa_random", torch.zeros_like(stage.feedback_phase))
    amplitude = torch.rand(1, 3, 224, 224)
    stage(amplitude).mean().backward()
    assert stage.raw_phase.grad is not None
    assert bool(torch.isfinite(stage.raw_phase.grad).all())
    assert float(stage.raw_phase.grad.norm()) > 0.0


def test_token_axis_fixed_feedback_matches_bp_for_matching_connector(
    tmp_path: Path,
) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    torch.manual_seed(17)
    bp = QwenStemSeparableOpticalImageNetBackbone(stem, model_config()).stages[0]
    torch.manual_seed(17)
    fixed = QwenStemSeparableOpticalImageNetBackbone(stem, model_config()).stages[0]
    fixed.load_state_dict(bp.state_dict())
    fixed.set_feedback("fa_pretrained", bp.phase().detach())
    bp.eval()
    fixed.eval()
    bp_input = torch.rand(1, 3, 224, 224, requires_grad=True)
    fixed_input = bp_input.detach().clone().requires_grad_(True)
    bp_output = bp(bp_input)
    fixed_output = fixed(fixed_input)
    torch.testing.assert_close(bp_output, fixed_output, rtol=2e-5, atol=2e-6)
    bp_output.square().mean().backward()
    fixed_output.square().mean().backward()
    torch.testing.assert_close(bp_input.grad, fixed_input.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(
        bp.raw_phase.grad,
        fixed.raw_phase.grad,
        rtol=3e-4,
        atol=3e-5,
    )


def test_p09_strict_load_rejects_p11_checkpoint(tmp_path: Path) -> None:
    stem = fake_stem(tmp_path / "stem.pt")
    p11 = QwenStemSeparableOpticalImageNetBackbone(stem, model_config())
    p09 = QwenStemSlimMixerOpticalImageNetBackbone(stem, model_config())
    with pytest.raises(RuntimeError, match="p11_separable_architecture_signature"):
        p09.load_state_dict(p11.state_dict(), strict=True)
