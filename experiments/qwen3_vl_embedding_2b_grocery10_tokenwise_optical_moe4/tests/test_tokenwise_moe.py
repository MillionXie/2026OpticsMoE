from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4.optics.tokenwise_moe import (
    PerTokenTopKRouter,
    TokenwiseLayout,
    TokenwiseOpticalMoE,
)
from experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4.modeling import (
    TokenwiseLanguageSurrogate,
)
from experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4.settings import (
    load_settings,
)


EXPERIMENT = Path(__file__).resolve().parents[1]


def fake_settings(second_plane_mode: str = "expert") -> SimpleNamespace:
    return SimpleNamespace(
        token_grid_rows=2,
        token_grid_cols=2,
        token_feature_side=4,
        expert_grid_rows=2,
        expert_grid_cols=2,
        expert_gap=1,
        token_group_gap=1,
        propagation_padding=2,
        num_experts=4,
        top_k=2,
        router_temperature=1.0,
        router_layernorm_enabled=True,
        router_layernorm_affine=False,
        router_noise_std=0.0,
        router_gate_init_std=0.01,
        amplitude_weight_domain="amplitude",
        input_normalization="layernorm",
        input_nonlinearity="softplus",
        input_amplitude_normalization="none",
        share_expert_phase_across_tokens=True,
        phase_parameterization="sigmoid",
        phase_init="zeros",
        phase_init_std=0.0,
        phase_dropout_mode="none",
        phase_dropout_p=0.0,
        phase_dropout_block_size=2,
        phase_dropout_batch_shared=True,
        wavelength_nm=532.0,
        pixel_pitch_um=16.0,
        propagation_distance_m=0.001,
        k_space_enabled=False,
        theta_max_deg=0.65,
        second_plane_mode=second_plane_mode,
        oeo_layernorm_eps=1e-5,
        oeo_elementwise_affine=False,
        oeo_nonlinearity="relu",
        oeo_reapply_routing_weights=True,
        oeo_hard_route_mask=True,
        oeo_preserve_response_amplitude=True,
        oeo_response_gain_min=0.25,
        oeo_response_gain_max=4.0,
        final_layernorm_eps=1e-5,
        final_layernorm_affine=False,
        final_aggregation="routing_weighted_sum",
        residual_enabled=False,
        residual_scale=1.0,
        residual_scale_trainable=False,
        capture_intermediate_fields=True,
        visualization_sample_count=2,
        max_language_tokens=4,
        hidden_size=16,
        language_adapter_layernorm_affine=False,
    )


def test_default_config_panel_is_950_and_adapter_free() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "grocery10_tokenwise_moe4.yaml")
    assert settings.hidden_size == 1024
    assert settings.max_tokens == 196
    assert settings.expert_group_height == settings.expert_group_width == 66
    assert settings.active_height == settings.active_width == 950
    assert settings.canvas_size == 990
    assert settings.num_experts == 4
    assert settings.top_k == 2
    assert settings.k_space_enabled is False
    assert settings.residual_enabled is False
    assert settings.second_plane_mode == "global"


def test_8um_hardware_geometry_keeps_panel_pixels_but_changes_physics() -> None:
    settings = load_settings(
        EXPERIMENT / "configs" / "grocery10_tokenwise_moe4_8um_hardware_geometry.yaml"
    )
    assert settings.active_height == settings.active_width == 950
    assert settings.canvas_size == 990
    assert settings.pixel_pitch_um == 8.0


def test_layout_has_unique_token_expert_apertures() -> None:
    layout = TokenwiseLayout(14, 14, 32, 2, 2, 2, 2, 20)
    indexes = layout.linear_indices()
    assert indexes.shape == (196, 4, 32, 32)
    assert indexes.unique().numel() == indexes.numel()
    assert layout.active_height == 950
    assert layout.canvas_size == 990


def test_router_is_per_token_top2_and_padding_is_excluded() -> None:
    settings = fake_settings()
    router = PerTokenTopKRouter(16, settings)
    tokens = torch.randn(2, 4, 16)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    result = router(tokens, valid)
    assert result["weights"].shape == (2, 4, 4)
    assert torch.equal(result["selected_mask"].sum(-1)[valid], torch.full((5,), 2))
    assert torch.all(result["weights"][~valid] == 0)
    assert torch.allclose(result["weights"].sum(-1)[valid], torch.ones(5), atol=1e-6)
    assert torch.isfinite(result["balance_loss"])


@pytest.mark.parametrize("second_plane_mode", ["global", "expert"])
def test_core_shape_finite_nonnegative_reload_and_all_phase_gradients(
    second_plane_mode: str,
) -> None:
    settings = fake_settings(second_plane_mode)
    core = TokenwiseOpticalMoE(16, settings)
    hidden = torch.randn(5, 16)
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    output = core(hidden, cu)
    assert output.shape == hidden.shape
    assert output.dtype == hidden.dtype
    assert torch.isfinite(output).all()
    assert core.last_input_amplitude is not None
    assert core.last_input_amplitude.min() >= 0
    assert core.last_reload_amplitude is not None
    assert core.last_reload_amplitude.min() >= 0
    loss = output.square().mean()
    loss.backward()
    assert core.first_expert_phase.raw_phase.grad is not None
    assert torch.isfinite(core.first_expert_phase.raw_phase.grad).all()
    assert core.second_phase.raw_phase.grad is not None
    assert torch.isfinite(core.second_phase.raw_phase.grad).all()
    assert core.router.gate.weight.grad is not None


def test_raw_zero_phase_initializes_physical_phase_to_pi() -> None:
    core = TokenwiseOpticalMoE(16, fake_settings())
    assert torch.count_nonzero(core.first_expert_phase.raw_phase) == 0
    assert torch.allclose(
        core.first_expert_phase.phase(),
        torch.full_like(core.first_expert_phase.raw_phase, math.pi),
    )


def test_visual_token_overflow_is_explicit() -> None:
    core = TokenwiseOpticalMoE(16, fake_settings())
    hidden = torch.randn(5, 16)
    with pytest.raises(RuntimeError, match="visual token count 5 exceeds token panel capacity 4"):
        core(hidden, torch.tensor([0, 5], dtype=torch.int32))


def test_no_hidden_dimension_adapter_linear_exists() -> None:
    core = TokenwiseOpticalMoE(16, fake_settings())
    linear_shapes = [
        tuple(module.weight.shape)
        for module in core.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    assert linear_shapes == [(4, 16)]  # the router only
    assert not any("input_adapter" in name or "output_adapter" in name for name, _ in core.named_parameters())


def test_second_plane_reuses_one_router_forward() -> None:
    core = TokenwiseOpticalMoE(16, fake_settings("expert"))
    calls = 0

    def count(_module, _args, _output):
        nonlocal calls
        calls += 1

    handle = core.router.register_forward_hook(count)
    try:
        core(torch.randn(4, 16), torch.tensor([0, 4], dtype=torch.int32))
    finally:
        handle.remove()
    assert calls == 1


def test_parameter_counts_match_shared_default_design() -> None:
    settings = load_settings(EXPERIMENT / "configs" / "grocery10_tokenwise_moe4.yaml")
    core = TokenwiseOpticalMoE(1024, settings)
    report = core.parameter_breakdown()
    assert report["router_parameters"] == 4100
    assert report["first_expert_phase_parameters"] == 4096
    assert report["second_phase_parameters"] == 950 * 950
    assert report["input_adapter_parameters"] == 0
    assert report["output_adapter_parameters"] == 0
    assert report["total_parameters"] == 910696


def test_response_amplitude_preservation_is_finite_and_bounded() -> None:
    core = TokenwiseOpticalMoE(16, fake_settings())
    core(torch.randn(4, 16), torch.tensor([0, 4], dtype=torch.int32))
    assert core.last_response_gain is not None
    selected = core.last_routing["selected_mask"]
    gains = core.last_response_gain[selected]
    assert torch.isfinite(gains).all()
    assert float(gains.min()) >= 0.25
    assert float(gains.max()) <= 4.0


def test_language_adapter_and_optical_core_forward_backward() -> None:
    settings = fake_settings()
    language = TokenwiseLanguageSurrogate(32, settings)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    language.prepare_batch(mask)
    hidden = torch.randn(2, 4, 32, requires_grad=True)
    output = language(hidden)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    output[mask].square().mean().backward()
    assert language.input_adapter.weight.grad is not None
    assert language.output_adapter.weight.grad is not None
    assert language.optical_core.first_expert_phase.raw_phase.grad is not None


def test_language_token_overflow_is_explicit() -> None:
    language = TokenwiseLanguageSurrogate(32, fake_settings())
    with pytest.raises(RuntimeError, match="language token count 5 exceeds optical panel capacity 4"):
        language.prepare_batch(torch.ones(1, 5, dtype=torch.bool))


def test_nonshared_position_expert_ablation_has_independent_masks() -> None:
    settings = fake_settings()
    settings.share_expert_phase_across_tokens = False
    core = TokenwiseOpticalMoE(16, settings)
    assert core.first_expert_phase.raw_phase.shape == (4, 4, 4, 4)
    assert core.parameter_breakdown()["shared_expert_phase_across_tokens"] is False


def test_optical_language_configs_disable_deepstack_and_define_ablation() -> None:
    shared = load_settings(
        EXPERIMENT / "configs" / "grocery10_tokenwise_vision_language_moe4_shared.yaml"
    )
    nonshared = load_settings(
        EXPERIMENT / "configs" / "grocery10_tokenwise_vision_language_moe4_nonshared.yaml"
    )
    assert shared.student_language_mode == "optical_moe"
    assert shared.student_deepstack_enabled is False
    assert shared.max_language_tokens == 196
    assert shared.oeo_preserve_response_amplitude is True
    assert shared.share_expert_phase_across_tokens is True
    assert shared.evaluation_checkpoint == "best_observed_test"
    assert nonshared.share_expert_phase_across_tokens is False
