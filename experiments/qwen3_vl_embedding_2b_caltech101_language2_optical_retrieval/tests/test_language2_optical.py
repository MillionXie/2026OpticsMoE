from types import SimpleNamespace

import torch

from ..hardware_bridge import load_ccd
from ..optical_blocks import LanguageSecondLayerOpticalCore, RobustCCDNormalizer


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        electronic_width=6,
        electronic_expansion=2.0,
        electronic_token_mixer_enabled=True,
        electronic_token_mixer_kernel_size=3,
        electronic_vision_token_mixer_type="depthwise_conv2d",
        electronic_language_token_mixer_type="depthwise_conv1d",
        electronic_vision_token_mixer_kernel_size=3,
        electronic_language_token_mixer_kernel_size=3,
        electronic_dropout=0.0,
        electronic_initial_residual_weight=0.1,
        electronic_layers=2,
        optical_fusion_initial=0.05,
        language_optical_grid_size=8,
        language_optical_canvas_size=16,
        language_optical_input_rms=0.5,
        language_optical_ccd_target_mean=0.25,
        language_optical_max_shift_pixels=0,
        language_optical_ccd_shift_pixels=0,
        language_optical_gain_min=0.5,
        language_optical_gain_max=2.0,
        language_optical_offset_fraction=0.03,
        language_optical_read_noise_fraction=0.01,
        language_optical_background_quantile=0.01,
        language_optical_normalization_clip=12.0,
        language_optical_log_compression=1.0,
        language_optical_phase_parameterization="sigmoid",
        language_optical_phase_init="small_normal",
        language_optical_phase_init_std=0.02,
        language_optical_phase_dropout_mode="block_phase_bypass",
        language_optical_phase_dropout_p=0.05,
        language_optical_phase_dropout_block_size=2,
        language_optical_wavelength_nm=532.0,
        language_optical_pixel_pitch_um=16.0,
        language_optical_distance_m=0.01,
        language_optical_k_space_enabled=False,
        language_optical_theta_max_deg=1.0,
    )


def test_ccd_normalization_rejects_global_gain_and_offset() -> None:
    normalizer = RobustCCDNormalizer(8, _settings()).eval()
    value = torch.rand(2, 8, 8) + 0.1
    first = normalizer(value)
    second = normalizer(value * 3.7 + 0.4)
    assert torch.allclose(first, second, atol=2.0e-4, rtol=2.0e-4)


def test_language_second_layer_optics_has_finite_phase_gradient() -> None:
    core = LanguageSecondLayerOpticalCore(10, 8, _settings()).train()
    groups = [torch.randn(7, 10), torch.randn(5, 10)]
    packed, latent = core.forward_groups(groups, causal=True)
    loss = packed.square().mean() + core.optical_branch.current_operating_loss
    loss.backward()
    assert packed.shape == (12, 10)
    assert latent.shape == (2, 7, 6)
    gradient = core.optical_branch.phase.raw_phase.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_zero_initialized_optical_decoder_preserves_electronic_path() -> None:
    core = LanguageSecondLayerOpticalCore(10, 8, _settings()).eval()
    groups = [torch.randn(6, 10)]
    _, latent = core.forward_groups(groups, causal=True)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    input_latent = core.input_norm(core.input_adapter(groups[0].unsqueeze(0)))
    first = core.blocks[0](input_latent, padding_mask=mask, causal=True)
    electronic = core.blocks[1](first, padding_mask=mask, causal=True)
    expected = core.output_norm(electronic)
    assert torch.allclose(latent, expected, atol=1.0e-6, rtol=1.0e-6)


def test_physical_ccd_is_block_binned_without_interpolation(tmp_path) -> None:
    root = tmp_path / "ccd_captured"
    root.mkdir()
    physical = torch.arange(448 * 448, dtype=torch.float32).reshape(448, 448)
    torch.save(physical, root / "sample.pt")
    logical = load_ccd(
        tmp_path,
        "sample",
        use_simulation=False,
        flip_vertical=False,
        flip_horizontal=False,
    )
    expected = physical.reshape(224, 2, 224, 2).mean(dim=(1, 3))
    assert logical.shape == (224, 224)
    assert torch.equal(logical, expected)
