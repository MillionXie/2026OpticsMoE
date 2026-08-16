import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from ..hardware_bridge import load_ccd
from ..optical_blocks import LanguageSecondLayerOpticalCore, RobustCCDNormalizer
from ..settings import load_settings


EXPERIMENT = Path(__file__).resolve().parents[1]


def _settings():
    settings = load_settings(
        EXPERIMENT / "configs" / "release" / "caltech101_language2_optical_residual.yaml"
    )
    settings.electronic_width = 6
    settings.electronic_expansion = 2.0
    settings.electronic_dropout = 0.0
    settings.language_optical_max_shift_pixels = 0
    settings.language_optical_phase_shift_pixels = 0
    settings.language_optical_ccd_shift_pixels = 0
    settings.k_space_constraint_enabled = False
    return settings


def test_ccd_normalization_rejects_global_gain_and_offset() -> None:
    settings = _settings()
    normalizer = RobustCCDNormalizer(settings).eval()
    value = torch.rand(2, 478, 478) + 0.1
    first = normalizer(value)
    second = normalizer(value * 3.7 + 0.4)
    assert torch.allclose(first, second, atol=2.0e-4, rtol=2.0e-4)


def test_moe4_language_second_layer_has_router_and_finite_phase_gradient() -> None:
    core = LanguageSecondLayerOpticalCore(10, 224, _settings()).train()
    groups = [torch.randn(5, 10)]
    packed, latent = core.forward_groups(groups, causal=True)
    loss = packed.square().mean() + core.optical_branch.current_operating_loss
    loss.backward()
    routing = core.last_routing
    assert packed.shape == (5, 10)
    assert latent.shape == (1, 5, 6)
    assert routing["selected_mask"].shape == (1, 4)
    assert torch.equal(routing["selected_mask"].sum(dim=1), torch.tensor([2]))
    gradients = [
        parameter.grad
        for name, parameter in core.optical_branch.named_parameters()
        if "raw_phase" in name
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_nearly_closed_optical_gate_preserves_electronic_path() -> None:
    core = LanguageSecondLayerOpticalCore(10, 224, _settings()).eval()
    core.optical_fusion_logit.data.fill_(-30.0)
    groups = [torch.randn(5, 10)]
    _, latent = core.forward_groups(groups, causal=True)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    input_latent = core.input_norm(core.input_adapter(groups[0].unsqueeze(0)))
    first = core.blocks[0](input_latent, padding_mask=mask, causal=True)
    electronic = core.blocks[1](first, padding_mask=mask, causal=True)
    assert torch.allclose(latent, core.output_norm(electronic), atol=1.0e-6, rtol=1.0e-6)


def test_physical_uint8_ccd_is_flipped_then_block_binned(tmp_path) -> None:
    root = tmp_path / "ccd_captured"
    root.mkdir()
    physical = np.arange(28 * 28, dtype=np.uint16).reshape(28, 28) % 256
    Image.fromarray(physical.astype(np.uint8), mode="L").save(root / "sample.png")
    settings = SimpleNamespace(
        hardware_ccd_target_size=14,
        hardware_ccd_physical_binning_factor=2,
        hardware_ccd_flip_vertical=True,
        hardware_ccd_flip_horizontal=False,
    )
    logical = load_ccd(
        tmp_path, "sample", use_simulation=False, settings=settings,
        flip_vertical=None, flip_horizontal=None,
    )
    expected = torch.from_numpy(np.flip(physical.astype(np.float32), axis=0).copy())
    expected = expected.reshape(14, 2, 14, 2).mean(dim=(1, 3))
    report = json.loads((tmp_path / "ccd_registered" / "sample.json").read_text())
    assert torch.equal(logical, expected)
    assert report["source_shape"] == [28, 28]
    assert report["flip_vertical"] is True
    assert report["registration_action"] == "flip_then_exact_2x2_block_mean"


def test_project_rejects_unprocessed_ccd_size(tmp_path) -> None:
    root = tmp_path / "ccd_captured"
    root.mkdir()
    Image.fromarray(np.zeros((30, 30), dtype=np.uint8), mode="L").save(root / "sample.png")
    settings = SimpleNamespace(
        hardware_ccd_target_size=14,
        hardware_ccd_physical_binning_factor=2,
        hardware_ccd_flip_vertical=False,
        hardware_ccd_flip_horizontal=False,
    )
    try:
        load_ccd(tmp_path, "sample", use_simulation=False, settings=settings)
    except RuntimeError as error:
        assert "hardware_sdk" in str(error)
    else:
        raise AssertionError("unprocessed CCD size was accepted")
