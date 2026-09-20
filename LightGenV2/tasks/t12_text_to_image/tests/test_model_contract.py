from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
from torch import nn

from LightGenV2.tasks.t12_text_to_image.modeling import (
    ParallelHybridBackbone,
    PatchDiscriminator,
    ScaleMatchedFusion,
    build_model,
)
from LightGenV2.tasks.t12_text_to_image.settings import load_settings
from LightGenV2.tasks.t12_text_to_image.training import _freeze_warmstarted_base, _load_warmstart


TASK_DIR = Path(__file__).resolve().parents[1]


def smoke_settings():
    return load_settings(TASK_DIR / "configs" / "smoke.yaml")


class ElectronicRecorder(nn.Module):
    def __init__(self, calls: list[torch.Tensor]) -> None:
        super().__init__()
        self.calls = calls

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        self.calls.append(tokens.detach().clone())
        return tokens + 0.25


class OpticalRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expert_calls: list[torch.Tensor] = []
        self.global_calls: list[torch.Tensor] = []
        self.last_routing = {"weights": torch.full((2, 4), 0.25)}

    def expert(self, tokens: torch.Tensor) -> torch.Tensor:
        self.expert_calls.append(tokens.detach().clone())
        return tokens - 0.25

    def global_block(self, tokens: torch.Tensor) -> torch.Tensor:
        self.global_calls.append(tokens.detach().clone())
        return tokens - 0.5


def test_electronic_and_optical_inputs_are_parallel_at_each_stage() -> None:
    settings = smoke_settings()
    backbone = ParallelHybridBackbone(settings, settings.width)
    electronic_calls: list[torch.Tensor] = []
    backbone.electronic1 = ElectronicRecorder(electronic_calls)
    backbone.electronic2 = ElectronicRecorder(electronic_calls)
    optics = OpticalRecorder()
    backbone.optical = optics
    tokens = torch.randn(2, settings.token_grid**2, settings.width)
    condition = torch.randn(2, settings.width)
    output = backbone(tokens, condition)
    assert torch.equal(electronic_calls[0], optics.expert_calls[0])
    assert torch.equal(electronic_calls[1], optics.global_calls[0])
    assert output.shape == tokens.shape


def test_scale_matched_fusion_preserves_electronic_rms() -> None:
    fusion = ScaleMatchedFusion(0.4, 0.05, 0.95, 1e-6)
    electronic = torch.randn(3, 16, 32)
    optical = torch.randn(3, 16, 32) * 20
    fused = fusion(electronic, optical)
    rms = lambda x: x.square().mean((1, 2)).sqrt()
    assert torch.allclose(rms(fused), rms(electronic), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", ["lightgen_parallel", "qwen_vae_baseline"])
def test_both_rows_emit_one_vae_latent_without_a_loop(variant: str) -> None:
    settings = smoke_settings()
    settings = dataclasses.replace(
        settings,
        variant=variant,
        optical_backend="compact_fft" if variant == "lightgen_parallel" else "none",
    )
    model = build_model(settings)
    text = torch.randn(2, settings.text_dim)
    generated = model.generate(text, seed=3)
    assert generated.shape == (2, 4, 8, 8)
    assert model.architecture_report()["inference_iterations"] == 1
    optical_names = [name for name, _ in model.named_parameters() if "optical" in name]
    assert bool(optical_names) is (variant == "lightgen_parallel")


def test_seed_controls_style_sampling() -> None:
    settings = smoke_settings()
    model = build_model(settings).eval()
    text = torch.randn(1, settings.text_dim)
    assert torch.equal(model.generate(text, seed=8), model.generate(text, seed=8))
    assert not torch.equal(model.generate(text, seed=8), model.generate(text, seed=9))


def test_formal_audited_dc20_geometry_accepts_14_square_tokens() -> None:
    settings = load_settings(TASK_DIR / "configs" / "lightgen_parallel.yaml")
    model = build_model(settings).eval()
    with torch.no_grad():
        generated = model.generate(torch.randn(1, settings.text_dim), seed=1)
    assert generated.shape == (1, 4, 28, 28)
    diagnostics = model.generator.backbone.core.fusion_diagnostics()
    assert set(diagnostics) == {"block1", "block2"}
    assert diagnostics["block1"]["fused_to_electronic_rms_ratio"] == pytest.approx(1.0, abs=1e-5)


def test_deep_single_pass_decoder_and_patch_discriminator_shapes() -> None:
    settings = load_settings(TASK_DIR / "configs" / "qwen_vae_baseline_gan.yaml")
    model = build_model(settings).eval()
    latent = model.generate(torch.randn(2, settings.text_dim), seed=4)
    assert latent.shape == (2, 4, 28, 28)
    assert model.architecture_report()["decoder_residual_depth"] == 4
    logits, features = PatchDiscriminator(settings.discriminator_width)(torch.randn(2, 3, 224, 224))
    assert logits.shape[0] == 2
    assert logits.ndim == 4
    assert len(features) == 4


def test_deep_decoder_can_warmstart_from_shallow_checkpoint(tmp_path: Path) -> None:
    shallow_settings = load_settings(TASK_DIR / "configs" / "qwen_vae_baseline.yaml")
    deep_settings = load_settings(TASK_DIR / "configs" / "qwen_vae_baseline_gan.yaml")
    shallow = build_model(shallow_settings)
    with torch.no_grad():
        shallow.generator.text_projection[1].weight.fill_(0.125)
        shallow.generator.head.net[6].weight.fill_(0.25)
    checkpoint = tmp_path / "shallow.pt"
    torch.save({
        "epoch": 80,
        "variant": shallow_settings.variant,
        "settings": shallow_settings.to_dict(),
        "architecture": shallow.architecture_report(),
        "model": shallow.state_dict(),
    }, checkpoint)

    deep = build_model(deep_settings)
    report = _load_warmstart(deep, deep_settings, checkpoint)

    assert report["source_epoch"] == 80
    assert report["source_decoder_depth"] == 0
    assert report["target_decoder_depth"] == 4
    assert report["new_tensors"]
    assert all(key.startswith("generator.head.net.") for key in report["new_tensors"])
    assert torch.equal(
        deep.generator.text_projection[1].weight,
        shallow.generator.text_projection[1].weight,
    )
    assert torch.equal(deep.generator.head.net[10].weight, shallow.generator.head.net[6].weight)

    freeze = _freeze_warmstarted_base(deep, deep_settings.decoder_depth)
    assert freeze["trainable_parameters"] > 0
    trainable = [name for name, parameter in deep.named_parameters() if parameter.requires_grad]
    assert trainable == freeze["trainable_tensors"]
    assert all(name.startswith("generator.head.net.") for name in trainable)

    text = torch.randn(2, deep_settings.text_dim)
    style = torch.randn(2, deep_settings.style_dim)
    refined, base = deep.generator.forward_with_base_decoder(text, style)
    assert refined.shape == base.shape == (2, 4, 28, 28)
