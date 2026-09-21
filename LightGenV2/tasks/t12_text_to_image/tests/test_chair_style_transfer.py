from __future__ import annotations

import torch

from LightGenV2.tasks.t12_text_to_image.chair_style_training import apply_product_style
from LightGenV2.tasks.t12_text_to_image.chair_style_transfer import (
    ChairStyleDiscriminator, ChairStyleTransfer, architecture_report, load_chair_style_config,
)
from LightGenV2.tasks.t12_text_to_image.settings import TASK_DIR


def _model(name: str) -> tuple[ChairStyleTransfer, object]:
    config = load_chair_style_config(TASK_DIR / f"configs/{name}")
    return ChairStyleTransfer(config).eval(), config


def test_electronic_style_model_preserves_white_background_and_is_small() -> None:
    model, config = _model("chair_style_electronic.yaml")
    discriminator = ChairStyleDiscriminator(config.text_dim)
    reference = torch.ones(2, 3, 128, 128)
    reference[:, :, 32:96, 48:80] = 0
    text = torch.randn(2, config.text_dim)
    with torch.inference_mode(): output = model(reference, text)
    assert output.shape == reference.shape
    assert torch.equal(output[:, :, :16, :16], reference[:, :, :16, :16])
    assert architecture_report(model, discriminator)["total_training_parameters"] < 10_000_000


def test_optical_and_electronic_are_parallel_and_phase_receives_gradient() -> None:
    model, config = _model("chair_style_optical.yaml")
    reference = torch.ones(1, 3, 128, 128); reference[:, :, 30:100, 40:90] = -0.2
    output = model(reference, torch.randn(1, config.text_dim)); output.mean().backward()
    assert model.bottleneck.optical.router_phase.grad is not None
    report = architecture_report(model)
    assert report["electronic_and_optical_parallel"] is True
    assert report["optical_backend"] == "compact_fft_simulation"


def test_synthetic_styles_do_not_change_background() -> None:
    reference = torch.ones(4, 3, 128, 128); reference[:, :, 32:96, 48:80] = -0.1
    target = apply_product_style(reference, torch.arange(4))
    assert torch.equal(target[:, :, :16, :16], reference[:, :, :16, :16])
    assert not torch.equal(target[:, :, 40:80, 52:76], reference[:, :, 40:80, 52:76])
