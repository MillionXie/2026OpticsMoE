from __future__ import annotations

import torch

from LightGenV2.tasks.t12_text_to_image.cleanrender_gan import (
    CleanRenderDiscriminator,
    CleanRenderGenerator,
    architecture_report,
    load_cleanrender_config,
)
from LightGenV2.tasks.t12_text_to_image.settings import TASK_DIR


def test_cleanrender_training_model_is_under_ten_million() -> None:
    config = load_cleanrender_config(TASK_DIR / "configs/qwen_cleanrender_direct_gan.yaml")
    generator = CleanRenderGenerator(config, categories=3)
    discriminator = CleanRenderDiscriminator(config, categories=3)
    report = architecture_report(generator, discriminator)
    assert report["total_trainable_training_parameters"] < 10_000_000
    assert report["under_10m_training_budget"] is True
    assert report["unet"] is False
    assert report["vae"] is False


def test_cleanrender_generator_is_seeded_and_direct_rgb() -> None:
    config = load_cleanrender_config(TASK_DIR / "configs/qwen_cleanrender_direct_gan.yaml")
    model = CleanRenderGenerator(config, categories=3).eval()
    text = torch.randn(1, config.text_dim)
    with torch.inference_mode():
        first = model.generate(text, seed=17)
        repeated = model.generate(text, seed=17)
        different = model.generate(text, seed=18)
    assert first.shape == (1, 3, 128, 128)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)
