from __future__ import annotations

import torch

from LightGenV2.tasks.t12_text_to_image.cleanrender_gan import CleanRenderDiscriminator, CleanRenderGenerator
from LightGenV2.tasks.t12_text_to_image.cleanrender_vae import (
    CleanRenderImageEncoder,
    load_cleanrender_vae_config,
    vae_architecture_report,
)
from LightGenV2.tasks.t12_text_to_image.settings import TASK_DIR


def test_chair_vae_gan_is_under_ten_million() -> None:
    config = load_cleanrender_vae_config(TASK_DIR / "configs/qwen_cleanrender_chair_vae_gan.yaml")
    encoder = CleanRenderImageEncoder(config)
    decoder = CleanRenderGenerator(config.base, categories=1)
    discriminator = CleanRenderDiscriminator(config.base, categories=1)
    report = vae_architecture_report(encoder, decoder, discriminator)
    assert report["total_trainable_training_parameters"] < 10_000_000
    assert report["vae"] is True
    assert report["unet"] is False
    assert report["adversarial_training"] is True


def test_reference_variation_is_seeded_and_one_pass() -> None:
    config = load_cleanrender_vae_config(TASK_DIR / "configs/qwen_cleanrender_chair_vae_gan.yaml")
    encoder = CleanRenderImageEncoder(config).eval()
    decoder = CleanRenderGenerator(config.base, categories=1).eval()
    image = torch.randn(1, 3, 128, 128)
    text = torch.randn(1, config.base.text_dim)
    with torch.inference_mode():
        mean, log_variance = encoder(image)
        first_z = encoder.seeded_variation(mean, 0.3, 17)
        repeated_z = encoder.seeded_variation(mean, 0.3, 17)
        different_z = encoder.seeded_variation(mean, 0.3, 18)
        output = decoder(text, first_z)
    assert mean.shape == log_variance.shape == (1, config.base.noise_dim)
    assert torch.equal(first_z, repeated_z)
    assert not torch.equal(first_z, different_z)
    assert output.shape == (1, 3, 128, 128)
