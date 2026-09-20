from pathlib import Path

import torch

from LightGenV2.tasks.t12_text_to_image.electronic_baseline import (
    ConditionalImageLatentDiscriminator,
    QwenElectronicGenerator,
    augment_image_latent_pair,
    load_electronic_gan_config,
)


TASK_DIR = Path(__file__).resolve().parents[1]


def test_prior_only_electronic_generator_is_single_pass_and_seeded() -> None:
    config = load_electronic_gan_config(TASK_DIR / "configs/qwen_vae_prior_gan.yaml")
    generator = QwenElectronicGenerator(64, config, categories=4).eval()
    text = torch.randn(2, 64)
    first = generator.generate(text, seed=42)
    assert first.shape == (2, 4, 28, 28)
    assert torch.equal(first, generator.generate(text, seed=42))
    assert not torch.equal(first, generator.generate(text, seed=43))
    report = generator.architecture_report()
    assert report["training_path_equals_inference_path"] is True
    assert report["image_conditioned_posterior"] is False
    assert report["inference_iterations"] == 1


def test_joint_discriminator_and_paired_augmentation_contract() -> None:
    discriminator = ConditionalImageLatentDiscriminator(64, categories=4, width=16)
    image = torch.randn(2, 3, 224, 224)
    latent = torch.randn(2, 4, 28, 28)
    text = torch.randn(2, 64)
    augmented_image, augmented_latent = augment_image_latent_pair(image, latent, 1.0)
    score, category, features = discriminator(augmented_image, augmented_latent, text)
    assert score.shape == (2, 1)
    assert category.shape == (2, 4)
    assert len(features) == 8
