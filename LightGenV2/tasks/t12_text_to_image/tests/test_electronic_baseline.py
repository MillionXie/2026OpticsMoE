from dataclasses import replace
from pathlib import Path

import torch

from LightGenV2.tasks.t12_text_to_image.electronic_baseline import (
    ConditionalImageLatentDiscriminator,
    QwenElectronicGenerator,
    augment_image_latent_pair,
    load_electronic_gan_config,
)
from LightGenV2.tasks.t12_text_to_image.electronic_baseline_training import (
    _class_conditional_moment_loss,
    _sample_keyed_noise,
)
from LightGenV2.tasks.t12_text_to_image.electronic_turbo import (
    QwenTurboConditionAdapter,
    load_turbo_adapter_config,
)
from LightGenV2.tasks.t12_text_to_image.electronic_turbo_training import _teacher_prompt
from LightGenV2.tasks.t12_text_to_image.electronic_turbo_infer import _load_adapter


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


def test_refinement_noise_is_stable_per_sample_and_normally_distributed() -> None:
    ids = [f"sample-{index}" for index in range(128)]
    first = _sample_keyed_noise(ids, 256, torch.device("cpu"), seed=42)
    second = _sample_keyed_noise(ids, 256, torch.device("cpu"), seed=42)
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    assert abs(float(first.mean())) < 0.03
    assert 0.95 < float(first.std()) < 1.05


def test_class_conditional_moments_compare_within_categories() -> None:
    real = torch.randn(6, 4, 8, 8)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    assert float(_class_conditional_moment_loss(real, real, labels)) == 0.0
    shifted = real.clone()
    shifted[labels == 1] += 1
    assert float(_class_conditional_moment_loss(shifted, real, labels)) > 0.4


def test_qwen_turbo_adapter_reconstructs_one_step_condition_shape(tmp_path: Path) -> None:
    config = replace(
        load_turbo_adapter_config(TASK_DIR / "configs/qwen_sd_turbo_one_step.yaml"),
        pca_rank=4, hidden_dim=32, depth=1,
    )
    token_count, condition_dim = 5, 8
    basis = torch.linalg.qr(torch.randn(token_count * condition_dim, config.pca_rank)).Q.T
    adapter = QwenTurboConditionAdapter(
        64, config, torch.zeros(token_count * condition_dim), basis,
        torch.zeros(config.pca_rank), torch.ones(config.pca_rank), token_count, condition_dim,
    )
    probe = torch.randn(3, 64)
    adapter.eval()
    condition = adapter.condition(probe)
    assert condition.shape == (3, token_count, condition_dim)
    report = adapter.architecture_report()
    assert report["qwen_is_inference_conditioner"] is True
    assert report["inference_iterations"] == 1
    assert report["unet_calls"] == 1
    assert report["vae_decoder_calls"] == 1
    checkpoint = tmp_path / "adapter.pt"
    torch.save({
        "variant": "qwen_sd_turbo_one_step",
        "config": config.__dict__,
        "settings": {"text_dim": 64},
        "architecture": report,
        "adapter": adapter.state_dict(),
    }, checkpoint)
    restored, payload = _load_adapter(checkpoint, torch.device("cpu"))
    assert payload["variant"] == "qwen_sd_turbo_one_step"
    assert torch.allclose(restored.condition(probe), condition)


def test_turbo_teacher_prompt_requests_full_single_object() -> None:
    prompt = _teacher_prompt("a black leather shoe on a plain neutral background")
    assert "one footwear shoe" in prompt
    assert "black leather material and color" in prompt
    assert "laces and sole clearly visible" in prompt
    assert "entire shoe visible" in prompt
