"""Training and diagnostics for the prior-only electronic text-to-image GAN."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import CachedLatentDataset, read_manifest
from .electronic_baseline import (
    ConditionalImageLatentDiscriminator,
    ElectronicGANConfig,
    QwenElectronicGenerator,
    augment_image_latent_pair,
)
from .feature_cache import _model_source
from .settings import Settings


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _category_indices(values: Sequence[str], categories: Sequence[str], device: torch.device) -> torch.Tensor:
    lookup = {name: index for index, name in enumerate(categories)}
    return torch.tensor([lookup[value] for value in values], device=device, dtype=torch.long)


def _feature_moments(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = features.float().flatten(2)
    return flattened.mean((0, 2)), flattened.std((0, 2), unbiased=False)


def _latent_moment_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    fake_mean, fake_std = _feature_moments(fake)
    real_mean, real_std = _feature_moments(real)
    return F.l1_loss(fake_mean, real_mean) + F.l1_loss(fake_std, real_std)


def _feature_matching_loss(fake: Sequence[torch.Tensor], real: Sequence[torch.Tensor]) -> torch.Tensor:
    losses = []
    for fake_value, real_value in zip(fake, real):
        fake_mean, fake_std = _feature_moments(fake_value)
        real_mean, real_std = _feature_moments(real_value)
        losses.extend((F.l1_loss(fake_mean, real_mean), F.l1_loss(fake_std, real_std)))
    return sum(losses) / len(losses)


@torch.no_grad()
def _update_ema(ema: nn.Module, model: nn.Module, beta: float) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - beta)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _polynomial_mmd(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Unbiased cubic MMD used only as a fixed validation-selection proxy."""

    first, second = first.float(), second.float()
    dimension = first.shape[1]
    kernel_xx = (first @ first.T / dimension + 1.0).pow(3)
    kernel_yy = (second @ second.T / dimension + 1.0).pow(3)
    kernel_xy = (first @ second.T / dimension + 1.0).pow(3)
    count = len(first)
    if count < 2:
        raise ValueError("MMD requires at least two samples")
    xx = (kernel_xx.sum() - kernel_xx.diag().sum()) / (count * (count - 1))
    yy = (kernel_yy.sum() - kernel_yy.diag().sum()) / (count * (count - 1))
    return xx + yy - 2 * kernel_xy.mean()


def _loader(settings: Settings, config: ElectronicGANConfig, split: str, shuffle: bool) -> DataLoader:
    return DataLoader(
        CachedLatentDataset(settings.cache_dir / f"{split}.pt"),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        generator=torch.Generator().manual_seed(settings.seed),
        drop_last=shuffle,
    )


def _decode(vae: nn.Module, latent: torch.Tensor, scaling: float) -> torch.Tensor:
    return vae.decode(latent / scaling).sample.clamp(-1, 1)


def _train_epoch(
    generator: QwenElectronicGenerator,
    ema: QwenElectronicGenerator,
    discriminator: ConditionalImageLatentDiscriminator,
    vae: nn.Module,
    loader: DataLoader,
    categories: Sequence[str],
    settings: Settings,
    config: ElectronicGANConfig,
    device: torch.device,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    totals: dict[str, float] = {}
    samples = 0
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    ema_beta = 0.5 ** (config.batch_size / max(1.0, config.ema_halflife_kimg * 1000.0))
    for batch in loader:
        text = batch["text"].to(device, non_blocking=True)
        real_latent = batch["latent"].to(device, non_blocking=True)
        labels = _category_indices(batch["category"], categories, device)
        noise = generator.sample_noise(len(text), device)

        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda"
        ):
            real_image = _decode(vae, real_latent, scaling)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda"
        ):
            fake_latent, text_category = generator.forward_with_aux(text, noise)
            fake_image = _decode(vae, fake_latent, scaling)
            second_noise = generator.sample_noise(len(text), device)
            second_latent = generator(text, second_noise)
            second_image = _decode(vae, second_latent, scaling)
            real_augmented, real_latent_augmented = augment_image_latent_pair(
                real_image, real_latent, config.augmentation_probability
            )
            fake_augmented, fake_latent_augmented = augment_image_latent_pair(
                fake_image, fake_latent, config.augmentation_probability
            )

        _set_requires_grad(discriminator, True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda"
        ):
            real_score, real_category, real_features = discriminator(
                real_augmented, real_latent_augmented, text
            )
            fake_score, _, _ = discriminator(
                fake_augmented.detach(), fake_latent_augmented.detach(), text
            )
            discriminator_hinge = (
                F.relu(1.0 - real_score.float()).mean()
                + F.relu(1.0 + fake_score.float()).mean()
            )
            discriminator_category = F.cross_entropy(real_category.float(), labels)
            discriminator_loss = discriminator_hinge + config.category_weight * discriminator_category
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        discriminator_optimizer.step()

        _set_requires_grad(discriminator, False)
        generator_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda"
        ):
            fake_score, fake_category, fake_features = discriminator(
                fake_augmented, fake_latent_augmented, text
            )
            generator_adversarial = -fake_score.float().mean()
            generator_category = F.cross_entropy(fake_category.float(), labels)
            text_category_loss = F.cross_entropy(text_category.float(), labels)
            feature_matching = _feature_matching_loss(fake_features, [value.detach() for value in real_features])
            latent_moments = _latent_moment_loss(fake_latent, real_latent)
            visible_difference = (fake_image.float() - second_image.float()).abs().mean((1, 2, 3))
            diversity = F.relu(config.diversity_target - visible_difference).mean()
            generator_loss = (
                config.adversarial_weight * generator_adversarial
                + config.category_weight * generator_category
                + config.feature_matching_weight * feature_matching
                + config.latent_moment_weight * latent_moments
                + config.text_category_weight * text_category_loss
                + config.diversity_weight * diversity
            )
        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        generator_optimizer.step()
        _update_ema(ema, generator, ema_beta)
        _set_requires_grad(discriminator, True)

        count = len(text)
        samples += count
        metrics = {
            "generator_loss": float(generator_loss.detach()),
            "generator_adversarial": float(generator_adversarial.detach()),
            "generator_category": float(generator_category.detach()),
            "text_category": float(text_category_loss.detach()),
            "feature_matching": float(feature_matching.detach()),
            "latent_moments": float(latent_moments.detach()),
            "diversity_penalty": float(diversity.detach()),
            "visible_seed_difference": float(visible_difference.detach().mean()),
            "discriminator_loss": float(discriminator_loss.detach()),
            "discriminator_hinge": float(discriminator_hinge.detach()),
            "discriminator_category": float(discriminator_category.detach()),
            "real_score": float(real_score.detach().mean()),
            "fake_score": float(fake_score.detach().mean()),
            "real_category_accuracy": float((real_category.argmax(1) == labels).float().mean()),
            "fake_category_accuracy": float((fake_category.argmax(1) == labels).float().mean()),
        }
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / samples for key, value in totals.items()}


@torch.inference_mode()
def _validate(
    generator: QwenElectronicGenerator,
    discriminator: ConditionalImageLatentDiscriminator,
    vae: nn.Module,
    loader: DataLoader,
    categories: Sequence[str],
    config: ElectronicGANConfig,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    real_latents, fake_latents, real_pixels, fake_pixels = [], [], [], []
    correct = count = 0
    for batch_index, batch in enumerate(loader):
        text = batch["text"].to(device)
        real_latent = batch["latent"].to(device)
        labels = _category_indices(batch["category"], categories, device)
        noise = generator.sample_noise(len(text), device, seed=seed + batch_index)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda"
        ):
            fake_latent = generator(text, noise)
            real_image = _decode(vae, real_latent, scaling)
            fake_image = _decode(vae, fake_latent, scaling)
            _, category_logits, _ = discriminator(fake_image, fake_latent, text)
        correct += int((category_logits.argmax(1) == labels).sum())
        count += len(text)
        real_latents.append(F.adaptive_avg_pool2d(real_latent.float(), (7, 7)).flatten(1).cpu())
        fake_latents.append(F.adaptive_avg_pool2d(fake_latent.float(), (7, 7)).flatten(1).cpu())
        real_pixels.append(F.adaptive_avg_pool2d(real_image.float(), (8, 8)).flatten(1).cpu())
        fake_pixels.append(F.adaptive_avg_pool2d(fake_image.float(), (8, 8)).flatten(1).cpu())
    latent_mmd = float(_polynomial_mmd(torch.cat(real_latents), torch.cat(fake_latents)))
    image_mmd = float(_polynomial_mmd(torch.cat(real_pixels), torch.cat(fake_pixels)))
    return {
        "selection_proxy": latent_mmd + image_mmd,
        "latent_polynomial_mmd": latent_mmd,
        "image_polynomial_mmd": image_mmd,
        "discriminator_category_accuracy": correct / count,
    }


def _images(value: torch.Tensor) -> list[Image.Image]:
    array = value.float().add(1).mul(127.5).clamp(0, 255).byte().cpu()
    return [Image.fromarray(item.permute(1, 2, 0).numpy(), mode="RGB") for item in array]


@torch.inference_mode()
def _write_seed_grid(
    generator: QwenElectronicGenerator,
    vae: nn.Module,
    settings: Settings,
    categories: Sequence[str],
    epoch: int,
    output: Path,
    device: torch.device,
) -> None:
    dataset = CachedLatentDataset(settings.cache_dir / "test.pt")
    rows = read_manifest(settings.data_dir / "test.jsonl")
    chosen = [next(index for index, value in enumerate(dataset.payload["categories"]) if value == name) for name in categories]
    text = dataset.payload["text"][chosen].float().to(device)
    real_latent = dataset.payload["latent"][chosen].float().to(device)
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    real = _images(_decode(vae, real_latent, scaling))
    seeds = (11, 29, 47, 83)
    generated: list[tuple[str, list[Image.Image]]] = []
    for seed in seeds:
        shared = generator.sample_noise(1, device, seed=seed).expand(len(text), -1)
        generated.append((f"prior seed {seed}", _images(_decode(vae, generator(text, shared), scaling))))
    cell, label, header = 224, 150, 28
    canvas = Image.new("RGB", (label + cell * len(categories), header + cell * (1 + len(seeds))), "white")
    draw = ImageDraw.Draw(canvas)
    captions = [rows[index].caption for index in chosen]
    for column, (category, caption) in enumerate(zip(categories, captions)):
        draw.text((label + column * cell + 4, 4), f"{category}: {caption}"[:32], fill="black")
    for row_index, (name, images) in enumerate([("real reference", real), *generated]):
        top = header + row_index * cell
        draw.text((4, top + cell // 2), name, fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (label + column * cell, top))
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output / f"prior_seed_grid_epoch_{epoch:04d}.png")


def train_electronic_baseline(
    settings: Settings,
    config: ElectronicGANConfig,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    _seed_everything(settings.seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    categories = sorted(set(CachedLatentDataset(settings.cache_dir / "train.pt").payload["categories"]))
    generator = QwenElectronicGenerator(settings.text_dim, config, settings.latent_channels).to(device)
    ema = copy.deepcopy(generator).eval().requires_grad_(False)
    discriminator = ConditionalImageLatentDiscriminator(
        settings.text_dim, len(categories), config.discriminator_width, config.latent_discriminator_weight
    ).to(device)
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=config.generator_learning_rate, betas=(0.0, 0.99), weight_decay=0.0
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=config.discriminator_learning_rate, betas=(0.0, 0.99), weight_decay=0.0
    )
    from diffusers import AutoencoderKL

    vae_source, vae_local = _model_source(settings.vae_checkpoint, settings.vae_model)
    vae = AutoencoderKL.from_pretrained(vae_source, local_files_only=vae_local).to(device).eval().requires_grad_(False)
    train_loader = _loader(settings, config, "train", True)
    val_loader = _loader(settings, config, "val", False)
    history = []
    best_proxy = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_epoch(
            generator, ema, discriminator, vae, train_loader, categories, settings, config, device,
            generator_optimizer, discriminator_optimizer,
        )
        row: dict[str, Any] = {"epoch": epoch, "train": train_metrics}
        evaluate = epoch == 1 or epoch % config.sample_every_epochs == 0 or epoch == config.epochs
        if evaluate:
            row["validation"] = _validate(
                ema, discriminator, vae, val_loader, categories, config, device, settings.seed + 10_000
            )
            _write_seed_grid(generator, vae, settings, categories, epoch, output_dir / "samples/raw", device)
            _write_seed_grid(ema, vae, settings, categories, epoch, output_dir / "samples/ema", device)
        history.append(row)
        print(json.dumps(row), flush=True)
        payload = {
            "schema_version": 1,
            "epoch": epoch,
            "variant": "qwen_vae_prior_gan",
            "settings": settings.to_dict(),
            "electronic_gan_config": config.__dict__,
            "categories": categories,
            "architecture": generator.architecture_report(),
            "generator": generator.state_dict(),
            "generator_ema": ema.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "metrics": row,
        }
        torch.save(payload, output_dir / "last_checkpoint.pt")
        if epoch % config.checkpoint_every_epochs == 0 or epoch == 1:
            torch.save(payload, output_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        proxy = row.get("validation", {}).get("selection_proxy")
        if proxy is not None and math.isfinite(proxy) and proxy < best_proxy:
            best_proxy = proxy
            torch.save(payload, output_dir / "best_proxy_checkpoint.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    report = {
        "epochs": config.epochs,
        "best_validation_proxy": best_proxy,
        "categories": categories,
        "architecture": generator.architecture_report(),
        "training_data_flow": "cached Qwen text + independent N(0,I) z -> electronic generator -> VAE latent -> frozen VAE decoder",
        "inference_data_flow": "Qwen text + independent N(0,I) z -> electronic generator -> VAE latent -> frozen VAE decoder",
    }
    (output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["train_electronic_baseline"]
