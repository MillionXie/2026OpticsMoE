"""Training for the chair-only compact conditional VAE-GAN."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .cleanrender_gan import CleanRenderDiscriminator, CleanRenderGenerator
from .cleanrender_training import CleanRenderDataset
from .cleanrender_vae import (
    CleanRenderImageEncoder,
    CleanRenderVAEConfig,
    config_payload,
    vae_architecture_report,
)
from .dataset import validate_split_contract


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(data_dir: Path, split: str, config: CleanRenderVAEConfig, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        CleanRenderDataset(data_dir, split, config.base.image_size),
        batch_size=config.base.batch_size,
        shuffle=shuffle,
        num_workers=config.base.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.base.num_workers > 0,
        drop_last=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def _update_ema(ema: nn.Module, model: nn.Module, beta: float) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1 - beta)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _augment_pair(image: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return image
    flipped = torch.rand(len(image), 1, 1, 1, device=image.device) < probability
    return torch.where(flipped, image.flip(-1), image)


def _foreground_reconstruction(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    losses = []
    for size in (32, 64):
        fake_low = F.adaptive_avg_pool2d(fake.float(), (size, size))
        real_low = F.adaptive_avg_pool2d(real.float(), (size, size))
        foreground = (1 - real_low).abs().mean(1, keepdim=True).div(0.25).clamp(0, 1)
        weight = 1 + 7 * foreground
        losses.append(((fake_low - real_low).abs() * weight).sum() / (weight.sum() * 3))
    return torch.stack(losses).mean()


def _edge_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    gray_fake = fake.float().mean(1, keepdim=True)
    gray_real = real.float().mean(1, keepdim=True)
    kernel_x = fake.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    fake_edge = torch.sqrt(
        F.conv2d(gray_fake, kernel_x, padding=1).square()
        + F.conv2d(gray_fake, kernel_y, padding=1).square() + 1e-6
    )
    real_edge = torch.sqrt(
        F.conv2d(gray_real, kernel_x, padding=1).square()
        + F.conv2d(gray_real, kernel_y, padding=1).square() + 1e-6
    )
    foreground = (1 - gray_real).abs().div(0.2).clamp(0, 1)
    return ((fake_edge - real_edge).abs() * (1 + 4 * foreground)).mean()


def _white_border(image: torch.Tensor) -> torch.Tensor:
    width = max(2, image.shape[-1] // 16)
    border = torch.cat((
        image[:, :, :width, :].flatten(2), image[:, :, -width:, :].flatten(2),
        image[:, :, width:-width, :width].flatten(2), image[:, :, width:-width, -width:].flatten(2),
    ), dim=2)
    return (border.float() - 1).abs().mean()


def _feature_statistics(fake: list[torch.Tensor], real: list[torch.Tensor]) -> torch.Tensor:
    values = []
    for fake_value, real_value in zip(fake, real):
        fake_flat, real_flat = fake_value.float().flatten(2), real_value.float().flatten(2)
        values.extend((
            F.l1_loss(fake_flat.mean((0, 2)), real_flat.mean((0, 2)).detach()),
            F.l1_loss(fake_flat.std((0, 2)), real_flat.std((0, 2)).detach()),
        ))
    return torch.stack(values).mean()


def _train_epoch(
    encoder: CleanRenderImageEncoder,
    encoder_ema: CleanRenderImageEncoder,
    decoder: CleanRenderGenerator,
    decoder_ema: CleanRenderGenerator,
    discriminator: CleanRenderDiscriminator,
    loader: DataLoader,
    config: CleanRenderVAEConfig,
    device: torch.device,
    autoencoder_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    epoch: int,
) -> dict[str, float]:
    encoder.train(); decoder.train(); discriminator.train()
    totals: dict[str, float] = {}; samples = 0
    base = config.base
    beta = 0.5 ** (base.batch_size / max(1.0, base.ema_halflife_kimg * 1000))
    kl_scale = config.kl_weight * min(1.0, epoch / config.kl_warmup_epochs)
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=base.amp and device.type == "cuda")
    for batch in loader:
        real = _augment_pair(batch["image"].to(device, non_blocking=True), base.augmentation_probability)
        text = batch["text"].to(device, non_blocking=True)
        with torch.autocast(**autocast):
            mean, log_variance = encoder(real)
            posterior = encoder.reparameterize(mean, log_variance)
            reconstruction = decoder(text, posterior)
            prior = torch.randn(len(real), base.noise_dim, device=device)
            prior_image = decoder(text, prior)

        _set_requires_grad(discriminator, True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast):
            real_score, _, _ = discriminator(real, text)
            reconstruction_score, _, _ = discriminator(reconstruction.detach(), text)
            prior_score, _, _ = discriminator(prior_image.detach(), text)
            mismatch_score, _, _ = discriminator(real, text.roll(1, 0))
            discriminator_loss = (
                F.relu(1 - real_score.float()).mean()
                + 0.35 * F.relu(1 + reconstruction_score.float()).mean()
                + 0.65 * F.relu(1 + prior_score.float()).mean()
                + config.mismatch_weight * F.relu(1 + mismatch_score.float()).mean()
            )
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        discriminator_optimizer.step()

        _set_requires_grad(discriminator, False)
        autoencoder_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast):
            reconstruction_score, _, reconstruction_features = discriminator(reconstruction, text)
            prior_score, _, prior_features = discriminator(prior_image, text)
            _, _, real_features = discriminator(real, text)
            reconstruction_loss = _foreground_reconstruction(reconstruction, real)
            edge = _edge_loss(reconstruction, real)
            kl = -0.5 * (1 + log_variance.float() - mean.float().square() - log_variance.float().exp()).mean()
            feature_statistics = _feature_statistics(prior_features, real_features)
            border = 0.5 * (_white_border(reconstruction) + _white_border(prior_image))
            second_prior_image = decoder(text, torch.randn_like(prior))
            seed_difference = (prior_image.float() - second_prior_image.float()).abs().mean((1, 2, 3))
            diversity = F.relu(base.diversity_target - seed_difference).mean()
            autoencoder_loss = (
                config.reconstruction_weight * reconstruction_loss
                + config.edge_weight * edge
                + kl_scale * kl
                - config.reconstruction_adversarial_weight * reconstruction_score.float().mean()
                - config.prior_adversarial_weight * prior_score.float().mean()
                + base.feature_statistics_weight * feature_statistics
                + base.white_border_weight * border
                + base.diversity_weight * diversity
            )
        autoencoder_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 5.0)
        autoencoder_optimizer.step()
        _update_ema(encoder_ema, encoder, beta)
        _update_ema(decoder_ema, decoder, beta)
        _set_requires_grad(discriminator, True)

        count = len(real); samples += count
        metrics = {
            "autoencoder_loss": float(autoencoder_loss.detach()),
            "foreground_reconstruction": float(reconstruction_loss.detach()),
            "edge_loss": float(edge.detach()),
            "kl": float(kl.detach()),
            "kl_scale": kl_scale,
            "reconstruction_adversarial_score": float(reconstruction_score.detach().mean()),
            "prior_adversarial_score": float(prior_score.detach().mean()),
            "feature_statistics": float(feature_statistics.detach()),
            "white_border": float(border.detach()),
            "visible_seed_difference": float(seed_difference.detach().mean()),
            "discriminator_loss": float(discriminator_loss.detach()),
            "real_score": float(real_score.detach().mean()),
        }
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / samples for key, value in totals.items()}


def _polynomial_mmd(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    dimension, count = first.shape[1], len(first)
    xx = (first @ first.T / dimension + 1).pow(3)
    yy = (second @ second.T / dimension + 1).pow(3)
    xy = (first @ second.T / dimension + 1).pow(3)
    return (
        (xx.sum() - xx.diag().sum()) / (count * (count - 1))
        + (yy.sum() - yy.diag().sum()) / (count * (count - 1))
        - 2 * xy.mean()
    )


@torch.inference_mode()
def _validate(
    encoder: CleanRenderImageEncoder,
    decoder: CleanRenderGenerator,
    loader: DataLoader,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    encoder.eval(); decoder.eval()
    real_vectors, prior_vectors = [], []
    reconstruction_total = edge_total = 0.0; count = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        real, text = batch["image"].to(device), batch["text"].to(device)
        mean, _ = encoder(real)
        reconstruction = decoder(text, mean)
        prior = torch.randn(len(real), decoder.config.noise_dim, generator=generator, device=device)
        generated = decoder(text, prior)
        reconstruction_total += float(_foreground_reconstruction(reconstruction, real)) * len(real)
        edge_total += float(_edge_loss(reconstruction, real)) * len(real)
        count += len(real)
        real_vectors.append(F.adaptive_avg_pool2d(real.float(), (16, 16)).flatten(1).cpu())
        prior_vectors.append(F.adaptive_avg_pool2d(generated.float(), (16, 16)).flatten(1).cpu())
    mmd = float(_polynomial_mmd(torch.cat(real_vectors), torch.cat(prior_vectors)))
    reconstruction = reconstruction_total / count
    edge = edge_total / count
    return {
        "selection_proxy": mmd + reconstruction,
        "prior_low_resolution_polynomial_mmd": mmd,
        "posterior_foreground_reconstruction": reconstruction,
        "posterior_edge_loss": edge,
    }


def _pil(images: torch.Tensor) -> list[Image.Image]:
    array = images.float().add(1).mul(127.5).clamp(0, 255).byte().cpu()
    return [Image.fromarray(value.permute(1, 2, 0).numpy(), mode="RGB") for value in array]


@torch.inference_mode()
def _write_grid(
    encoder: CleanRenderImageEncoder,
    decoder: CleanRenderGenerator,
    dataset: CleanRenderDataset,
    output: Path,
    epoch: int,
    device: torch.device,
    strength: float,
) -> None:
    chosen = list(range(min(6, len(dataset))))
    real_tensor = torch.stack([dataset[index]["image"] for index in chosen]).to(device)
    text = dataset.text[chosen].float().to(device)
    mean, _ = encoder(real_tensor)
    rows: list[tuple[str, list[Image.Image]]] = [("real", _pil(real_tensor)), ("recon", _pil(decoder(text, mean)))]
    for seed in (11, 29, 47):
        varied = encoder.seeded_variation(mean, strength, seed)
        rows.append((f"ref+{seed}", _pil(decoder(text, varied))))
    for seed in (11, 29):
        rows.append((f"prior {seed}", _pil(decoder.generate(text, seed))))
    size, label, header = decoder.config.image_size, 80, 24
    canvas = Image.new("RGB", (label + size * len(chosen), header + size * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for column in range(len(chosen)):
        draw.text((label + column * size + 4, 4), f"chair {column + 1}", fill="black")
    for row_index, (name, images) in enumerate(rows):
        top = header + row_index * size
        draw.text((4, top + size // 2), name, fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (label + column * size, top))
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output / f"vae_grid_epoch_{epoch:04d}.png")


def train_cleanrender_vae_gan(
    data_dir: Path,
    output_dir: Path,
    config: CleanRenderVAEConfig,
    device: torch.device,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    _seed_everything(seed)
    contract = validate_split_contract(data_dir)
    if [category.upper() for category in contract["categories"]] != ["CHAIR"]:
        raise ValueError(f"Chair VAE-GAN requires exactly one chair category, got {contract['categories']}")
    output_dir.mkdir(parents=True, exist_ok=False)
    encoder = CleanRenderImageEncoder(config).to(device)
    decoder = CleanRenderGenerator(config.base, categories=1).to(device)
    encoder_ema = copy.deepcopy(encoder).eval().requires_grad_(False)
    decoder_ema = copy.deepcopy(decoder).eval().requires_grad_(False)
    discriminator = CleanRenderDiscriminator(config.base, categories=1).to(device)
    architecture = vae_architecture_report(encoder, decoder, discriminator)
    if not architecture["under_10m_training_budget"]:
        raise ValueError(f"Chair VAE-GAN exceeds 10M budget: {architecture}")
    autoencoder_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.base.generator_learning_rate, betas=(0.0, 0.99),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.base.discriminator_learning_rate, betas=(0.0, 0.99),
    )
    train_loader = _loader(data_dir, "train", config, seed, True)
    val_loader = _loader(data_dir, "val", config, seed, False)
    history: list[dict[str, Any]] = []; best_proxy = float("inf")
    for epoch in range(1, config.base.epochs + 1):
        row: dict[str, Any] = {
            "epoch": epoch,
            "train": _train_epoch(
                encoder, encoder_ema, decoder, decoder_ema, discriminator, train_loader,
                config, device, autoencoder_optimizer, discriminator_optimizer, epoch,
            ),
        }
        evaluate = epoch == 1 or epoch % config.base.sample_every_epochs == 0 or epoch == config.base.epochs
        if evaluate:
            row["validation"] = _validate(encoder_ema, decoder_ema, val_loader, device, seed + 10_000)
            _write_grid(
                encoder_ema, decoder_ema, val_loader.dataset, output_dir / "samples", epoch,
                device, config.reference_variation_strength,
            )
        history.append(row)
        print(json.dumps(row), flush=True)
        payload = {
            "schema_version": 1,
            "epoch": epoch,
            "variant": "qwen_cleanrender_chair_vae_gan",
            "config": config_payload(config),
            "categories": contract["categories"],
            "architecture": architecture,
            "encoder": encoder.state_dict(),
            "encoder_ema": encoder_ema.state_dict(),
            "decoder": decoder.state_dict(),
            "decoder_ema": decoder_ema.state_dict(),
            "discriminator": discriminator.state_dict(),
            "metrics": row,
        }
        torch.save(payload, output_dir / "last_checkpoint.pt")
        if epoch == 1 or epoch % config.base.checkpoint_every_epochs == 0:
            torch.save(payload, output_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        proxy = row.get("validation", {}).get("selection_proxy")
        if proxy is not None and math.isfinite(proxy) and proxy < best_proxy:
            best_proxy = proxy
            torch.save(payload, output_dir / "best_checkpoint.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "variant": "qwen_cleanrender_chair_vae_gan",
        "epochs": config.base.epochs,
        "best_validation_proxy": best_proxy,
        "categories": contract["categories"],
        "architecture": architecture,
        "training_data_flow": "chair image -> VAE posterior; cached frozen-Qwen caption + posterior/prior -> one-pass RGB decoder; hinge GAN",
        "pure_generation": "caption -> frozen Qwen -> seeded N(0,I) -> one decoder call",
        "reference_variation": "reference chair -> encoder mean + seeded Gaussian offset -> caption-conditioned one decoder call",
        "random_seed_is_real": True,
        "adversarial_training": True,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


__all__ = ["train_cleanrender_vae_gan"]
