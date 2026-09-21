"""Training loop for the compact direct-pixel CleanRender GAN."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .cleanrender_gan import (
    CleanRenderDiscriminator,
    CleanRenderGANConfig,
    CleanRenderGenerator,
    architecture_report,
)
from .dataset import read_manifest, sha256, validate_split_contract


class CleanRenderDataset(Dataset[dict[str, Any]]):
    def __init__(self, data_dir: Path, split: str, image_size: int) -> None:
        self.rows = read_manifest(data_dir / f"{split}.jsonl")
        payload = torch.load(
            data_dir / "qwen_text_cache" / f"{split}.pt",
            map_location="cpu", weights_only=False, mmap=True,
        )
        if payload["meta"]["manifest_sha256"] != sha256(data_dir / f"{split}.jsonl"):
            raise ValueError(f"Stale Qwen cache for {split}")
        if payload["sample_ids"] != [row.sample_id for row in self.rows]:
            raise ValueError(f"Qwen cache row order differs from {split} manifest")
        self.text = payload["text"]
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row.image_path) as handle:
            image = ImageOps.fit(
                ImageOps.exif_transpose(handle).convert("RGB"),
                (self.image_size, self.image_size), method=Image.Resampling.LANCZOS,
            )
        array = np.asarray(image).copy()
        value = torch.from_numpy(array).permute(2, 0, 1).float().div(127.5).sub(1.0)
        return {
            "sample_id": row.sample_id,
            "category": row.category,
            "text": self.text[index].float(),
            "image": value,
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    data_dir: Path,
    split: str,
    config: CleanRenderGANConfig,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        CleanRenderDataset(data_dir, split, config.image_size),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def _category_indices(names: Sequence[str], categories: Sequence[str], device: torch.device) -> torch.Tensor:
    lookup = {name: index for index, name in enumerate(categories)}
    return torch.tensor([lookup[name] for name in names], dtype=torch.long, device=device)


def _sample_keyed_noise(
    sample_ids: Sequence[str], dimension: int, device: torch.device, seed: int,
) -> torch.Tensor:
    rows = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
        rows.append(torch.randn(dimension, generator=generator))
    return torch.stack(rows).to(device)


def _augment(images: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return images
    active = torch.rand(len(images), 1, 1, 1, device=images.device) < probability
    return torch.where(active, images.flip(-1), images)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def _update_ema(ema: nn.Module, model: nn.Module, beta: float) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1 - beta)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _feature_statistics(fake: list[torch.Tensor], real: list[torch.Tensor]) -> torch.Tensor:
    losses = []
    for fake_value, real_value in zip(fake, real):
        fake_flat = fake_value.float().flatten(2)
        real_flat = real_value.float().flatten(2)
        losses.append(F.l1_loss(fake_flat.mean((0, 2)), real_flat.mean((0, 2)).detach()))
        losses.append(F.l1_loss(fake_flat.std((0, 2)), real_flat.std((0, 2)).detach()))
    return torch.stack(losses).mean()


def _white_border(image: torch.Tensor) -> torch.Tensor:
    width = max(2, image.shape[-1] // 16)
    border = torch.cat((
        image[:, :, :width, :].flatten(2), image[:, :, -width:, :].flatten(2),
        image[:, :, width:-width, :width].flatten(2), image[:, :, width:-width, -width:].flatten(2),
    ), dim=2)
    return (border.float() - 1.0).abs().mean()


def _foreground_low_frequency_reconstruction(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Supervise silhouette/color while preventing the white canvas from dominating L1."""

    fake_low = F.adaptive_avg_pool2d(fake.float(), (32, 32))
    real_low = F.adaptive_avg_pool2d(real.float(), (32, 32))
    foreground = (1.0 - real_low).abs().mean(1, keepdim=True).div(0.35).clamp(0, 1)
    weight = 1.0 + 5.0 * foreground
    return ((fake_low - real_low).abs() * weight).sum() / (weight.sum() * fake_low.shape[1])


def _mismatch_indices(labels: torch.Tensor) -> torch.Tensor:
    values = labels.tolist()
    result = []
    for index, label in enumerate(values):
        result.append(next((other for other, candidate in enumerate(values) if candidate != label), (index + 1) % len(values)))
    return torch.tensor(result, dtype=torch.long, device=labels.device)


def _train_epoch(
    generator: CleanRenderGenerator,
    ema: CleanRenderGenerator,
    discriminator: CleanRenderDiscriminator,
    loader: DataLoader,
    categories: Sequence[str],
    config: CleanRenderGANConfig,
    device: torch.device,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    seed: int,
) -> dict[str, float]:
    generator.train(); discriminator.train()
    totals: dict[str, float] = {}; samples = 0
    ema_beta = 0.5 ** (config.batch_size / max(1.0, config.ema_halflife_kimg * 1000.0))
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda")
    for batch in loader:
        real = batch["image"].to(device, non_blocking=True)
        text = batch["text"].to(device, non_blocking=True)
        labels = _category_indices(batch["category"], categories, device)
        noise = (
            _sample_keyed_noise(batch["sample_id"], config.noise_dim, device, seed)
            if config.fixed_noise_per_sample
            else generator.sample_noise(len(real), device)
        )

        with torch.autocast(**autocast):
            fake, text_category = generator.forward_with_aux(text, noise)
            second_noise = generator.sample_noise(len(real), device)
            second_fake = generator(text, second_noise)
            real_augmented = _augment(real, config.augmentation_probability)
            fake_augmented = _augment(fake, config.augmentation_probability)

        _set_requires_grad(discriminator, True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast):
            real_score, real_category, real_features = discriminator(real_augmented, text)
            fake_score, _, _ = discriminator(fake_augmented.detach(), text)
            mismatch = _mismatch_indices(labels)
            mismatch_score, _, _ = discriminator(real_augmented, text[mismatch])
            discriminator_hinge = (
                F.relu(1 - real_score.float()).mean()
                + 0.5 * F.relu(1 + fake_score.float()).mean()
                + 0.5 * F.relu(1 + mismatch_score.float()).mean()
            )
            discriminator_category = F.cross_entropy(real_category.float(), labels)
            discriminator_loss = discriminator_hinge + config.category_weight * discriminator_category
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        discriminator_optimizer.step()

        _set_requires_grad(discriminator, False)
        generator_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast):
            generated_score, generated_category, fake_features = discriminator(fake_augmented, text)
            generator_adversarial = -generated_score.float().mean()
            generator_category = F.cross_entropy(generated_category.float(), labels)
            text_category_loss = F.cross_entropy(text_category.float(), labels)
            feature_statistics = _feature_statistics(fake_features, [value.detach() for value in real_features])
            border = _white_border(fake)
            low_frequency_reconstruction = _foreground_low_frequency_reconstruction(fake, real)
            visible_difference = (fake.float() - second_fake.float()).abs().mean((1, 2, 3))
            diversity = F.relu(config.diversity_target - visible_difference).mean()
            generator_loss = (
                generator_adversarial
                + config.category_weight * generator_category
                + config.text_category_weight * text_category_loss
                + config.feature_statistics_weight * feature_statistics
                + config.white_border_weight * border
                + config.low_frequency_reconstruction_weight * low_frequency_reconstruction
                + config.diversity_weight * diversity
            )
        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        generator_optimizer.step()
        _update_ema(ema, generator, ema_beta)
        _set_requires_grad(discriminator, True)

        count = len(real); samples += count
        metrics = {
            "generator_loss": float(generator_loss.detach()),
            "generator_adversarial": float(generator_adversarial.detach()),
            "generator_category": float(generator_category.detach()),
            "text_category": float(text_category_loss.detach()),
            "feature_statistics": float(feature_statistics.detach()),
            "white_border": float(border.detach()),
            "low_frequency_reconstruction": float(low_frequency_reconstruction.detach()),
            "diversity_penalty": float(diversity.detach()),
            "visible_seed_difference": float(visible_difference.detach().mean()),
            "discriminator_loss": float(discriminator_loss.detach()),
            "discriminator_hinge": float(discriminator_hinge.detach()),
            "real_category_accuracy": float((real_category.argmax(1) == labels).float().mean()),
            "fake_category_accuracy": float((generated_category.argmax(1) == labels).float().mean()),
            "real_score": float(real_score.detach().mean()),
            "fake_score": float(fake_score.detach().mean()),
        }
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / samples for key, value in totals.items()}


def _polynomial_mmd(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    dimension = first.shape[1]
    xx = (first @ first.T / dimension + 1).pow(3)
    yy = (second @ second.T / dimension + 1).pow(3)
    xy = (first @ second.T / dimension + 1).pow(3)
    count = len(first)
    return (
        (xx.sum() - xx.diag().sum()) / (count * (count - 1))
        + (yy.sum() - yy.diag().sum()) / (count * (count - 1))
        - 2 * xy.mean()
    )


@torch.inference_mode()
def _validate(
    generator: CleanRenderGenerator,
    discriminator: CleanRenderDiscriminator,
    loader: DataLoader,
    categories: Sequence[str],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    generator.eval(); discriminator.eval()
    real_vectors, fake_vectors = [], []
    correct = count = 0; diversity = []
    for batch_index, batch in enumerate(loader):
        real = batch["image"].to(device)
        text = batch["text"].to(device)
        labels = _category_indices(batch["category"], categories, device)
        fake = generator.generate(text, seed + batch_index)
        second = generator.generate(text, seed + 1000 + batch_index)
        _, logits, _ = discriminator(fake, text)
        correct += int((logits.argmax(1) == labels).sum()); count += len(real)
        real_vectors.append(F.adaptive_avg_pool2d(real.float(), (16, 16)).flatten(1).cpu())
        fake_vectors.append(F.adaptive_avg_pool2d(fake.float(), (16, 16)).flatten(1).cpu())
        diversity.append((fake.float() - second.float()).abs().mean((1, 2, 3)).cpu())
    real_values, fake_values = torch.cat(real_vectors), torch.cat(fake_vectors)
    mmd = float(_polynomial_mmd(real_values, fake_values))
    return {
        "selection_proxy": mmd,
        "low_resolution_polynomial_mmd": mmd,
        "discriminator_category_accuracy": correct / count,
        "mean_seed_pixel_difference": float(torch.cat(diversity).mean()),
    }


def _pil(images: torch.Tensor) -> list[Image.Image]:
    array = images.float().add(1).mul(127.5).clamp(0, 255).byte().cpu()
    return [Image.fromarray(value.permute(1, 2, 0).numpy(), mode="RGB") for value in array]


@torch.inference_mode()
def _write_grid(
    generator: CleanRenderGenerator,
    dataset: CleanRenderDataset,
    categories: Sequence[str],
    output: Path,
    epoch: int,
    device: torch.device,
) -> None:
    chosen = [next(index for index, row in enumerate(dataset.rows) if row.category == category) for category in categories]
    text = dataset.text[chosen].float().to(device)
    real = _pil(torch.stack([dataset[index]["image"] for index in chosen]))
    seeds = (11, 29, 47, 83)
    rows = [(f"seed {seed}", _pil(generator.generate(text, seed))) for seed in seeds]
    size, label, header = generator.config.image_size, 82, 24
    canvas = Image.new("RGB", (label + size * len(categories), header + size * (1 + len(rows))), "white")
    draw = ImageDraw.Draw(canvas)
    for column, category in enumerate(categories):
        draw.text((label + column * size + 4, 4), category, fill="black")
    for row_index, (name, images) in enumerate([("real", real), *rows]):
        top = header + row_index * size
        draw.text((4, top + size // 2), name, fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (label + column * size, top))
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output / f"seed_grid_epoch_{epoch:04d}.png")


def train_cleanrender_gan(
    data_dir: Path,
    output_dir: Path,
    config: CleanRenderGANConfig,
    device: torch.device,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    _seed_everything(seed)
    contract = validate_split_contract(data_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    categories = contract["categories"]
    generator = CleanRenderGenerator(config, len(categories)).to(device)
    ema = copy.deepcopy(generator).eval().requires_grad_(False)
    discriminator = CleanRenderDiscriminator(config, len(categories)).to(device)
    architecture = architecture_report(generator, discriminator)
    if not architecture["under_10m_training_budget"]:
        raise ValueError(f"CleanRender model exceeds 10M budget: {architecture}")
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.generator_learning_rate, betas=(0.0, 0.99)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.discriminator_learning_rate, betas=(0.0, 0.99)
    )
    train_loader = _loader(data_dir, "train", config, seed, True)
    val_loader = _loader(data_dir, "val", config, seed, False)
    val_dataset = val_loader.dataset
    history: list[dict[str, Any]] = []
    best_proxy = float("inf")
    for epoch in range(1, config.epochs + 1):
        row: dict[str, Any] = {
            "epoch": epoch,
            "train": _train_epoch(
                generator, ema, discriminator, train_loader, categories, config, device,
                generator_optimizer, discriminator_optimizer, seed,
            ),
        }
        evaluate = epoch == 1 or epoch % config.sample_every_epochs == 0 or epoch == config.epochs
        if evaluate:
            row["validation"] = _validate(ema, discriminator, val_loader, categories, device, seed + 10_000)
            _write_grid(ema, val_dataset, categories, output_dir / "samples", epoch, device)
        history.append(row)
        print(json.dumps(row), flush=True)
        payload = {
            "schema_version": 1,
            "epoch": epoch,
            "variant": "qwen_cleanrender_direct_gan",
            "config": asdict(config),
            "categories": categories,
            "architecture": architecture,
            "generator": generator.state_dict(),
            "generator_ema": ema.state_dict(),
            "discriminator": discriminator.state_dict(),
            "metrics": row,
        }
        torch.save(payload, output_dir / "last_checkpoint.pt")
        if epoch == 1 or epoch % config.checkpoint_every_epochs == 0:
            torch.save(payload, output_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        proxy = row.get("validation", {}).get("selection_proxy")
        if proxy is not None and math.isfinite(proxy) and proxy < best_proxy:
            best_proxy = proxy
            torch.save(payload, output_dir / "best_checkpoint.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "variant": "qwen_cleanrender_direct_gan",
        "epochs": config.epochs,
        "best_validation_proxy": best_proxy,
        "categories": categories,
        "architecture": architecture,
        "training_data_flow": "image + cached frozen-Qwen caption -> conditional adversarial training",
        "inference_data_flow": "caption -> frozen Qwen -> Gaussian seed + one direct RGB decoder call",
        "input_image_at_inference": False,
        "random_seed_is_real": True,
        "training_noise": (
            "stable N(0,I) code keyed by sample_id" if config.fixed_noise_per_sample
            else "fresh independent N(0,I) code per step"
        ),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


__all__ = ["CleanRenderDataset", "train_cleanrender_gan"]
