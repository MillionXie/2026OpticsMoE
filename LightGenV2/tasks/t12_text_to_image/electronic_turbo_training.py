"""Train and audit the Qwen-to-SD-Turbo one-step condition adapter."""

from __future__ import annotations

import copy
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .dataset import CachedLatentDataset, read_manifest
from .electronic_turbo import QwenTurboConditionAdapter, TurboAdapterConfig
from .feature_cache import _encode_qwen
from .settings import Settings


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _teacher_prompt(caption: str) -> str:
    description = caption.strip()
    if description.lower().startswith("a "):
        description = description[2:]
    description = re.sub(r"\s+on a plain (?:neutral|white) background\s*$", "", description, flags=re.I)
    lower = description.lower()
    contracts = {
        "shoe": ("footwear shoe", "laces and sole clearly visible", "entire shoe visible"),
        "chair": ("chair", "seat, backrest, and all legs clearly visible", "entire chair visible"),
        "lamp": ("lamp", "shade, stand, and base clearly visible", "entire lamp visible"),
        "table": ("table", "tabletop and all legs clearly visible", "entire table visible"),
    }
    category = next((name for name in contracts if re.search(rf"\b{name}\b", lower)), None)
    if category is None:
        return (
            f"a full view studio product photo of one {description}, "
            "entire object visible, centered, clean product photography"
        )
    noun, geometry, visibility = contracts[category]
    attributes = re.sub(rf"\b{category}\b", "", description, flags=re.I).strip(" ,")
    attribute_phrase = f", {attributes} material and color" if attributes else ""
    return (
        f"a full view studio product photo of one {noun}, {geometry}{attribute_phrase}, "
        f"{visibility}, centered, plain neutral background, clean product photography"
    )


def _synthetic_product_prompts() -> list[str]:
    """Small text-only coverage set; it adds no images and is never used for metric evaluation."""

    styles = ("minimalist", "retro", "Scandinavian", "industrial", "watercolor illustration style", "modern")
    colors = ("red", "orange", "mustard yellow", "teal blue", "matte black", "white", "forest green", "dark brown")
    materials = {
        "chair": ("velvet", "leather", "linen", "wood"),
        "lamp": ("painted metal", "brushed steel", "ceramic", "wood"),
        "shoe": ("leather", "suede", "canvas", "textile"),
        "table": ("dark walnut", "oak wood", "painted metal", "marble"),
    }
    return [
        f"a {style} {color} {material} {category} on a plain neutral background"
        for category, category_materials in materials.items()
        for style in styles
        for color in colors
        for material in category_materials
    ]


@torch.inference_mode()
def _encode_synthetic_qwen(
    prompts: Sequence[str], qwen_checkpoint: Path, batch_size: int, device: torch.device
) -> torch.Tensor:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(qwen_checkpoint, local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_checkpoint, local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    chunks = []
    for start in range(0, len(prompts), batch_size):
        chunks.append(_encode_qwen(qwen, processor, prompts[start : start + batch_size], device))
    del qwen, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(chunks)


@torch.inference_mode()
def _encode_teacher(
    captions: Sequence[str], turbo_checkpoint: Path, batch_size: int, device: torch.device
) -> torch.Tensor:
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(turbo_checkpoint / "tokenizer", local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(
        turbo_checkpoint / "text_encoder", variant="fp16",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    chunks = []
    prompts = [_teacher_prompt(value) for value in captions]
    for start in range(0, len(prompts), batch_size):
        tokens = tokenizer(
            prompts[start : start + batch_size], padding="max_length",
            max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        chunks.append(encoder(tokens).last_hidden_state.float().cpu())
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(chunks)


def _fit_condition_manifold(
    teacher: torch.Tensor, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = teacher.flatten(1).float()
    unique = torch.unique(flat, dim=0)
    mean = unique.mean(0)
    centered = unique - mean
    actual_rank = min(rank, len(unique) - 1, centered.shape[1])
    if actual_rank != rank:
        raise ValueError(f"Requested PCA rank {rank}, but only {actual_rank} is available")
    _, _, vectors = torch.pca_lowrank(centered.to(device), q=rank, center=False, niter=4)
    basis = vectors.T.cpu()
    coefficients = (flat - mean) @ basis.T
    coefficient_mean = coefficients.mean(0)
    coefficient_std = coefficients.std(0, unbiased=False).clamp_min(1e-6)
    standardized = (coefficients - coefficient_mean) / coefficient_std
    return mean, basis, coefficient_mean, coefficient_std, standardized


def _validation_metrics(
    adapter: QwenTurboConditionAdapter,
    qwen: torch.Tensor,
    teacher: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    adapter.eval()
    predicted_chunks = []
    with torch.inference_mode():
        for start in range(0, len(qwen), 32):
            predicted_chunks.append(adapter.condition(qwen[start : start + 32].to(device)).cpu())
    predicted = torch.cat(predicted_chunks).flatten(1)
    target = teacher.flatten(1).float()
    cosine = F.cosine_similarity(predicted, target, dim=1).mean()
    normalized_mse = F.mse_loss(predicted, target) / target.var(unbiased=False).clamp_min(1e-8)
    return {"teacher_cosine": float(cosine), "teacher_normalized_mse": float(normalized_mse)}


@torch.inference_mode()
def _write_qwen_seed_grid(
    adapter: QwenTurboConditionAdapter,
    settings: Settings,
    turbo_checkpoint: Path,
    output: Path,
    epoch: int,
    device: torch.device,
) -> dict[str, Any]:
    from diffusers import StableDiffusionPipeline

    dataset = CachedLatentDataset(settings.cache_dir / "test.pt")
    rows = read_manifest(settings.data_dir / "test.jsonl")
    categories = sorted(set(dataset.payload["categories"]))
    chosen = [
        next(index for index, value in enumerate(dataset.payload["categories"]) if value == category)
        for category in categories
    ]
    qwen = dataset.payload["text"][chosen].float().to(device)
    captions = [rows[index].caption for index in chosen]
    adapter.eval()
    condition = adapter.condition(qwen)
    pipe = StableDiffusionPipeline.from_pretrained(
        turbo_checkpoint, torch_dtype=torch.float16, variant="fp16", local_files_only=True,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    condition = condition.to(dtype=pipe.unet.dtype)
    seeds = (11, 29, 47, 83)
    generated: list[list[Image.Image]] = []
    times = []
    for seed in seeds:
        generators = [torch.Generator(device=device).manual_seed(seed) for _ in categories]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        images = pipe(
            prompt_embeds=condition, num_inference_steps=1, guidance_scale=0.0,
            height=512, width=512, generator=generators,
        ).images
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        generated.append(images)
    cell, label, header = 512, 110, 42
    canvas = Image.new("RGB", (label + cell * len(categories), header + cell * len(seeds)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (category, caption) in enumerate(zip(categories, captions)):
        draw.text((label + column * cell + 4, 4), f"{category}: {caption}"[:64], fill="black")
    for row_index, (seed, images) in enumerate(zip(seeds, generated)):
        top = header + row_index * cell
        draw.text((4, top + cell // 2), f"seed {seed}", fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (label + column * cell, top))
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output / f"qwen_seed_grid_epoch_{epoch:04d}.png")
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"seconds_per_four_images": times, "inference_iterations": 1, "vae_decodes": 1}


def train_turbo_adapter(
    settings: Settings,
    config: TurboAdapterConfig,
    turbo_checkpoint: Path,
    qwen_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    _seed_everything(settings.seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    license_path = turbo_checkpoint / "LICENSE.md"
    if license_path.is_file():
        shutil.copy2(license_path, output_dir / "SD_TURBO_LICENSE.md")
    rows = {split: read_manifest(settings.data_dir / f"{split}.jsonl") for split in ("train", "val")}
    cached = {split: CachedLatentDataset(settings.cache_dir / f"{split}.pt") for split in rows}
    synthetic_prompts = _synthetic_product_prompts() if config.synthetic_text_augmentation else []
    synthetic_qwen = (
        _encode_synthetic_qwen(synthetic_prompts, qwen_checkpoint, config.qwen_batch_size, device)
        if synthetic_prompts else torch.empty(0, settings.text_dim)
    )
    teacher = {
        split: _encode_teacher(
            [item.caption for item in rows[split]], turbo_checkpoint, config.teacher_batch_size, device
        )
        for split in rows
    }
    synthetic_teacher = (
        _encode_teacher(synthetic_prompts, turbo_checkpoint, config.teacher_batch_size, device)
        if synthetic_prompts else torch.empty(0, *teacher["train"].shape[1:])
    )
    torch.save(
        {
            **{split: {"sample_ids": cached[split].payload["sample_ids"], "condition": teacher[split].half()}
               for split in teacher},
            "synthetic": {"prompts": synthetic_prompts, "qwen": synthetic_qwen.half(),
                          "condition": synthetic_teacher.half()},
        },
        output_dir / "teacher_condition_cache.pt",
    )
    token_count, condition_dim = teacher["train"].shape[1:]
    unique_indices = []
    seen_captions: set[str] = set()
    for index, item in enumerate(rows["train"]):
        if item.caption not in seen_captions:
            seen_captions.add(item.caption)
            unique_indices.append(index)
    manifold_teacher = torch.cat((teacher["train"][unique_indices], synthetic_teacher))
    mean, basis, coefficient_mean, coefficient_std, _ = _fit_condition_manifold(
        manifold_teacher, config.pca_rank, device
    )
    train_teacher = torch.cat((teacher["train"], synthetic_teacher))
    train_flat = train_teacher.flatten(1).float()
    train_coefficients = (train_flat - mean) @ basis.T
    train_targets = (train_coefficients - coefficient_mean) / coefficient_std
    adapter = QwenTurboConditionAdapter(
        settings.text_dim, config, mean, basis, coefficient_mean, coefficient_std,
        token_count, condition_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_qwen = torch.cat((cached["train"].payload["text"].float(), synthetic_qwen.float()))
    loader = DataLoader(
        TensorDataset(train_qwen, train_targets), batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, generator=torch.Generator().manual_seed(settings.seed),
    )
    best_state = copy.deepcopy(adapter.state_dict())
    best_epoch, best_mse, stale = 0, float("inf"), 0
    history = []
    for epoch in range(1, config.epochs + 1):
        adapter.train()
        total = samples = 0
        for qwen_batch, target_batch in loader:
            qwen_batch, target_batch = qwen_batch.to(device), target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = adapter(qwen_batch)
            loss = F.mse_loss(predicted, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(qwen_batch)
            samples += len(qwen_batch)
        row: dict[str, Any] = {"epoch": epoch, "train_standardized_coefficient_mse": total / samples}
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            row["validation"] = _validation_metrics(
                adapter, cached["val"].payload["text"].float(), teacher["val"], device
            )
            value = row["validation"]["teacher_normalized_mse"]
            if value < best_mse:
                best_mse, best_epoch, stale = value, epoch, 0
                best_state = copy.deepcopy(adapter.state_dict())
            else:
                stale += 10
        if epoch % config.sample_every_epochs == 0:
            row["sample"] = _write_qwen_seed_grid(
                adapter, settings, turbo_checkpoint, output_dir / "samples", epoch, device
            )
        history.append(row)
        print(json.dumps(row), flush=True)
        if stale >= config.early_stopping_patience:
            break
    adapter.load_state_dict(best_state)
    final_sample = _write_qwen_seed_grid(
        adapter, settings, turbo_checkpoint, output_dir / "samples", best_epoch, device
    )
    payload = {
        "schema_version": 1,
        "variant": "qwen_sd_turbo_one_step",
        "epoch": best_epoch,
        "config": config.__dict__,
        "settings": settings.to_dict(),
        "architecture": adapter.architecture_report(),
        "adapter": adapter.state_dict(),
        "best_validation_normalized_mse": best_mse,
        "teacher_prompt_template": _teacher_prompt("<caption>"),
        "turbo_checkpoint": str(turbo_checkpoint),
        "qwen_checkpoint": str(qwen_checkpoint),
        "synthetic_text_prompts": len(synthetic_prompts),
    }
    torch.save(payload, output_dir / "best_adapter.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    report = {
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best_mse,
        "architecture": adapter.architecture_report(),
        "training_flow": "cached Qwen text -> trainable condition adapter -> frozen CLIP condition manifold",
        "synthetic_text_prompts": len(synthetic_prompts),
        "inference_flow": "Qwen text + seeded Gaussian latent -> one frozen SD-Turbo UNet call -> one frozen VAE decode",
        "final_sample": final_sample,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["train_turbo_adapter"]
