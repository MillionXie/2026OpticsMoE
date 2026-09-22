"""Train the one-pass optical/electronic compositional background replacer."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .compact_turbo import latent_gradient_loss
from .electronic_turbo_infer import _load_adapter
from .product_repair_model import (
    RepairModelConfig,
    architecture_report,
    attach_decoder_optics,
    expand_reference_conditioning,
    one_step_edit,
)
from .product_scene_replace_data import (
    BRIGHTNESS,
    DIRECTIONS,
    ROOMS,
    TONES,
    ProductBackgroundReplacementDataset,
    write_pair_manifest,
)
from .product_scene_training import (
    SceneTrainingConfig,
    _masked_l1,
    _paired_noise,
    _seed_everything,
    load_scene_config,
)


class ReplacementLatentDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path) -> None:
        self.payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def __len__(self) -> int:
        return len(self.payload["reference"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = {
            key: self.payload[key][index].float()
            for key in ("reference", "target", "foreground_mask", "background_mask", "qwen_text")
        }
        for key in (
            "sample_ids", "source_ids", "prompts", "source_scene_ids", "target_scene_ids",
            "held_out_combinations", "room_indices", "tone_indices", "brightness_indices",
            "direction_indices",
        ):
            result[key] = self.payload[key][index]
        return result


@torch.inference_mode()
def cache_replacement_latents(
    *,
    data_dir: Path,
    instruction_cache: Path,
    vae_checkpoint: Path,
    output_dir: Path,
    image_size: int,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 4,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from diffusers import AutoencoderKL

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        vae_checkpoint, subfolder="vae", variant="fp16", torch_dtype=dtype,
        local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    scale = float(vae.config.scaling_factor)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "task": "existing scene + compositional instruction -> replaced background",
        "dataset": "ABO CleanRender CC BY 4.0 + procedural compositional backgrounds",
        "image_size": image_size,
        "latent_scaling_factor": scale,
        "attributes": {
            "rooms": list(ROOMS), "tones": list(TONES),
            "brightness": list(BRIGHTNESS), "directions": list(DIRECTIONS),
        },
        "splits": {},
    }
    tensor_keys = {"reference", "target", "foreground_mask", "background_mask", "qwen_text"}
    scalar_keys = {
        "room_indices": "room_index", "tone_indices": "tone_index",
        "brightness_indices": "brightness_index", "direction_indices": "direction_index",
    }
    string_keys = {
        "sample_ids": "sample_id", "source_ids": "source_id", "prompts": "prompt",
        "source_scene_ids": "source_scene_id", "target_scene_ids": "target_scene_id",
        "held_out_combinations": "held_out_combination",
    }
    for split in ("train", "val", "test"):
        dataset = ProductBackgroundReplacementDataset(data_dir, split, image_size, instruction_cache)
        write_pair_manifest(dataset, output_dir / f"{split}_pairs.jsonl")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        payload: dict[str, list[Any]] = {
            key: [] for key in (*tensor_keys, *scalar_keys, *string_keys)
        }
        for batch in loader:
            reference = batch["reference"].to(device=device, dtype=dtype, non_blocking=True)
            target = batch["target"].to(device=device, dtype=dtype, non_blocking=True)
            reference_latent = vae.encode(reference).latent_dist.mode() * scale
            target_latent = vae.encode(target).latent_dist.mode() * scale
            foreground = F.interpolate(batch["foreground_mask"].float(), reference_latent.shape[-2:], mode="area")
            payload["reference"].append(reference_latent.half().cpu())
            payload["target"].append(target_latent.half().cpu())
            payload["foreground_mask"].append(foreground.half().cpu())
            payload["background_mask"].append((1.0 - foreground).half().cpu())
            payload["qwen_text"].append(batch["qwen_text"].to(torch.bfloat16).cpu())
            for destination, source in scalar_keys.items():
                payload[destination].extend(batch[source].tolist())
            for destination, source in string_keys.items():
                payload[destination].extend(batch[source])
        packed = {
            key: torch.cat(value) if key in tensor_keys else value
            for key, value in payload.items()
        }
        torch.save(packed, output_dir / f"{split}.pt")
        summary["splits"][split] = {
            "pairs": len(dataset), "source_images": len(dataset.rows),
            "held_out_target_combinations": split in {"val", "test"},
        }
    (output_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


class TextAttributeRouter(nn.Module):
    def __init__(self, text_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(text_dim)
        self.room = nn.Linear(text_dim, len(ROOMS))
        self.tone = nn.Linear(text_dim, len(TONES))
        self.brightness = nn.Linear(text_dim, len(BRIGHTNESS))
        self.direction = nn.Linear(text_dim, len(DIRECTIONS))

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.norm(value.float())
        return {
            "room": self.room(value), "tone": self.tone(value),
            "brightness": self.brightness(value), "direction": self.direction(value),
        }


def _labels(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(batch[f"{name}_indices"], device=device, dtype=torch.long)
        for name in ("room", "tone", "brightness", "direction")
    }


def _attribute_loss(logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(F.cross_entropy(logits[name], labels[name]) for name in logits) / len(logits)


@torch.inference_mode()
def evaluate_replacement_model(
    unet, adapter, router, loader, sigma, device, *, residual_scale: float, noise_scale: float,
) -> dict[str, float]:
    unet.eval(); adapter.eval(); router.eval()
    totals = {
        "latent_mse": 0.0, "background_l1": 0.0, "foreground_change_l1": 0.0,
        "reference_mse": 0.0, "pair_delta_l1": 0.0,
        "room_accuracy": 0.0, "tone_accuracy": 0.0, "brightness_accuracy": 0.0,
        "direction_accuracy": 0.0, "exact_combination_accuracy": 0.0,
    }
    samples = 0
    generator = torch.Generator(device=device).manual_seed(9271)
    for batch in loader:
        reference = batch["reference"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        foreground = batch["foreground_mask"].to(device, non_blocking=True)
        background = batch["background_mask"].to(device, non_blocking=True)
        text = batch["qwen_text"].to(device, non_blocking=True)
        condition = adapter.condition(text)
        logits = router(text)
        labels = _labels(batch, device)
        noise = _paired_noise(reference, generator)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = one_step_edit(
                unet, noise, reference, condition, sigma,
                residual_scale=residual_scale, noise_scale=noise_scale,
            ).float()
        count = len(reference)
        totals["latent_mse"] += float(F.mse_loss(output, target)) * count
        totals["reference_mse"] += float(F.mse_loss(reference, target)) * count
        totals["background_l1"] += float(_masked_l1(output, target, background)) * count
        totals["foreground_change_l1"] += float(_masked_l1(output, reference, foreground)) * count
        totals["pair_delta_l1"] += float(_masked_l1(
            output[0::2] - output[1::2], target[0::2] - target[1::2], background[0::2]
        )) * count
        correct = []
        for name in logits:
            value = logits[name].argmax(-1) == labels[name]
            correct.append(value)
            totals[f"{name}_accuracy"] += float(value.float().mean()) * count
        totals["exact_combination_accuracy"] += float(torch.stack(correct).all(0).float().mean()) * count
        samples += count
    result = {key: value / samples for key, value in totals.items()}
    result["mse_improvement_over_copy"] = 1.0 - result["latent_mse"] / max(result["reference_mse"], 1e-8)
    return result


@torch.inference_mode()
def _sample_grid(
    *, unet, adapter, vae, sigma, latent_dataset: ReplacementLatentDataset,
    raw_dataset: ProductBackgroundReplacementDataset, output: Path, device: torch.device,
    residual_scale: float, noise_scale: float, seed: int,
) -> None:
    chosen = list(range(0, raw_dataset.targets_per_source))
    reference = latent_dataset.payload["reference"][chosen].float().to(device)
    text = latent_dataset.payload["qwen_text"][chosen].float().to(device)
    condition = adapter.condition(text)
    generator = torch.Generator(device=device).manual_seed(seed)
    shared_noise = torch.randn((1, *reference.shape[1:]), generator=generator, device=device)
    noise = shared_noise.expand(len(chosen), -1, -1, -1)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        generated_latent = one_step_edit(
            unet, noise, reference, condition, sigma,
            residual_scale=residual_scale, noise_scale=noise_scale,
        )
        generated = vae.decode(generated_latent.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
    generated = generated.float().clamp(-1, 1).cpu()
    cell, label_width = raw_dataset.image_size, 315
    canvas = Image.new("RGB", (label_width + cell * 3, cell * len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, index in enumerate(chosen):
        raw = raw_dataset[index]
        exact = generated[row_index] * raw["background_mask"] + raw["foreground_rgb"] * raw["foreground_mask"]
        for column, value in enumerate((raw["reference"], raw["target"], exact)):
            array = value.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(array), (label_width + column * cell, row_index * cell))
        draw.text((4, row_index * cell + 4), raw["target_scene_id"].replace("__", " / "), fill="black")
        draw.text((4, row_index * cell + 28), raw["prompt"][:48], fill="black")
        draw.text((4, row_index * cell + 52), "input scene | target scene | generated", fill="black")
        draw.text((4, row_index * cell + 76), "held-out composition", fill=(135, 25, 25))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94, subsampling=0)


def train_replacement_model(
    *, initial_unet: Path, turbo_checkpoint: Path, latent_cache_dir: Path,
    data_dir: Path, instruction_cache: Path, adapter_checkpoint: Path,
    output_dir: Path, model_config: RepairModelConfig,
    training_config: SceneTrainingConfig, device: torch.device,
    seed: int = 42, warm_start_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    _seed_everything(seed)
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    datasets = {name: ReplacementLatentDataset(latent_cache_dir / f"{name}.pt") for name in ("train", "val", "test")}
    loaders = {
        name: DataLoader(value, batch_size=2, shuffle=False, num_workers=training_config.num_workers, pin_memory=device.type == "cuda")
        for name, value in datasets.items()
    }
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    adapter.requires_grad_(True)
    router = TextAttributeRouter(datasets["train"].payload["qwen_text"].shape[1]).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        initial_unet, subfolder="unet", variant="fp16", torch_dtype=torch.float32, local_files_only=True,
    ).to(device)
    expand_reference_conditioning(unet)
    optical = attach_decoder_optics(unet, model_config)
    if warm_start_checkpoint is not None:
        warm = torch.load(warm_start_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        unet.load_state_dict(warm["unet"])
        adapter.load_state_dict(warm["adapter"])
        del warm
    else:
        nn.init.zeros_(unet.conv_out.weight)
        if unet.conv_out.bias is not None:
            nn.init.zeros_(unet.conv_out.bias)

    unet.enable_gradient_checkpointing()
    unet.requires_grad_(False)
    unet.conv_in.requires_grad_(True); unet.up_blocks.requires_grad_(True)
    unet.conv_norm_out.requires_grad_(True); unet.conv_out.requires_grad_(True)
    optical_ids = {id(parameter) for _, parameter in optical.optical_parameters()}
    optical_parameters = [p for p in unet.parameters() if p.requires_grad and id(p) in optical_ids]
    electronic_parameters = [p for p in unet.parameters() if p.requires_grad and id(p) not in optical_ids]
    optimizer = torch.optim.AdamW([
        {"params": electronic_parameters, "lr": training_config.learning_rate},
        {"params": optical_parameters, "lr": training_config.optical_learning_rate},
        {"params": list(adapter.parameters()), "lr": training_config.adapter_learning_rate},
        {"params": list(router.parameters()), "lr": training_config.adapter_learning_rate},
    ], weight_decay=training_config.weight_decay)
    scheduler = EulerDiscreteScheduler.from_pretrained(turbo_checkpoint, subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1, device=device)
    sigma = scheduler.sigmas[0]
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        turbo_checkpoint, subfolder="vae", variant="fp16", torch_dtype=dtype, local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    raw_val = ProductBackgroundReplacementDataset(data_dir, "val", training_config.image_size, instruction_cache)
    initial = evaluate_replacement_model(
        unet, adapter, router, loaders["val"], sigma, device,
        residual_scale=training_config.residual_scale, noise_scale=training_config.noise_scale,
    )
    best_value, best_epoch = math.inf, 0
    history: list[dict[str, Any]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, training_config.epochs + 1):
        unet.train(); adapter.train(); router.train()
        total = 0.0; count = 0
        for step, batch in enumerate(loaders["train"], 1):
            reference = batch["reference"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            foreground = batch["foreground_mask"].to(device, non_blocking=True)
            background = batch["background_mask"].to(device, non_blocking=True)
            text = batch["qwen_text"].to(device, non_blocking=True)
            condition = adapter.condition(text)
            logits = router(text); labels = _labels(batch, device)
            noise = _paired_noise(reference)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = one_step_edit(
                    unet, noise, reference, condition, sigma,
                    residual_scale=training_config.residual_scale, noise_scale=training_config.noise_scale,
                ).float()
                loss = F.mse_loss(output, target)
                loss = loss + training_config.background_weight * _masked_l1(output, target, background)
                loss = loss + training_config.foreground_preservation_weight * _masked_l1(output, reference, foreground)
                loss = loss + training_config.pair_difference_weight * _masked_l1(
                    output[0:1] - output[1:2], target[0:1] - target[1:2], background[0:1]
                )
                loss = loss + training_config.scene_router_weight * _attribute_loss(logits, labels)
                decoded_output = vae.decode(output.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0].float()
                with torch.no_grad():
                    decoded_target = vae.decode(target.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0].float()
                    decoded_reference = vae.decode(reference.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0].float()
                pixel_background = F.interpolate(background.float(), decoded_output.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
                pixel_foreground = 1.0 - pixel_background
                loss = loss + training_config.pixel_background_weight * _masked_l1(decoded_output, decoded_target, pixel_background)
                loss = loss + training_config.pixel_foreground_weight * _masked_l1(decoded_output, decoded_reference, pixel_foreground)
                loss = loss + training_config.detail_weight * latent_gradient_loss(output, target)
                scaled_loss = loss / training_config.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            if step % training_config.gradient_accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([*electronic_parameters, *optical_parameters, *adapter.parameters(), *router.parameters()], 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(reference); count += len(reference)
        validation = evaluate_replacement_model(
            unet, adapter, router, loaders["val"], sigma, device,
            residual_scale=training_config.residual_scale, noise_scale=training_config.noise_scale,
        )
        row = {"epoch": epoch, "train_loss": total / count, "validation": validation, "optical_alpha": float(optical.fusion.alpha.detach())}
        history.append(row); print(json.dumps(row), flush=True)
        if validation["latent_mse"] < best_value:
            best_value, best_epoch = validation["latent_mse"], epoch
            torch.save({
                "schema_version": 2, "epoch": epoch,
                "model_config": asdict(model_config), "training_config": asdict(training_config),
                "unet": {key: value.detach().half().cpu() for key, value in unet.state_dict().items()},
                "adapter": {key: value.detach().half().cpu() for key, value in adapter.state_dict().items()},
                "attribute_router": {key: value.detach().half().cpu() for key, value in router.state_dict().items()},
                "validation": validation,
            }, output_dir / "best_model.pt")
        if epoch % training_config.sample_every_epochs == 0:
            _sample_grid(
                unet=unet, adapter=adapter, vae=vae, sigma=sigma,
                latent_dataset=datasets["val"], raw_dataset=raw_val,
                output=output_dir / "samples" / f"epoch_{epoch:03d}.jpg", device=device,
                residual_scale=training_config.residual_scale, noise_scale=training_config.noise_scale,
                seed=seed + epoch,
            )

    best = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=False, mmap=True)
    unet.load_state_dict(best["unet"]); adapter.load_state_dict(best["adapter"]); router.load_state_dict(best["attribute_router"])
    del best
    test = evaluate_replacement_model(
        unet, adapter, router, loaders["test"], sigma, device,
        residual_scale=training_config.residual_scale, noise_scale=training_config.noise_scale,
    )
    qwen_meta = torch.load(instruction_cache, map_location="cpu", weights_only=False)["meta"]["qwen_pruning"]
    report_architecture = architecture_report(unet, vae, adapter, router, optical, model_config)
    report_architecture.update({
        "text_attribute_router_parameters": sum(p.numel() for p in router.parameters()),
        "qwen_text_encoder_parameters": qwen_meta["retained_text_encoder_parameters"],
        "qwen_language_layers": f"{qwen_meta['language_layers_retained']}/{qwen_meta['language_layers_original']}",
        "qwen_vision_tower_used": False,
        "qwen_lm_head_used": False,
    })
    report = {
        "schema_version": 2,
        "task": "background-present ABO lamp + compositional text -> replaced scene",
        "dataset": "ABO CleanRender CC BY 4.0 + generated compositional backgrounds",
        "initial_validation": initial, "best_epoch": best_epoch,
        "best_validation_latent_mse": best_value, "test_held_out_combinations": test,
        "architecture": report_architecture, "qwen_pruning": qwen_meta,
        "history": history, "training_seconds": time.perf_counter() - started,
        "gan_used": False, "diffusion_steps": 1,
        "warm_start_checkpoint": str(warm_start_checkpoint) if warm_start_checkpoint else None,
        "object_preservation": "exact source RGB alpha composite after generation",
    }
    (output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    del unet, vae, adapter, router, optimizer
    if device.type == "cuda": torch.cuda.empty_cache()
    return report


__all__ = [
    "ReplacementLatentDataset", "TextAttributeRouter", "cache_replacement_latents",
    "evaluate_replacement_model", "load_scene_config", "train_replacement_model",
]
