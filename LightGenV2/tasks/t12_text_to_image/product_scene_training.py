"""Cache and train the one-pass optical/electronic ABO lamp scene editor."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
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
from .product_scene_data import ProductSceneDataset, SCENES, write_pair_manifest


@dataclass(frozen=True)
class SceneTrainingConfig:
    image_size: int
    batch_size: int
    gradient_accumulation: int
    epochs: int
    learning_rate: float
    optical_learning_rate: float
    adapter_learning_rate: float
    weight_decay: float
    background_weight: float
    foreground_preservation_weight: float
    pair_difference_weight: float
    scene_router_weight: float
    pixel_background_weight: float
    pixel_foreground_weight: float
    detail_weight: float
    residual_scale: float
    noise_scale: float
    num_workers: int
    sample_every_epochs: int

    def validate(self) -> None:
        if min(
            self.image_size, self.batch_size, self.gradient_accumulation, self.epochs,
            self.learning_rate, self.optical_learning_rate, self.adapter_learning_rate,
            self.residual_scale, self.sample_every_epochs,
        ) <= 0:
            raise ValueError("Positive scene training settings are required")
        if self.batch_size != 2:
            raise ValueError("Scene counterfactual batches must contain two adjacent prompts")
        if min(
            self.weight_decay, self.background_weight, self.foreground_preservation_weight,
            self.pair_difference_weight, self.scene_router_weight,
            self.pixel_background_weight, self.pixel_foreground_weight,
            self.detail_weight, self.noise_scale, self.num_workers,
        ) < 0:
            raise ValueError("Non-negative scene regularization settings are required")


def load_scene_config(path: Path) -> tuple[RepairModelConfig, SceneTrainingConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    model, fusion, training = raw["model"], raw["fusion"], raw["training"]
    model_config = RepairModelConfig(
        optical_width=int(model["optical_width"]),
        optical_grid=int(model["optical_grid"]),
        optical_experts=int(model["optical_experts"]),
        optical_top_k=int(model["optical_top_k"]),
        alpha_initial=float(fusion["alpha_initial"]),
        alpha_minimum=float(fusion["alpha_minimum"]),
        alpha_maximum=float(fusion["alpha_maximum"]),
        rms_epsilon=float(fusion["rms_epsilon"]),
    )
    float_keys = {
        "learning_rate", "optical_learning_rate", "adapter_learning_rate", "weight_decay",
        "background_weight", "foreground_preservation_weight", "pair_difference_weight",
        "scene_router_weight", "pixel_background_weight", "pixel_foreground_weight",
        "detail_weight", "residual_scale", "noise_scale",
    }
    training_config = SceneTrainingConfig(**{
        key: float(value) if key in float_keys else int(value)
        for key, value in training.items()
    })
    model_config.validate()
    training_config.validate()
    return model_config, training_config


class SceneLatentDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path) -> None:
        self.payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def __len__(self) -> int:
        return len(self.payload["reference"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "reference": self.payload["reference"][index].float(),
            "target": self.payload["target"][index].float(),
            "foreground_mask": self.payload["foreground_mask"][index].float(),
            "background_mask": self.payload["background_mask"][index].float(),
            "qwen_text": self.payload["qwen_text"][index].float(),
            "sample_id": self.payload["sample_ids"][index],
            "source_id": self.payload["source_ids"][index],
            "prompt": self.payload["prompts"][index],
            "scene_index": self.payload["scene_indices"][index],
            "scene_id": self.payload["scene_ids"][index],
        }


@torch.inference_mode()
def cache_scene_latents(
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
        "schema_version": 1,
        "task": "white-background lamp + text -> styled product scene",
        "dataset": "ABO CleanRender CC BY 4.0 + procedural backgrounds",
        "image_size": image_size,
        "latent_scaling_factor": scale,
        "scenes": [scene["id"] for scene in SCENES],
        "splits": {},
    }
    for split in ("train", "val", "test"):
        dataset = ProductSceneDataset(data_dir, split, image_size, instruction_cache)
        write_pair_manifest(dataset, output_dir / f"{split}_pairs.jsonl")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        payload: dict[str, list[Any]] = {
            "reference": [], "target": [], "foreground_mask": [], "background_mask": [],
            "qwen_text": [], "sample_ids": [], "source_ids": [], "prompts": [],
            "scene_indices": [], "scene_ids": [],
        }
        for batch in loader:
            reference = batch["reference"].to(device=device, dtype=dtype, non_blocking=True)
            target = batch["target"].to(device=device, dtype=dtype, non_blocking=True)
            reference_latent = vae.encode(reference).latent_dist.mode() * scale
            target_latent = vae.encode(target).latent_dist.mode() * scale
            latent_size = reference_latent.shape[-2:]
            foreground = F.interpolate(batch["foreground_mask"].float(), latent_size, mode="area")
            background = 1.0 - foreground
            payload["reference"].append(reference_latent.half().cpu())
            payload["target"].append(target_latent.half().cpu())
            payload["foreground_mask"].append(foreground.half().cpu())
            payload["background_mask"].append(background.half().cpu())
            payload["qwen_text"].append(batch["qwen_text"].to(torch.bfloat16).cpu())
            for key, source in (
                ("sample_ids", "sample_id"), ("source_ids", "source_id"),
                ("prompts", "prompt"), ("scene_ids", "scene_id"),
            ):
                payload[key].extend(batch[source])
            payload["scene_indices"].extend(batch["scene_index"].tolist())
        packed = {
            key: torch.cat(value) if key in {
                "reference", "target", "foreground_mask", "background_mask", "qwen_text"
            } else value
            for key, value in payload.items()
        }
        torch.save(packed, output_dir / f"{split}.pt")
        summary["splits"][split] = len(dataset)
    (output_dir / "cache_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


class TextSceneRouter(nn.Module):
    def __init__(self, text_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, len(SCENES)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value.float())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_l1(value: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((value - target).abs() * mask).sum() / (mask.sum() * value.shape[1]).clamp_min(1.0)


def _paired_noise(reference: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    first = torch.randn(
        (len(reference) // 2, *reference.shape[1:]), device=reference.device,
        dtype=reference.dtype, generator=generator,
    )
    return first.repeat_interleave(2, dim=0)


@torch.inference_mode()
def evaluate_scene_model(
    unet, adapter, scene_router, loader, sigma, device, *, residual_scale: float, noise_scale: float,
) -> dict[str, float]:
    unet.eval()
    adapter.eval()
    scene_router.eval()
    totals = {
        "latent_mse": 0.0, "background_l1": 0.0, "foreground_change_l1": 0.0,
        "reference_mse": 0.0, "pair_delta_l1": 0.0, "scene_accuracy": 0.0,
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
        scene_logits = scene_router(text)
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
        labels = torch.as_tensor(batch["scene_index"], device=device, dtype=torch.long)
        totals["scene_accuracy"] += float((scene_logits.argmax(-1) == labels).float().mean()) * count
        samples += count
    result = {key: value / samples for key, value in totals.items()}
    result["mse_improvement_over_copy"] = 1.0 - result["latent_mse"] / max(
        result["reference_mse"], 1e-8
    )
    return result


@torch.inference_mode()
def _sample_grid(
    *, unet, adapter, vae, sigma, latent_dataset: SceneLatentDataset,
    raw_dataset: ProductSceneDataset, output: Path, device: torch.device,
    residual_scale: float, noise_scale: float, seed: int,
) -> None:
    chosen = list(range(len(SCENES)))
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
        generated = vae.decode(
            generated_latent.to(vae.dtype) / vae.config.scaling_factor, return_dict=False
        )[0]
    generated = generated.float().clamp(-1, 1).cpu()
    cell, label_width = raw_dataset.image_size, 250
    canvas = Image.new("RGB", (label_width + cell * 3, cell * len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, index in enumerate(chosen):
        raw = raw_dataset[index]
        # The source alpha is available by task definition.  Re-compositing the
        # exact source object after generation avoids latent-grid halos and
        # guarantees that the requested edit changes only the scene.
        exact = (
            generated[row_index] * raw["background_mask"]
            + raw["foreground_rgb"] * raw["foreground_mask"]
        )
        for column, value in enumerate((raw["reference"], raw["target"], exact)):
            array = value.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(array), (label_width + column * cell, row_index * cell))
        draw.text((4, row_index * cell + 4), raw["scene_label"], fill="black")
        draw.text((4, row_index * cell + 28), raw["prompt"][:39], fill="black")
        draw.text((4, row_index * cell + 52), "input | target | generated", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94, subsampling=0)


def train_scene_model(
    *,
    initial_unet: Path,
    turbo_checkpoint: Path,
    latent_cache_dir: Path,
    data_dir: Path,
    instruction_cache: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    model_config: RepairModelConfig,
    training_config: SceneTrainingConfig,
    device: torch.device,
    seed: int = 42,
    warm_start_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    _seed_everything(seed)
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    train = SceneLatentDataset(latent_cache_dir / "train.pt")
    val = SceneLatentDataset(latent_cache_dir / "val.pt")
    train_loader = DataLoader(
        train, batch_size=2, shuffle=False, num_workers=training_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val, batch_size=2, shuffle=False, num_workers=training_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    adapter.requires_grad_(True)
    scene_router = TextSceneRouter(train.payload["qwen_text"].shape[1]).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        initial_unet, subfolder="unet", variant="fp16", torch_dtype=torch.float32,
        local_files_only=True,
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
    unet.conv_in.requires_grad_(True)
    unet.up_blocks.requires_grad_(True)
    unet.conv_norm_out.requires_grad_(True)
    unet.conv_out.requires_grad_(True)
    optical_ids = {id(parameter) for _, parameter in optical.optical_parameters()}
    optical_parameters = [p for p in unet.parameters() if p.requires_grad and id(p) in optical_ids]
    electronic_parameters = [p for p in unet.parameters() if p.requires_grad and id(p) not in optical_ids]
    optimizer = torch.optim.AdamW([
        {"params": electronic_parameters, "lr": training_config.learning_rate},
        {"params": optical_parameters, "lr": training_config.optical_learning_rate},
        {"params": list(adapter.parameters()), "lr": training_config.adapter_learning_rate},
        {"params": list(scene_router.parameters()), "lr": training_config.adapter_learning_rate},
    ], weight_decay=training_config.weight_decay)
    scheduler = EulerDiscreteScheduler.from_pretrained(
        turbo_checkpoint, subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(1, device=device)
    sigma = scheduler.sigmas[0]
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        turbo_checkpoint, subfolder="vae", variant="fp16", torch_dtype=dtype,
        local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    raw_val = ProductSceneDataset(
        data_dir, "val", training_config.image_size, instruction_cache
    )
    initial = evaluate_scene_model(
        unet, adapter, scene_router, val_loader, sigma, device,
        residual_scale=training_config.residual_scale,
        noise_scale=training_config.noise_scale,
    )
    best_value = math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, training_config.epochs + 1):
        unet.train()
        adapter.train()
        scene_router.train()
        total = 0.0
        count = 0
        for step, batch in enumerate(train_loader, 1):
            reference = batch["reference"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            foreground = batch["foreground_mask"].to(device, non_blocking=True)
            background = batch["background_mask"].to(device, non_blocking=True)
            text = batch["qwen_text"].to(device, non_blocking=True)
            condition = adapter.condition(text)
            scene_logits = scene_router(text)
            labels = torch.as_tensor(batch["scene_index"], device=device, dtype=torch.long)
            noise = _paired_noise(reference)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = one_step_edit(
                    unet, noise, reference, condition, sigma,
                    residual_scale=training_config.residual_scale,
                    noise_scale=training_config.noise_scale,
                ).float()
                global_loss = F.mse_loss(output, target)
                background_loss = _masked_l1(output, target, background)
                foreground_loss = _masked_l1(output, reference, foreground)
                pair_loss = _masked_l1(
                    output[0:1] - output[1:2], target[0:1] - target[1:2], background[0:1]
                )
                loss = global_loss
                loss = loss + training_config.background_weight * background_loss
                loss = loss + training_config.foreground_preservation_weight * foreground_loss
                loss = loss + training_config.pair_difference_weight * pair_loss
                loss = loss + training_config.scene_router_weight * F.cross_entropy(scene_logits, labels)
                decoded_output = vae.decode(
                    output.to(vae.dtype) / vae.config.scaling_factor, return_dict=False
                )[0].float()
                with torch.no_grad():
                    decoded_target = vae.decode(
                        target.to(vae.dtype) / vae.config.scaling_factor, return_dict=False
                    )[0].float()
                    decoded_reference = vae.decode(
                        reference.to(vae.dtype) / vae.config.scaling_factor, return_dict=False
                    )[0].float()
                pixel_background = F.interpolate(
                    background.float(), decoded_output.shape[-2:], mode="bilinear", align_corners=False
                ).clamp(0, 1)
                pixel_foreground = 1.0 - pixel_background
                loss = loss + training_config.pixel_background_weight * _masked_l1(
                    decoded_output, decoded_target, pixel_background
                )
                loss = loss + training_config.pixel_foreground_weight * _masked_l1(
                    decoded_output, decoded_reference, pixel_foreground
                )
                loss = loss + training_config.detail_weight * latent_gradient_loss(output, target)
                scaled_loss = loss / training_config.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            if step % training_config.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([
                    *electronic_parameters, *optical_parameters,
                    *adapter.parameters(), *scene_router.parameters(),
                ], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(reference)
            count += len(reference)

        validation = evaluate_scene_model(
            unet, adapter, scene_router, val_loader, sigma, device,
            residual_scale=training_config.residual_scale,
            noise_scale=training_config.noise_scale,
        )
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "validation": validation,
            "optical_alpha": float(optical.fusion.alpha.detach()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["latent_mse"] < best_value:
            best_value = validation["latent_mse"]
            best_epoch = epoch
            torch.save({
                "schema_version": 1,
                "epoch": epoch,
                "model_config": asdict(model_config),
                "training_config": asdict(training_config),
                "unet": {key: value.detach().half().cpu() for key, value in unet.state_dict().items()},
                "adapter": {key: value.detach().half().cpu() for key, value in adapter.state_dict().items()},
                "scene_router": {
                    key: value.detach().half().cpu() for key, value in scene_router.state_dict().items()
                },
                "validation": validation,
            }, output_dir / "best_model.pt")
        if epoch % training_config.sample_every_epochs == 0:
            _sample_grid(
                unet=unet, adapter=adapter, vae=vae, sigma=sigma,
                latent_dataset=val, raw_dataset=raw_val,
                output=output_dir / "samples" / f"epoch_{epoch:03d}.jpg",
                device=device, residual_scale=training_config.residual_scale,
                noise_scale=training_config.noise_scale, seed=seed + epoch,
            )

    report_architecture = architecture_report(
        unet, vae, adapter, scene_router, optical, model_config
    )
    report_architecture["text_region_router_parameters"] = 0
    report_architecture["text_scene_router_parameters"] = sum(
        p.numel() for p in scene_router.parameters()
    )
    report_architecture["generation_tail_parameters"] = (
        report_architecture["unet_parameters"] + report_architecture["vae_decoder_parameters"]
        + report_architecture["qwen_condition_adapter_parameters"]
        + report_architecture["text_scene_router_parameters"]
    )
    report_architecture["trainable_parameters"] = (
        sum(p.numel() for p in unet.parameters() if p.requires_grad)
        + sum(p.numel() for p in adapter.parameters())
        + sum(p.numel() for p in scene_router.parameters())
    )
    report_architecture["object_preservation"] = (
        "exact source RGB is alpha-composited over the generated background after VAE decode"
    )
    report = {
        "schema_version": 1,
        "task": "white-background ABO lamp + Qwen instruction -> new styled scene",
        "dataset": "ABO CleanRender CC BY 4.0 + generated backgrounds",
        "categories": ["lamp"],
        "scenes": [scene["id"] for scene in SCENES],
        "initial_validation": initial,
        "best_epoch": best_epoch,
        "best_validation_latent_mse": best_value,
        "architecture": report_architecture,
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "gan_used": False,
        "baseline_contract": "Qwen + decoder only",
        "warm_start_checkpoint": str(warm_start_checkpoint) if warm_start_checkpoint else None,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    del unet, vae, adapter, scene_router, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


__all__ = [
    "SceneLatentDataset", "SceneTrainingConfig", "TextSceneRouter", "cache_scene_latents",
    "evaluate_scene_model", "load_scene_config", "train_scene_model",
]
