"""Cache and train the large one-step ABO product restoration model."""

from __future__ import annotations

import copy
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .compact_turbo import latent_gradient_loss
from .electronic_turbo_infer import _load_adapter
from .product_repair_data import ProductRepairDataset, SUPPORTED_CATEGORIES, write_pair_manifest
from .product_repair_model import (
    RepairModelConfig,
    architecture_report,
    attach_decoder_optics,
    expand_reference_conditioning,
    one_step_edit,
)


@dataclass(frozen=True)
class RepairTrainingConfig:
    image_size: int
    batch_size: int
    gradient_accumulation: int
    epochs: int
    learning_rate: float
    optical_learning_rate: float
    weight_decay: float
    detail_weight: float
    selected_weight: float
    preservation_weight: float
    adapter_learning_rate: float
    num_workers: int
    sample_every_epochs: int

    def validate(self) -> None:
        if min(
            self.image_size,
            self.batch_size,
            self.gradient_accumulation,
            self.epochs,
            self.learning_rate,
            self.optical_learning_rate,
            self.adapter_learning_rate,
            self.sample_every_epochs,
        ) <= 0:
            raise ValueError("Positive repair training settings are required")
        if min(
            self.weight_decay, self.detail_weight, self.selected_weight,
            self.preservation_weight, self.num_workers,
        ) < 0:
            raise ValueError("Non-negative regularization settings are required")


def load_repair_config(path: Path) -> tuple[RepairModelConfig, RepairTrainingConfig]:
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
    train_config = RepairTrainingConfig(**{
        key: (float(value) if key in {
            "learning_rate", "optical_learning_rate", "adapter_learning_rate",
            "weight_decay", "detail_weight", "selected_weight", "preservation_weight",
        } else int(value))
        for key, value in training.items()
    })
    model_config.validate()
    train_config.validate()
    return model_config, train_config


class RepairLatentDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path) -> None:
        self.payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def __len__(self) -> int:
        return len(self.payload["reference"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "reference": self.payload["reference"][index].float(),
            "target": self.payload["target"][index].float(),
            "selected_mask": self.payload["selected_mask"][index].float(),
            "distractor_mask": self.payload["distractor_mask"][index].float(),
            "qwen_text": self.payload["qwen_text"][index].float(),
            "sample_id": self.payload["sample_ids"][index],
            "prompt": self.payload["prompts"][index],
            "category": self.payload["categories"][index],
        }


@torch.inference_mode()
def cache_repair_latents(
    *,
    data_dir: Path,
    instruction_cache: Path,
    vae_checkpoint: Path,
    output_dir: Path,
    image_size: int,
    device: torch.device,
    seed: int = 42,
    batch_size: int = 8,
    num_workers: int = 4,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from diffusers import AutoencoderKL

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        vae_checkpoint, subfolder="vae", variant="fp16", torch_dtype=dtype, local_files_only=True
    ).to(device).eval().requires_grad_(False)
    scale = float(vae.config.scaling_factor)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "task": "text_selected_product_restoration",
        "image_size": image_size,
        "vae": str(vae_checkpoint),
        "latent_scaling_factor": scale,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        dataset = ProductRepairDataset(
            data_dir, split, image_size, instruction_cache, seed=seed
        )
        write_pair_manifest(dataset, output_dir / f"{split}_pairs.jsonl")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        payload: dict[str, list[Any]] = {
            "reference": [], "target": [], "selected_mask": [], "distractor_mask": [],
            "qwen_text": [], "sample_ids": [], "prompts": [], "categories": [],
        }
        for batch in loader:
            reference = batch["reference"].to(device=device, dtype=dtype, non_blocking=True)
            target = batch["target"].to(device=device, dtype=dtype, non_blocking=True)
            reference_latent = vae.encode(reference).latent_dist.mode() * scale
            target_latent = vae.encode(target).latent_dist.mode() * scale
            latent_size = reference_latent.shape[-2:]
            selected = F.interpolate(batch["selected_mask"].float(), latent_size, mode="area")
            distractor = F.interpolate(batch["distractor_mask"].float(), latent_size, mode="area")
            payload["reference"].append(reference_latent.half().cpu())
            payload["target"].append(target_latent.half().cpu())
            payload["selected_mask"].append(selected.half().cpu())
            payload["distractor_mask"].append(distractor.half().cpu())
            payload["qwen_text"].append(batch["qwen_text"].to(torch.bfloat16).cpu())
            for key, source_key in (
                ("sample_ids", "sample_id"), ("prompts", "prompt"), ("categories", "category")
            ):
                payload[key].extend(batch[source_key])
        packed = {
            key: torch.cat(value) if key in {
                "reference", "target", "selected_mask", "distractor_mask", "qwen_text"
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_l1(value: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((value - target).abs() * mask).sum() / (mask.sum() * value.shape[1]).clamp_min(1.0)


@torch.inference_mode()
def _evaluate(unet, adapter, loader, sigma, device) -> dict[str, float]:
    unet.eval()
    adapter.eval()
    totals = {"latent_mse": 0.0, "selected_l1": 0.0, "distractor_change_l1": 0.0,
              "reference_mse": 0.0}
    samples = 0
    generator = torch.Generator(device=device).manual_seed(7321)
    for batch in loader:
        reference = batch["reference"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        selected = batch["selected_mask"].to(device, non_blocking=True)
        distractor = batch["distractor_mask"].to(device, non_blocking=True)
        condition = adapter.condition(batch["qwen_text"].to(device))
        noise = torch.randn(reference.shape, generator=generator, device=device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = one_step_edit(unet, noise, reference, condition, sigma).float()
        count = len(reference)
        totals["latent_mse"] += float(F.mse_loss(output, target)) * count
        totals["reference_mse"] += float(F.mse_loss(reference, target)) * count
        totals["selected_l1"] += float(_masked_l1(output, target, selected)) * count
        totals["distractor_change_l1"] += float(_masked_l1(output, reference, distractor)) * count
        samples += count
    result = {key: value / samples for key, value in totals.items()}
    result["mse_improvement_over_copy"] = 1.0 - result["latent_mse"] / max(
        result["reference_mse"], 1e-8
    )
    return result


@torch.inference_mode()
def _sample_grid(
    *, unet, adapter, vae, sigma, latent_dataset: RepairLatentDataset,
    raw_dataset: ProductRepairDataset, output: Path, device: torch.device, seed: int,
) -> None:
    unet.eval()
    chosen = []
    for category in SUPPORTED_CATEGORIES:
        chosen.extend(
            index for index, value in enumerate(latent_dataset.payload["categories"])
            if value == category
        )
        chosen = chosen[: len(chosen) - max(0, len(chosen) % 2)]
    # Exactly two rows per category, preserving category order.
    chosen = [
        index
        for category in SUPPORTED_CATEGORIES
        for index in [
            i for i, value in enumerate(latent_dataset.payload["categories"]) if value == category
        ][:2]
    ]
    reference_latent = latent_dataset.payload["reference"][chosen].float().to(device)
    text = latent_dataset.payload["qwen_text"][chosen].float().to(device)
    condition = adapter.condition(text)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(reference_latent.shape, generator=generator, device=device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        output_latent = one_step_edit(unet, noise, reference_latent, condition, sigma)
        decoded = vae.decode(output_latent / vae.config.scaling_factor, return_dict=False)[0]
    decoded = decoded.float().clamp(-1, 1).cpu()
    cell, label_width = raw_dataset.image_size, 250
    canvas = Image.new("RGB", (label_width + 3 * cell, len(chosen) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, index in enumerate(chosen):
        raw = raw_dataset[index]
        for column, value in enumerate((raw["reference"], raw["target"], decoded[row_index])):
            array = value.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(array), (label_width + column * cell, row_index * cell))
        draw.text((4, row_index * cell + 4), raw["category"], fill="black")
        draw.text((4, row_index * cell + 26), raw["prompt"][:38], fill="black")
        draw.text((4, row_index * cell + 48), "input | target | generated", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def train_repair_model(
    *,
    initial_unet: Path,
    turbo_checkpoint: Path,
    latent_cache_dir: Path,
    data_dir: Path,
    instruction_cache: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    model_config: RepairModelConfig,
    training_config: RepairTrainingConfig,
    device: torch.device,
    seed: int = 42,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    _seed_everything(seed)
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    train = RepairLatentDataset(latent_cache_dir / "train.pt")
    val = RepairLatentDataset(latent_cache_dir / "val.pt")
    train_loader = DataLoader(
        train, batch_size=training_config.batch_size, shuffle=True,
        num_workers=training_config.num_workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val, batch_size=min(4, training_config.batch_size * 2), shuffle=False,
        num_workers=training_config.num_workers, pin_memory=device.type == "cuda",
    )
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    adapter.requires_grad_(True)
    unet = UNet2DConditionModel.from_pretrained(
        initial_unet, subfolder="unet", variant="fp16", torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    expand_reference_conditioning(unet)
    optical = attach_decoder_optics(unet, model_config)
    # The large pretrained decoder supplies useful multiscale features, while
    # a zero residual head makes the initial editor an exact copy operation.
    # Training then learns only the instruction-selected change.
    nn_init = torch.nn.init
    nn_init.zeros_(unet.conv_out.weight)
    if unet.conv_out.bias is not None:
        nn_init.zeros_(unet.conv_out.bias)
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
    raw_val = ProductRepairDataset(
        data_dir, "val", training_config.image_size, instruction_cache, seed=seed
    )
    initial = _evaluate(unet, adapter, val_loader, sigma, device)
    best_value = math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, training_config.epochs + 1):
        unet.train()
        adapter.train()
        total = 0.0
        count = 0
        for step, batch in enumerate(train_loader, 1):
            reference = batch["reference"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            condition = adapter.condition(batch["qwen_text"].to(device))
            selected = batch["selected_mask"].to(device, non_blocking=True)
            noise = torch.randn_like(reference)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = one_step_edit(unet, noise, reference, condition, sigma).float()
                global_loss = F.mse_loss(output, target)
                selected_loss = _masked_l1(output, target, selected)
                preservation_loss = _masked_l1(output, reference, 1.0 - selected)
                loss = global_loss + training_config.selected_weight * selected_loss
                loss = loss + training_config.preservation_weight * preservation_loss
                loss = loss + training_config.detail_weight * latent_gradient_loss(output, target)
                scaled_loss = loss / training_config.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            if step % training_config.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [*electronic_parameters, *optical_parameters, *adapter.parameters()], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(reference)
            count += len(reference)
        validation = _evaluate(unet, adapter, val_loader, sigma, device)
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
                "validation": validation,
            }, output_dir / "best_model.pt")
        if epoch % training_config.sample_every_epochs == 0:
            _sample_grid(
                unet=unet, adapter=adapter, vae=vae, sigma=sigma,
                latent_dataset=val, raw_dataset=raw_val,
                output=output_dir / "samples" / f"epoch_{epoch:03d}.jpg",
                device=device, seed=seed + epoch,
            )
    report_architecture = architecture_report(unet, vae, adapter, optical, model_config)
    report_architecture["trainable_parameters"] = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    report = {
        "schema_version": 1,
        "task": "input product image + text-selected local restoration -> new image",
        "dataset": "ABO CleanRender CC BY 4.0",
        "categories": list(SUPPORTED_CATEGORIES),
        "initial_validation": initial,
        "best_epoch": best_epoch,
        "best_validation_latent_mse": best_value,
        "architecture": report_architecture,
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "gan_used": False,
        "baseline_contract": "Qwen + decoder only",
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    del unet, vae, adapter, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


__all__ = [
    "RepairLatentDataset",
    "RepairTrainingConfig",
    "cache_repair_latents",
    "load_repair_config",
    "train_repair_model",
]
