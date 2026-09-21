"""Train and evaluate the compact one-pass latent student."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .distilled_decoder import DistilledDecoderConfig, QwenOneStepLatentStudent


class LatentDistillationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: Path) -> None:
        self.payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        required = {"qwen", "noise", "target", "meta"}
        if not required.issubset(self.payload):
            raise ValueError(f"Invalid latent distillation cache: {path}")

    def __len__(self) -> int:
        return len(self.payload["qwen"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "qwen": self.payload["qwen"][index].float(),
            "noise": self.payload["noise"][index].float(),
            "target": self.payload["target"][index].float(),
        }


def _loss(
    prediction: torch.Tensor, target: torch.Tensor, low_frequency_weight: float
) -> tuple[torch.Tensor, dict[str, float]]:
    latent = F.mse_loss(prediction, target)
    low = F.mse_loss(F.avg_pool2d(prediction, 4), F.avg_pool2d(target, 4))
    total = latent + low_frequency_weight * low
    return total, {"latent_mse": float(latent.detach()), "low_frequency_mse": float(low.detach())}


@torch.inference_mode()
def _evaluate(
    model: QwenOneStepLatentStudent, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_mse = total_cosine = target_sum = target_square_sum = pixels = samples = 0.0
    for batch in loader:
        noise = batch["noise"].to(device)
        target = batch["target"].to(device)
        prediction = model(noise, batch["qwen"].to(device))
        total_mse += float(F.mse_loss(prediction, target, reduction="sum"))
        total_cosine += float(F.cosine_similarity(prediction.flatten(1), target.flatten(1)).sum())
        target_sum += float(target.sum())
        target_square_sum += float((target * target).sum())
        pixels += target.numel()
        samples += len(target)
    target_mean = target_sum / pixels
    target_variance = max(target_square_sum / pixels - target_mean * target_mean, 1e-8)
    mse = total_mse / pixels
    return {
        "latent_mse": mse,
        "latent_normalized_mse": mse / target_variance,
        "latent_cosine": total_cosine / samples,
    }


def _tensor_to_pil(value: torch.Tensor) -> Image.Image:
    array = ((value.detach().float().clamp(-1, 1) + 1) * 127.5).byte()
    array = array.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


@torch.inference_mode()
def _write_comparison_grid(
    model: QwenOneStepLatentStudent,
    dataset: LatentDistillationDataset,
    turbo_checkpoint: Path,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    from diffusers import AutoencoderKL

    model.eval()
    chosen = [0, 4, 8, 12]
    qwen = torch.stack([dataset[index]["qwen"] for index in chosen]).to(device)
    noise = torch.stack([dataset[index]["noise"] for index in chosen]).to(device)
    target = torch.stack([dataset[index]["target"] for index in chosen]).to(device)
    prediction = model(noise, qwen)
    vae = AutoencoderKL.from_pretrained(
        turbo_checkpoint,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    target_images = vae.decode(
        target.to(vae.dtype) / vae.config.scaling_factor, return_dict=False
    )[0]
    predicted_images = vae.decode(
        prediction.to(vae.dtype) / vae.config.scaling_factor, return_dict=False
    )[0]
    cell, label = 512, 80
    canvas = Image.new("RGB", (label + 4 * cell, 2 * cell), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, cell // 2), "teacher", fill="black")
    draw.text((8, cell + cell // 2), "student", fill="black")
    for column, image in enumerate(target_images):
        canvas.paste(_tensor_to_pil(image), (label + column * cell, 0))
    for column, image in enumerate(predicted_images):
        canvas.paste(_tensor_to_pil(image), (label + column * cell, cell))
    canvas.save(output)
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"file": output.name, "columns": len(chosen), "rows": ["teacher", "student"]}


def train_distilled_decoder(
    cache_dir: Path,
    config: DistilledDecoderConfig,
    turbo_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    seed: int = 42,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_data = LatentDistillationDataset(cache_dir / "train.pt")
    val_data = LatentDistillationDataset(cache_dir / "val.pt")
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    text_dim = int(train_data.payload["qwen"].shape[1])
    model = QwenOneStepLatentStudent(text_dim, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_mse = 0, math.inf
    history = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = total_samples = 0.0
        for batch in train_loader:
            qwen = batch["qwen"].to(device, non_blocking=True)
            noise = batch["noise"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(noise, qwen)
                loss, _ = _loss(prediction, target, config.low_frequency_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(target)
            total_samples += len(target)
        scheduler.step()
        validation = _evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_samples,
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["latent_normalized_mse"] < best_mse:
            best_mse = validation["latent_normalized_mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    payload = {
        "schema_version": 1,
        "variant": "qwen_one_step_latent_student",
        "epoch": best_epoch,
        "seed": seed,
        "config": config.to_dict(),
        "text_dim": text_dim,
        "architecture": model.architecture_report(),
        "model": model.state_dict(),
        "best_validation_normalized_mse": best_mse,
        "init_noise_sigma": float(train_data.payload["meta"]["init_noise_sigma"]),
        "inference_flow": "Qwen text + seeded Gaussian latent -> compact student once -> VAE decode once",
    }
    torch.save(payload, output_dir / "best_student.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    sample = _write_comparison_grid(
        model, val_data, turbo_checkpoint, output_dir / "teacher_student_grid.png", device
    )
    report = {
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best_mse,
        "architecture": model.architecture_report(),
        "training_seconds": time.perf_counter() - started,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "sample": sample,
        "deployment": "frozen Qwen + compact student + VAE decoder; no PCA, CLIP, or SD-Turbo UNet",
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["LatentDistillationDataset", "train_distilled_decoder"]
