"""Fine-tune a block-pruned UNet student on real Qwen conditions."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .compact_turbo import CompactTurboConfig, latent_gradient_loss, one_step_denoise
from .electronic_turbo_infer import _load_adapter


def _unique_caption_indices(data_dir: Path) -> list[int]:
    rows = [json.loads(line) for line in (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    seen: set[str] = set()
    indices = []
    for index, row in enumerate(rows):
        if row["caption"] not in seen:
            seen.add(row["caption"])
            indices.append(index)
    return indices


@torch.inference_mode()
def _qwen_condition_cache(
    adapter_checkpoint: Path, feature_cache_dir: Path, data_dir: Path, device: torch.device
) -> dict[str, torch.Tensor]:
    adapter, _ = _load_adapter(adapter_checkpoint, device)
    result = {}
    for split in ("train", "val"):
        payload = torch.load(
            feature_cache_dir / f"{split}.pt", map_location="cpu", weights_only=False, mmap=True
        )
        text = payload["text"].float()
        if split == "train":
            text = text[_unique_caption_indices(data_dir)]
        chunks = [
            adapter.condition(text[start : start + 32].to(device)).half().cpu()
            for start in range(0, len(text), 32)
        ]
        result[split] = torch.cat(chunks)
    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


class QwenLatentCacheDataset(Dataset):
    def __init__(self, cache_path: Path, conditions: torch.Tensor, *, restrict_prompts: bool) -> None:
        self.payload = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
        self.conditions = conditions
        if restrict_prompts:
            self.indices = torch.nonzero(self.payload["prompt_indices"] < len(conditions)).flatten()
        else:
            self.indices = torch.arange(len(self.payload["target"]))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = int(self.indices[index])
        prompt = int(self.payload["prompt_indices"][sample])
        return {
            "noise": self.payload["noise"][sample].float(),
            "target": self.payload["target"][sample].float(),
            "condition": self.conditions[prompt].float(),
        }


@torch.inference_mode()
def _evaluate(unet, loader: DataLoader, sigma: torch.Tensor, device: torch.device) -> dict[str, float]:
    unet.eval()
    squared = cosine = target_sum = target_square = pixels = samples = 0.0
    for batch in loader:
        noise = batch["noise"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = one_step_denoise(unet, noise, condition, sigma).float()
        squared += float(F.mse_loss(output, target, reduction="sum"))
        cosine += float(F.cosine_similarity(output.flatten(1), target.flatten(1)).sum())
        target_sum += float(target.sum())
        target_square += float(target.square().sum())
        pixels += target.numel()
        samples += len(target)
    mean = target_sum / pixels
    variance = target_square / pixels - mean * mean
    mse = squared / pixels
    return {"normalized_mse": mse / variance, "cosine": cosine / samples, "mse": mse}


def train_compact_turbo(
    *,
    initial_unet: Path,
    turbo_checkpoint: Path,
    latent_cache_dir: Path,
    adapter_checkpoint: Path,
    feature_cache_dir: Path,
    data_dir: Path,
    output_dir: Path,
    config: CompactTurboConfig,
    device: torch.device,
    seed: int,
) -> dict:
    """Refine a compact UNet against cached one-step SD-Turbo latents."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    from diffusers import EulerDiscreteScheduler, UNet2DConditionModel

    conditions = _qwen_condition_cache(adapter_checkpoint, feature_cache_dir, data_dir, device)
    train = QwenLatentCacheDataset(
        latent_cache_dir / "train.pt", conditions["train"], restrict_prompts=True
    )
    val = QwenLatentCacheDataset(
        latent_cache_dir / "val.pt", conditions["val"], restrict_prompts=False
    )
    train_loader = DataLoader(
        train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val, batch_size=min(config.batch_size, 8), shuffle=False,
        num_workers=config.num_workers, pin_memory=device.type == "cuda",
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        turbo_checkpoint, subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(1, device=device)
    sigma = scheduler.sigmas[0]
    unet = UNet2DConditionModel.from_pretrained(
        initial_unet, variant="fp16", torch_dtype=torch.float32, local_files_only=True
    ).to(device)
    unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    initial = _evaluate(unet, val_loader, sigma, device)
    best = math.inf
    history = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        unet.train(); total = count = 0.0
        for batch in train_loader:
            noise = batch["noise"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            condition = batch["condition"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = one_step_denoise(unet, noise, condition, sigma).float()
                loss = F.mse_loss(output, target)
                loss = loss + config.detail_weight * latent_gradient_loss(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(target); count += len(target)
        validation = _evaluate(unet, val_loader, sigma, device)
        row = {"epoch": epoch, "train_loss": total / count, "validation": validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["normalized_mse"] < best:
            best = validation["normalized_mse"]
            unet.save_pretrained(output_dir / "best_unet", safe_serialization=True, variant="fp16")
            (output_dir / "best.json").write_text(json.dumps(row, indent=2) + "\n")
    report = {
        "schema_version": 1,
        "variant": "qwen_bksdm_v2_tiny_one_step",
        "initial_validation": initial,
        "best_normalized_mse": best,
        "train_samples": len(train),
        "validation_samples": len(val),
        "unet_parameters": sum(parameter.numel() for parameter in unet.parameters()),
        "inference_iterations": 1,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    del unet
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


__all__ = ["QwenLatentCacheDataset", "train_compact_turbo"]
