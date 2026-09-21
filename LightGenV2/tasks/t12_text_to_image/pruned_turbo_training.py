"""Recover a structurally pruned compact Turbo UNet by latent distillation."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .compact_turbo import CompactTurboConfig, latent_gradient_loss, one_step_denoise
from .compact_turbo_training import QwenLatentCacheDataset, _evaluate, _qwen_condition_cache
from .pruned_turbo import DEFAULT_PRUNE_SPEC, apply_attention_pruning, save_pruned_unet


def train_pruned_turbo(
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
    scheduler.set_timesteps(1, device=device); sigma = scheduler.sigmas[0]
    unet = UNet2DConditionModel.from_pretrained(
        initial_unet, variant="fp16", torch_dtype=torch.float32, local_files_only=True
    )
    original_parameters = sum(parameter.numel() for parameter in unet.parameters())
    prune_report = apply_attention_pruning(unet, DEFAULT_PRUNE_SPEC)
    unet = unet.to(device)
    unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    initial = _evaluate(unet, val_loader, sigma, device)
    print(json.dumps({"epoch": 0, "validation": initial, "pruning": prune_report}), flush=True)
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
        history.append(row); print(json.dumps(row), flush=True)
        if validation["normalized_mse"] < best:
            best = validation["normalized_mse"]
            save_pruned_unet(
                unet, output_dir / "best_unet", base_unet=initial_unet,
                prune_report=prune_report, metadata={"epoch": epoch, "validation": validation},
            )
            (output_dir / "best.json").write_text(json.dumps(row, indent=2) + "\n")
    report = {
        "schema_version": 1,
        "variant": "qwen_bksdm_v2_tiny_pruned_deep_up_attention",
        "initial_validation": initial,
        "best_normalized_mse": best,
        "train_samples": len(train),
        "validation_samples": len(val),
        "original_unet_parameters": original_parameters,
        "pruned_unet_parameters": prune_report["remaining_parameters"],
        "removed_parameters": prune_report["removed_parameters"],
        "reduction_from_tiny": prune_report["removed_parameters"] / original_parameters,
        "inference_iterations": 1,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    del unet
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


__all__ = ["train_pruned_turbo"]
