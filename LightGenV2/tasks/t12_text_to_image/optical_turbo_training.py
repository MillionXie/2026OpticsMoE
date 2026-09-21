"""Train the V0 parallel optical mid block against cached one-step targets."""

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

from .compact_turbo import latent_gradient_loss, one_step_denoise
from .optical_turbo import OpticalTurboConfig, attach_parallel_optical_mid


def _unique_caption_indices(data_dir: Path) -> list[int]:
    rows = [json.loads(line) for line in (data_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    seen: set[str] = set()
    result = []
    for index, row in enumerate(rows):
        if row["caption"] not in seen:
            seen.add(row["caption"])
            result.append(index)
    return result


class NativeConditionLatentDataset(Dataset):
    def __init__(
        self, split: str, latent_cache_dir: Path, condition_cache: Path, data_dir: Path
    ) -> None:
        self.payload = torch.load(
            latent_cache_dir / f"{split}.pt", map_location="cpu", weights_only=False, mmap=True
        )
        teacher = torch.load(condition_cache, map_location="cpu", weights_only=False, mmap=True)
        if split == "train":
            self.conditions = torch.cat((
                teacher["train"]["condition"][_unique_caption_indices(data_dir)],
                teacher["synthetic"]["condition"],
            ))
        else:
            self.conditions = teacher["val"]["condition"]

    def __len__(self) -> int:
        return len(self.payload["target"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        prompt = int(self.payload["prompt_indices"][index])
        return {
            "noise": self.payload["noise"][index].float(),
            "target": self.payload["target"][index].float(),
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


def train_optical_turbo(
    *,
    compact_unet: Path,
    turbo_checkpoint: Path,
    latent_cache_dir: Path,
    condition_cache: Path,
    data_dir: Path,
    output_dir: Path,
    config: OpticalTurboConfig,
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

    scheduler = EulerDiscreteScheduler.from_pretrained(
        turbo_checkpoint, subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(1, device=device)
    sigma = scheduler.sigmas[0]
    unet = UNet2DConditionModel.from_pretrained(
        compact_unet, variant="fp16", torch_dtype=torch.float32, local_files_only=True
    ).to(device)
    unet.requires_grad_(False)
    wrapper = attach_parallel_optical_mid(unet, config).to(device)
    wrapper.freeze_electronic()
    train = NativeConditionLatentDataset("train", latent_cache_dir, condition_cache, data_dir)
    val = NativeConditionLatentDataset("val", latent_cache_dir, condition_cache, data_dir)
    train_loader = DataLoader(
        train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val, batch_size=min(config.batch_size, 8), shuffle=False,
        num_workers=config.num_workers, pin_memory=device.type == "cuda",
    )
    phase, other = [], []
    for name, parameter in wrapper.optical_parameters():
        (phase if "phase" in name else other).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": other, "lr": config.learning_rate},
        {"params": phase, "lr": config.phase_learning_rate},
    ], weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    initial = _evaluate(unet, val_loader, sigma, device)
    print(json.dumps({"epoch": 0, "validation": initial, "alpha": float(wrapper.fusion.alpha)}), flush=True)
    best = math.inf
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        unet.eval(); wrapper.train(); wrapper.electronic.eval(); total = count = 0.0
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
            torch.nn.utils.clip_grad_norm_(other + phase, 1.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(target); count += len(target)
        validation = _evaluate(unet, val_loader, sigma, device)
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "validation": validation,
            "alpha": float(wrapper.fusion.alpha.detach()),
            "expert_gate": float(torch.sigmoid(wrapper.expert_gate.detach())),
            "global_gate": float(torch.sigmoid(wrapper.global_gate.detach())),
            "output_gate": float(torch.sigmoid(wrapper.output_gate.detach())),
        }
        history.append(row); print(json.dumps(row), flush=True)
        if validation["normalized_mse"] < best:
            best = validation["normalized_mse"]
            best_state = wrapper.optical_state_dict()
            payload = {
                "schema_version": 1,
                "variant": "qwen_bksdm_v2_tiny_parallel_optical_mid_v0",
                "epoch": epoch,
                "config": config.__dict__,
                "compact_unet": str(compact_unet),
                "optical_state": best_state,
                "validation": validation,
                "architecture": wrapper.architecture_report(),
            }
            torch.save(payload, output_dir / "best_optical_mid.pt")
            (output_dir / "best.json").write_text(json.dumps(row, indent=2) + "\n")
    if best_state is None:
        raise RuntimeError("No optical checkpoint was selected")
    wrapper.load_optical_state_dict(best_state)
    report = {
        "schema_version": 1,
        "variant": "qwen_bksdm_v2_tiny_parallel_optical_mid_v0",
        "initial_validation": initial,
        "best_normalized_mse": best,
        "train_samples": len(train),
        "validation_samples": len(val),
        "architecture": wrapper.architecture_report(),
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    del unet
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


__all__ = ["NativeConditionLatentDataset", "train_optical_turbo"]
