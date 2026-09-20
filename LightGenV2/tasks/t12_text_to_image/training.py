"""Training loop for the paired LightGen and Qwen+VAE baseline profiles."""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CachedLatentDataset
from .losses import conditional_vae_loss
from .modeling import TextConditionedVAE, build_model
from .settings import Settings


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(settings: Settings, split: str, shuffle: bool) -> DataLoader:
    return DataLoader(
        CachedLatentDataset(settings.cache_dir / f"{split}.pt"),
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=settings.num_workers > 0,
        generator=torch.Generator().manual_seed(settings.seed),
    )


def _beta(settings: Settings, epoch: int) -> float:
    if settings.kl_warmup_epochs <= 0:
        return settings.kl_weight
    return settings.kl_weight * min(1.0, float(epoch + 1) / settings.kl_warmup_epochs)


def _epoch(
    model: TextConditionedVAE,
    loader: DataLoader,
    device: torch.device,
    settings: Settings,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    samples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            text = batch["text"].to(device, non_blocking=True)
            latent = batch["latent"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=settings.amp and device.type == "cuda"):
                output = model.forward_train(text, latent)
                loss, metrics = conditional_vae_loss(
                    output, latent, kl_weight=_beta(settings, epoch), free_bits=settings.free_bits
                )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = len(text)
            samples += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / samples for key, value in totals.items()}


def train(settings: Settings, device: torch.device) -> dict[str, Any]:
    seed_everything(settings.seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [settings.output_dir / name for name in ("best_checkpoint.pt", "last_checkpoint.pt")]
    if any(path.exists() for path in existing):
        raise FileExistsError("Training checkpoints already exist; choose a new --run-dir to preserve the run")
    model = build_model(settings, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    train_loader = _loader(settings, "train", True)
    val_loader = _loader(settings, "val", False)
    best = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(settings.epochs):
        train_metrics = _epoch(model, train_loader, device, settings, epoch, optimizer)
        val_metrics = _epoch(model, val_loader, device, settings, epoch, None)
        row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        payload = {
            "schema_version": 1,
            "epoch": epoch + 1,
            "variant": settings.variant,
            "settings": settings.to_dict(),
            "architecture": model.architecture_report(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val": val_metrics,
        }
        torch.save(payload, settings.output_dir / "last_checkpoint.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(payload, settings.output_dir / "best_checkpoint.pt")
    (settings.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    report = {"best_val_loss": best, "epochs": settings.epochs, "architecture": model.architecture_report()}
    (settings.output_dir / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["seed_everything", "train"]
