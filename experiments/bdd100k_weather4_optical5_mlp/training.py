from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import DataBundle, make_loader
from .metrics import classification_metrics, write_history, write_json
from .visualization import save_detector_outputs, save_light_fields, save_phase_masks, save_training_curves


def train_model(model: nn.Module, data: DataBundle, settings: object, device: torch.device, output_dir: Path) -> list[dict[str, Any]]:
    train_loader = make_loader(data.train, settings.batch_size, settings.num_workers, True, settings.seed)
    validation_loader = make_loader(data.validation, settings.batch_size, settings.num_workers, False, settings.seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    best_score = -1.0
    diagnostic_batch = next(iter(validation_loader))[0][:8].to(device, non_blocking=True)
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter()
        model.train()
        model.set_epoch(epoch)
        train_loss = 0.0
        train_targets: list[int] = []
        train_predictions: list[int] = []
        for images, labels in _progress(train_loader, settings.progress, f"Train {epoch}/{settings.epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * labels.numel()
            train_targets.extend(labels.detach().cpu().tolist())
            train_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        validation = evaluate(model, validation_loader, device, data.class_names)
        train_metrics = classification_metrics(train_targets, train_predictions, data.class_names)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss / max(1, len(train_targets)),
            "train_top1_accuracy": train_metrics["top1_accuracy"],
            "validation_loss": validation["loss"],
            "validation_top1_accuracy": validation["top1_accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "validation_balanced_accuracy": validation["balanced_accuracy"],
            "phase_dropout_active": bool(settings.phase_dropout.enabled and epoch >= settings.phase_dropout.start_epoch),
            "epoch_time_sec": time.perf_counter() - started,
        }
        history.append(row)
        scheduler.step()
        checkpoint = _checkpoint(model, optimizer, scheduler, epoch, best_score, settings)
        torch.save(checkpoint, output_dir / "checkpoints" / "last.pt")
        if validation["macro_f1"] > best_score:
            best_score = float(validation["macro_f1"])
            checkpoint["best_validation_macro_f1"] = best_score
            torch.save(checkpoint, output_dir / "checkpoints" / "best.pt")
            write_json(output_dir / "metrics" / "best_validation.json", {"epoch": epoch, **validation})
        if epoch == 1 or epoch % settings.save_interval_epochs == 0 or epoch == settings.epochs:
            _save_diagnostics(model, diagnostic_batch, output_dir / "figures", epoch)
        write_history(output_dir / "metrics" / "training_history.csv", history)
        save_training_curves(history, output_dir / "figures" / "training_curves.png")
        print(
            f"[epoch {epoch:03d}] train_loss={row['train_loss']:.4f} "
            f"val_top1={row['validation_top1_accuracy']:.4f} val_macro_f1={row['validation_macro_f1']:.4f} "
            f"time={row['epoch_time_sec']:.1f}s"
        )
    return history


@torch.no_grad()
def evaluate(model: nn.Module, loader: Any, device: torch.device, class_names: list[str]) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, labels))
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    result = classification_metrics(targets, predictions, class_names)
    result["loss"] = total_loss / max(1, len(targets))
    return result


def load_best_checkpoint(model: nn.Module, output_dir: Path, device: torch.device) -> dict[str, Any]:
    path = output_dir / "checkpoints" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {path}. Run --phase train first.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _checkpoint(model: nn.Module, optimizer: Any, scheduler: Any, epoch: int, best_score: float, settings: object) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_validation_macro_f1": best_score,
        "phase_dropout": settings.regularization.get("phase_dropout", {}),
    }


@torch.no_grad()
def _save_diagnostics(model: nn.Module, images: torch.Tensor, figures_dir: Path, epoch: int) -> None:
    was_training = model.training
    model.eval()
    _, diagnostics = model(images, return_diagnostics=True)
    save_phase_masks(model, figures_dir, epoch)
    save_light_fields(diagnostics, figures_dir, epoch, sample_index=0)
    save_detector_outputs(diagnostics, figures_dir, epoch)
    model.train(was_training)


def _progress(loader: Any, enabled: bool, description: str):
    if not enabled:
        return loader
    try:
        from tqdm.auto import tqdm
        return tqdm(loader, desc=description)
    except ImportError:
        return loader
