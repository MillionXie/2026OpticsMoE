from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .datasets import DatasetBundle, make_loader
from .model import Ablation, OpticalClassifier
from .settings import Settings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_history(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def build_model(settings: Settings, device: torch.device) -> OpticalClassifier:
    return OpticalClassifier(settings.optical, settings.num_classes).to(device)


def build_optimizer(model: OpticalClassifier, settings: Settings) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = [
        {
            "params": list(model.phase_parameters()),
            "lr": settings.optimizer.phase_learning_rate,
            "name": "phase",
        },
        {
            "params": list(model.electronic_parameters()),
            "lr": settings.optimizer.electronic_learning_rate,
            "name": "electronic",
        },
    ]
    residual = list(model.residual_parameters())
    if residual:
        groups.append(
            {
                "params": residual,
                "lr": settings.optimizer.residual_learning_rate,
                "name": "residual",
            }
        )
    return torch.optim.AdamW(
        groups,
        betas=settings.optimizer.betas,
        eps=settings.optimizer.eps,
        weight_decay=settings.optimizer.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: Settings,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = settings.optimizer.warmup_epochs
    total = settings.training.epochs
    minimum = settings.optimizer.min_learning_rate_ratio

    def multiplier(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = float(epoch - warmup) / float(max(total - warmup - 1, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum + (1.0 - minimum) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _train_epoch(
    model: OpticalClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    settings: Settings,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started = time.perf_counter()
    use_amp = settings.training.use_amp and device.type == "cuda"
    maximum = settings.training.max_train_batches
    for batch_index, (images, targets) in enumerate(loader, start=1):
        if maximum is not None and batch_index > maximum:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets, label_smoothing=settings.training.label_smoothing)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.training.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        count = int(targets.numel())
        total_loss += float(loss.detach()) * count
        total_correct += int((logits.detach().argmax(dim=1) == targets).sum())
        total_samples += count
        if batch_index % settings.training.log_interval_batches == 0 or batch_index == len(loader):
            print(
                f"[train] epoch={epoch}/{settings.training.epochs} batch={batch_index}/{len(loader)} "
                f"loss={total_loss/max(total_samples,1):.5f} acc={total_correct/max(total_samples,1):.4f}",
                flush=True,
            )
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "samples": float(total_samples),
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate(
    model: OpticalClassifier,
    loader,
    device: torch.device,
    settings: Settings,
    *,
    ablation: Ablation = "normal",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    maximum = settings.training.max_evaluation_batches
    for batch_index, (images, targets) in enumerate(loader, start=1):
        if maximum is not None and batch_index > maximum:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images, ablation=ablation)
        total_loss += float(F.cross_entropy(logits, targets, reduction="sum"))
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total_samples += int(targets.numel())
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "samples": float(total_samples),
    }


@torch.no_grad()
def _diagnostics(
    model: OpticalClassifier,
    loader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    images, _ = next(iter(loader))
    _, stages = model(images[: min(8, len(images))].to(device), return_diagnostics=True)
    return {
        "optical_weights": model.optical_weights(),
        "stage_optical_rms": [float(row["optical_rms"].cpu()) for row in stages],
        "stage_skip_rms": [float(row["skip_rms"].cpu()) for row in stages],
        "stage_output_rms": [float(row["output_rms"].cpu()) for row in stages],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "phase_parameters": sum(parameter.numel() for parameter in model.phase_parameters()),
        "electronic_parameters": sum(parameter.numel() for parameter in model.electronic_parameters()),
    }


def _load_initial_checkpoint(model: OpticalClassifier, settings: Settings) -> dict[str, object] | None:
    path = settings.training.init_checkpoint
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if settings.training.load_backbone_only:
        state = {name: value for name, value in state.items() if name.startswith("stages.")}
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected backbone keys: {result.unexpected_keys}")
    else:
        model.load_state_dict(state, strict=True)
    return {"path": str(path), "load_backbone_only": settings.training.load_backbone_only}


def _checkpoint(
    model: OpticalClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    *,
    epoch: int,
    best_validation_accuracy: float,
    history: list[dict[str, object]],
    settings: Settings,
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "history": history,
        "settings_digest": settings.digest(),
    }


def train_seed(
    settings: Settings,
    datasets: DatasetBundle,
    device: torch.device,
    seed: int,
    *,
    force: bool = False,
) -> dict[str, object]:
    set_seed(seed)
    run_dir = settings.output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "resolved_config.json", asdict(settings))
    train_loader = make_loader(datasets.train, settings, train=True, seed=seed)
    validation_loader = make_loader(datasets.validation, settings, train=False, seed=seed + 1)
    test_loader = make_loader(datasets.test, settings, train=False, seed=seed + 2)
    model = build_model(settings, device)
    optimizer = build_optimizer(model, settings)
    scheduler = build_scheduler(optimizer, settings)
    scaler = torch.cuda.amp.GradScaler(enabled=settings.training.use_amp and device.type == "cuda")
    latest_path = run_dir / "latest.pt"
    start_epoch = 1
    best_accuracy = -1.0
    history: list[dict[str, object]] = []
    initialization: dict[str, object] | None = None
    if latest_path.exists() and not force:
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        if payload.get("settings_digest") != settings.digest():
            raise RuntimeError("Existing checkpoint was created with a different configuration; use --force")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload["epoch"]) + 1
        best_accuracy = float(payload["best_validation_accuracy"])
        history = list(payload["history"])
        initialization = {"resumed_from": str(latest_path)}
    else:
        initialization = _load_initial_checkpoint(model, settings)
    for epoch in range(start_epoch, settings.training.epochs + 1):
        train_metrics = _train_epoch(model, train_loader, optimizer, scaler, device, settings, epoch)
        validation_metrics = evaluate(model, validation_loader, device, settings)
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "epoch_seconds": train_metrics["seconds"],
            "phase_lr": optimizer.param_groups[0]["lr"],
            "electronic_lr": optimizer.param_groups[1]["lr"],
            "mean_optical_weight": float(np.mean(model.optical_weights())),
        }
        history.append(row)
        improved = float(validation_metrics["accuracy"]) > best_accuracy
        best_accuracy = max(best_accuracy, float(validation_metrics["accuracy"]))
        scheduler.step()
        payload = _checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            best_validation_accuracy=best_accuracy,
            history=history,
            settings=settings,
        )
        _atomic_torch_save(payload, latest_path)
        if improved:
            _atomic_torch_save(payload, run_dir / "best.pt")
        if epoch % settings.training.checkpoint_interval_epochs == 0:
            _atomic_torch_save(payload, run_dir / f"epoch_{epoch:03d}.pt")
        _write_history(run_dir / "history.csv", history)
        print(
            f"[epoch] {epoch}/{settings.training.epochs} train={train_metrics['accuracy']:.4f} "
            f"val={validation_metrics['accuracy']:.4f} best={best_accuracy:.4f} "
            f"optical_weight={row['mean_optical_weight']:.4f}",
            flush=True,
        )
    return evaluate_checkpoint(settings, datasets, device, seed, checkpoint="best.pt", initialization=initialization)


def evaluate_checkpoint(
    settings: Settings,
    datasets: DatasetBundle,
    device: torch.device,
    seed: int,
    *,
    checkpoint: str = "best.pt",
    initialization: dict[str, object] | None = None,
) -> dict[str, object]:
    run_dir = settings.output_dir / f"seed_{seed}"
    model = build_model(settings, device)
    payload = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    validation_loader = make_loader(datasets.validation, settings, train=False, seed=seed + 1)
    test_loader = make_loader(datasets.test, settings, train=False, seed=seed + 2)
    validation = evaluate(model, validation_loader, device, settings)
    ablations = {
        name: evaluate(model, test_loader, device, settings, ablation=name)
        for name in ("normal", "optical_off", "phase_random", "phase_shuffle")
    }
    full = float(ablations["normal"]["accuracy"])
    off = float(ablations["optical_off"]["accuracy"])
    chance = 1.0 / settings.num_classes
    dependence = (full - off) / max(full - chance, 1e-12)
    result: dict[str, object] = {
        "seed": seed,
        "checkpoint": str(run_dir / checkpoint),
        "selected_epoch": int(payload["epoch"]),
        "best_validation_accuracy": float(payload["best_validation_accuracy"]),
        "validation": validation,
        "test": ablations,
        "optical_dependence": {
            "absolute_full_minus_off": full - off,
            "normalized_fraction": dependence,
            "chance_accuracy": chance,
        },
        "diagnostics": _diagnostics(model, validation_loader, device),
        "initialization": initialization,
    }
    _write_json(run_dir / "evaluation.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def aggregate(settings: Settings) -> dict[str, object]:
    rows = []
    for seed in settings.training.seeds:
        path = settings.output_dir / f"seed_{seed}" / "evaluation.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError("No evaluation.json files found")
    accuracies = np.asarray([row["test"]["normal"]["accuracy"] for row in rows], dtype=float)
    off = np.asarray([row["test"]["optical_off"]["accuracy"] for row in rows], dtype=float)
    result = {
        "seeds": [row["seed"] for row in rows],
        "test_accuracy_mean": float(accuracies.mean()),
        "test_accuracy_std": float(accuracies.std(ddof=1)) if len(rows) > 1 else 0.0,
        "optical_off_accuracy_mean": float(off.mean()),
        "full_minus_off_mean": float((accuracies - off).mean()),
        "runs": rows,
    }
    _write_json(settings.output_dir / "aggregate.json", result)
    return result
