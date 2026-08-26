from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .io_utils import write_csv, write_json
from .modeling import RobustRawCCDMNIST4D2NN, ShiftMap
from .settings import V2Settings


def _batch_metrics(
    model: RobustRawCCDMNIST4D2NN,
    output: dict[str, torch.Tensor | ShiftMap],
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]:
    loss, target_loss, background_loss = model.raw_ccd_loss(output, targets)
    detector_energy = output["detector_energy"]
    ccd_intensity = output["ccd_intensity"]
    if not isinstance(detector_energy, torch.Tensor) or not isinstance(
        ccd_intensity, torch.Tensor
    ):
        raise TypeError("Model output tensors are missing")
    predictions = detector_energy.argmax(dim=1)
    rows = torch.arange(len(targets), device=targets.device)
    target_energy = detector_energy[rows, targets]
    target_mask = model.detector_masks[targets]
    background_energy = (ccd_intensity * (1.0 - target_mask)).sum(dim=(-2, -1))
    return (
        loss,
        target_loss,
        background_loss,
        int((predictions == targets).sum()),
        float(target_energy.detach().sum()),
        float(background_energy.detach().sum()),
    )


def train_epoch(
    model: RobustRawCCDMNIST4D2NN,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    settings: V2Settings,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "target_region_mse": 0.0,
        "background_mse": 0.0,
        "correct": 0.0,
        "target_energy": 0.0,
        "background_energy": 0.0,
        "samples": 0.0,
        "grad_rms": 0.0,
        "steps": 0.0,
    }
    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss, target_loss, background_loss, correct, target_energy, background_energy = (
            _batch_metrics(model, output, targets)
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at batch {step}")
        loss.backward()
        gradient = model.raw_phase.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"Missing or non-finite raw_phase gradient at batch {step}")
        totals["grad_rms"] += float(gradient.square().mean().sqrt())
        totals["steps"] += 1
        torch.nn.utils.clip_grad_norm_([model.raw_phase], settings.gradient_clip_norm)
        optimizer.step()
        batch = len(targets)
        totals["loss"] += float(loss.detach()) * batch
        totals["target_region_mse"] += float(target_loss.detach()) * batch
        totals["background_mse"] += float(background_loss.detach()) * batch
        totals["correct"] += correct
        totals["target_energy"] += target_energy
        totals["background_energy"] += background_energy
        totals["samples"] += batch
        if settings.log_interval_batches > 0 and step % settings.log_interval_batches == 0:
            print(
                f"  batch={step}/{len(loader)} "
                f"loss={totals['loss']/totals['samples']:.5f} "
                f"acc={totals['correct']/totals['samples']:.4f}",
                flush=True,
            )
    count = totals["samples"]
    return {
        "loss": totals["loss"] / count,
        "target_region_mse": totals["target_region_mse"] / count,
        "background_mse": totals["background_mse"] / count,
        "accuracy": totals["correct"] / count,
        "target_energy": totals["target_energy"] / count,
        "background_energy": totals["background_energy"] / count,
        "phase_grad_rms": totals["grad_rms"] / max(1.0, totals["steps"]),
    }


@torch.no_grad()
def evaluate(
    model: RobustRawCCDMNIST4D2NN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals = {
        "loss": 0.0,
        "target_region_mse": 0.0,
        "background_mse": 0.0,
        "correct": 0.0,
        "target_energy": 0.0,
        "background_energy": 0.0,
        "samples": 0.0,
    }
    confusion = torch.zeros(4, 4, dtype=torch.long)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output = model(images)
        loss, target_loss, background_loss, correct, target_energy, background_energy = (
            _batch_metrics(model, output, targets)
        )
        detector_energy = output["detector_energy"]
        if not isinstance(detector_energy, torch.Tensor):
            raise TypeError("detector_energy must be a tensor")
        predictions = detector_energy.argmax(dim=1)
        for target, prediction in zip(targets.cpu(), predictions.cpu()):
            confusion[int(target), int(prediction)] += 1
        batch = len(targets)
        totals["loss"] += float(loss) * batch
        totals["target_region_mse"] += float(target_loss) * batch
        totals["background_mse"] += float(background_loss) * batch
        totals["correct"] += correct
        totals["target_energy"] += target_energy
        totals["background_energy"] += background_energy
        totals["samples"] += batch
    count = totals["samples"]
    return {
        "loss": totals["loss"] / count,
        "target_region_mse": totals["target_region_mse"] / count,
        "background_mse": totals["background_mse"] / count,
        "accuracy": totals["correct"] / count,
        "target_energy": totals["target_energy"] / count,
        "background_energy": totals["background_energy"] / count,
        "samples": int(count),
        "confusion_matrix": confusion.tolist(),
    }


def save_checkpoint(
    path: Path,
    model: RobustRawCCDMNIST4D2NN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    settings: V2Settings,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "settings": settings.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path, model: RobustRawCCDMNIST4D2NN, device: torch.device
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MNIST4 v2 checkpoint is missing: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def save_phase_preview(path: Path, model: RobustRawCCDMNIST4D2NN) -> None:
    phase = model.phase().detach().cpu().numpy()
    encoded = np.rint(np.mod(phase, 2.0 * math.pi) * 255.0 / (2.0 * math.pi))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded.astype(np.uint8), mode="L").save(path)


def train_model(
    model: RobustRawCCDMNIST4D2NN,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    settings: V2Settings,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(
        [model.raw_phase], lr=settings.phase_learning_rate, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.epochs, eta_min=settings.min_learning_rate
    )
    save_phase_preview(settings.output_dir / "phase_initial.png", model)
    rows: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, settings.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, settings, device)
        validation_metrics = evaluate(model, validation_loader, device)
        phase_stats = model.phase_statistics()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
                if key not in {"confusion_matrix", "samples"}
            },
            **phase_stats,
        }
        rows.append(row)
        write_csv(settings.output_dir / "metrics" / "training_log.csv", rows)
        save_checkpoint(
            settings.output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            epoch,
            row,
            settings,
        )
        improved = validation_metrics["accuracy"] > best_accuracy or (
            validation_metrics["accuracy"] == best_accuracy
            and validation_metrics["loss"] < best_loss
        )
        if improved:
            best_accuracy = float(validation_metrics["accuracy"])
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            save_checkpoint(
                settings.output_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                row,
                settings,
            )
            save_phase_preview(settings.output_dir / "phase_best.png", model)
        print(
            f"epoch {epoch:03d} train_loss={train_metrics['loss']:.5f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"phase_std={phase_stats['phase_std_rad']:.4f}rad "
            f"grad={train_metrics['phase_grad_rms']:.3e}",
            flush=True,
        )
        scheduler.step()
    load_checkpoint(settings.output_dir / "checkpoints" / "best.pt", model, device)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
        "phase_statistics": model.phase_statistics(),
    }
    write_json(settings.output_dir / "metrics" / "training_summary.json", summary)
    return summary
