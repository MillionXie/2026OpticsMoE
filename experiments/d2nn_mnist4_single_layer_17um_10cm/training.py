from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .io_utils import write_csv, write_json
from .modeling import SingleLayerMNIST4D2NN
from .settings import Settings


def _batch_metrics(
    model: SingleLayerMNIST4D2NN,
    output: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]:
    loss, template_mse_loss, detector_ce_loss = model.optical_routing_loss(
        output, targets
    )
    predictions = output["detector_fraction"].argmax(dim=1)
    rows = torch.arange(len(targets), device=targets.device)
    target_fraction = output["detector_fraction"][rows, targets]
    capture_fraction = output["detector_fraction"].sum(dim=1)
    return (
        loss,
        template_mse_loss,
        detector_ce_loss,
        int((predictions == targets).sum()),
        float(target_fraction.sum()),
        float(capture_fraction.sum()),
    )


def train_epoch(
    model: SingleLayerMNIST4D2NN,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    settings: Settings,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_template_mse_loss = 0.0
    total_detector_ce_loss = 0.0
    total_correct = 0
    total_target_fraction = 0.0
    total_capture_fraction = 0.0
    total_count = 0
    grad_rms_sum = 0.0
    grad_steps = 0
    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss, template_mse_loss, detector_ce_loss, correct, target_sum, capture_sum = _batch_metrics(
            model, output, targets
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at batch {step}")
        loss.backward()
        gradient = model.raw_phase.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"Missing or non-finite raw_phase gradient at batch {step}")
        grad_rms_sum += float(gradient.square().mean().sqrt())
        grad_steps += 1
        torch.nn.utils.clip_grad_norm_(
            [model.raw_phase], settings.gradient_clip_norm
        )
        optimizer.step()
        batch = len(targets)
        total_loss += float(loss) * batch
        total_template_mse_loss += float(template_mse_loss) * batch
        total_detector_ce_loss += float(detector_ce_loss) * batch
        total_correct += correct
        total_target_fraction += target_sum
        total_capture_fraction += capture_sum
        total_count += batch
        if settings.log_interval_batches > 0 and step % settings.log_interval_batches == 0:
            print(
                f"  batch={step}/{len(loader)} loss={total_loss/total_count:.5f} "
                f"acc={total_correct/total_count:.4f}",
                flush=True,
            )
    return {
        "loss": total_loss / total_count,
        "template_mse_loss": total_template_mse_loss / total_count,
        "detector_ce_loss": total_detector_ce_loss / total_count,
        "accuracy": total_correct / total_count,
        "target_fraction": total_target_fraction / total_count,
        "capture_fraction": total_capture_fraction / total_count,
        "phase_grad_rms": grad_rms_sum / max(1, grad_steps),
    }


@torch.no_grad()
def evaluate(
    model: SingleLayerMNIST4D2NN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_template_mse_loss = 0.0
    total_detector_ce_loss = 0.0
    total_correct = 0
    total_target_fraction = 0.0
    total_capture_fraction = 0.0
    total_count = 0
    confusion = torch.zeros(4, 4, dtype=torch.long)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output = model(images)
        loss, template_mse_loss, detector_ce_loss, correct, target_sum, capture_sum = _batch_metrics(
            model, output, targets
        )
        predictions = output["detector_fraction"].argmax(dim=1)
        for target, prediction in zip(targets.cpu(), predictions.cpu()):
            confusion[int(target), int(prediction)] += 1
        batch = len(targets)
        total_loss += float(loss) * batch
        total_template_mse_loss += float(template_mse_loss) * batch
        total_detector_ce_loss += float(detector_ce_loss) * batch
        total_correct += correct
        total_target_fraction += target_sum
        total_capture_fraction += capture_sum
        total_count += batch
    return {
        "loss": total_loss / total_count,
        "template_mse_loss": total_template_mse_loss / total_count,
        "detector_ce_loss": total_detector_ce_loss / total_count,
        "accuracy": total_correct / total_count,
        "target_fraction": total_target_fraction / total_count,
        "capture_fraction": total_capture_fraction / total_count,
        "samples": total_count,
        "confusion_matrix": confusion.tolist(),
    }


def save_checkpoint(
    path: Path,
    model: SingleLayerMNIST4D2NN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    settings: Settings,
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
    path: Path, model: SingleLayerMNIST4D2NN, device: torch.device
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MNIST4 checkpoint is missing: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def save_phase_preview(path: Path, model: SingleLayerMNIST4D2NN) -> None:
    phase = model.phase().detach().cpu().numpy()
    encoded = np.rint(np.mod(phase, 2.0 * math.pi) * 255.0 / (2.0 * math.pi))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded.astype(np.uint8), mode="L").save(path)


def train_model(
    model: SingleLayerMNIST4D2NN,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    settings: Settings,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(
        [model.raw_phase], lr=settings.phase_learning_rate, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=settings.epochs,
        eta_min=settings.min_learning_rate,
    )
    save_phase_preview(settings.output_dir / "phase_initial.png", model)
    rows: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, settings.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, settings, device
        )
        validation_metrics = evaluate(model, validation_loader, device)
        phase_stats = model.phase_statistics()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_template_mse_loss": train_metrics["template_mse_loss"],
            "train_detector_ce_loss": train_metrics["detector_ce_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_target_fraction": train_metrics["target_fraction"],
            "train_capture_fraction": train_metrics["capture_fraction"],
            "phase_grad_rms": train_metrics["phase_grad_rms"],
            "validation_loss": validation_metrics["loss"],
            "validation_template_mse_loss": validation_metrics["template_mse_loss"],
            "validation_detector_ce_loss": validation_metrics["detector_ce_loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_target_fraction": validation_metrics["target_fraction"],
            "validation_capture_fraction": validation_metrics["capture_fraction"],
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
            f"raw_std={phase_stats['raw_phase_std']:.4f} "
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
