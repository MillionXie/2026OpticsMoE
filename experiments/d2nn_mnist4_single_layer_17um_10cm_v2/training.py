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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]:
    loss, target_loss, background_loss, detector_ce = model.raw_ccd_loss(
        output, targets
    )
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
        detector_ce,
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
        "detector_ce": 0.0,
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
        (
            loss,
            target_loss,
            background_loss,
            detector_ce,
            correct,
            target_energy,
            background_energy,
        ) = _batch_metrics(model, output, targets)
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
        totals["detector_ce"] += float(detector_ce.detach()) * batch
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
        "detector_ce": totals["detector_ce"] / count,
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
        "detector_ce": 0.0,
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
        (
            loss,
            target_loss,
            background_loss,
            detector_ce,
            correct,
            target_energy,
            background_energy,
        ) = _batch_metrics(model, output, targets)
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
        totals["detector_ce"] += float(detector_ce) * batch
        totals["correct"] += correct
        totals["target_energy"] += target_energy
        totals["background_energy"] += background_energy
        totals["samples"] += batch
    count = totals["samples"]
    return {
        "loss": totals["loss"] / count,
        "target_region_mse": totals["target_region_mse"] / count,
        "background_mse": totals["background_mse"] / count,
        "detector_ce": totals["detector_ce"] / count,
        "accuracy": totals["correct"] / count,
        "target_energy": totals["target_energy"] / count,
        "background_energy": totals["background_energy"] / count,
        "samples": int(count),
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def evaluate_robust_validation(
    model: RobustRawCCDMNIST4D2NN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    trials: int,
    seed: int,
) -> dict[str, float | int]:
    """Evaluate fixed-seed stochastic hardware perturbations on validation only.

    The function deliberately does not call ``evaluate`` because that function
    switches to eval mode and therefore disables the training-only optical
    perturbation sampler.  There are no train-mode electronic layers in this
    model; train mode here only activates the explicitly modelled hardware
    disturbances.
    """

    if trials <= 0:
        raise ValueError("Robust validation trials must be positive")
    previous_training = model.training
    previous_robustness = model.robustness_training_active
    device_indexes: list[int] = []
    if device.type == "cuda":
        device_indexes = [
            torch.cuda.current_device() if device.index is None else int(device.index)
        ]
    trial_accuracies: list[float] = []
    trial_losses: list[float] = []
    try:
        with torch.random.fork_rng(devices=device_indexes):
            for trial in range(int(trials)):
                trial_seed = int(seed) + trial
                torch.manual_seed(trial_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(trial_seed)
                model.train()
                model.set_robustness_training_active(True)
                correct = 0
                samples = 0
                loss_sum = 0.0
                for images, targets in loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    output = model(images)
                    loss, _, _, _, batch_correct, _, _ = _batch_metrics(
                        model, output, targets
                    )
                    batch = len(targets)
                    correct += batch_correct
                    samples += batch
                    loss_sum += float(loss) * batch
                trial_accuracies.append(correct / samples)
                trial_losses.append(loss_sum / samples)
    finally:
        model.set_robustness_training_active(previous_robustness)
        model.train(previous_training)
    return {
        "accuracy_mean": float(np.mean(trial_accuracies)),
        "accuracy_min": float(np.min(trial_accuracies)),
        "accuracy_std": float(np.std(trial_accuracies)),
        "loss_mean": float(np.mean(trial_losses)),
        "trials": int(trials),
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
    best_robust_accuracy = -1.0
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, settings.epochs + 1):
        robustness_active = (
            settings.robustness_enabled
            and epoch > settings.robustness_warmup_epochs
        )
        model.set_robustness_training_active(robustness_active)
        train_metrics = train_epoch(model, train_loader, optimizer, settings, device)
        validation_metrics = evaluate(model, validation_loader, device)
        robust_validation = (
            evaluate_robust_validation(
                model,
                validation_loader,
                device,
                trials=settings.robust_validation_trials,
                seed=settings.random_seed + 100_000,
            )
            if settings.robustness_enabled
            else {
                "accuracy_mean": validation_metrics["accuracy"],
                "accuracy_min": validation_metrics["accuracy"],
                "accuracy_std": 0.0,
                "loss_mean": validation_metrics["loss"],
                "trials": 1,
            }
        )
        phase_stats = model.phase_statistics()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "robustness_active": robustness_active,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
                if key not in {"confusion_matrix", "samples"}
            },
            **{f"robust_validation_{key}": value for key, value in robust_validation.items()},
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
        robust_accuracy = float(robust_validation["accuracy_mean"])
        improved = (
            robust_accuracy > best_robust_accuracy
            or (
                robust_accuracy == best_robust_accuracy
                and validation_metrics["accuracy"] > best_accuracy
            )
            or (
                robust_accuracy == best_robust_accuracy
                and validation_metrics["accuracy"] == best_accuracy
                and validation_metrics["loss"] < best_loss
            )
        )
        selection_eligible = bool(
            not settings.require_robust_update_for_selection or robustness_active
        )
        if improved and selection_eligible:
            best_robust_accuracy = robust_accuracy
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
            f"robust_val={robust_accuracy:.4f} "
            f"selection_eligible={'yes' if selection_eligible else 'baseline-only'} "
            f"robustness={'on' if robustness_active else 'warmup'} "
            f"phase_std={phase_stats['phase_std_rad']:.4f}rad "
            f"grad={train_metrics['phase_grad_rms']:.3e}",
            flush=True,
        )
        scheduler.step()
    load_checkpoint(settings.output_dir / "checkpoints" / "best.pt", model, device)
    summary = {
        "best_epoch": best_epoch,
        "selection_metric": "fixed_seed_robust_validation_accuracy_mean",
        "selection_requires_robust_update": (
            settings.require_robust_update_for_selection
        ),
        "best_robust_validation_accuracy": best_robust_accuracy,
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
        "phase_statistics": model.phase_statistics(),
    }
    write_json(settings.output_dir / "metrics" / "training_summary.json", summary)
    return summary
