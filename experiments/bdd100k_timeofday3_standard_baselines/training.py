from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from ..qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual.sampling import (
    EpochClassMixedSampler,
)
from .data import DataBundle, labels_of, make_loader
from .metrics import metrics_from_logits, write_confusion_csv, write_csv, write_json
from .models import StandardD2NNTimeOfDayClassifier
from .visualization import save_confusion_matrix, save_optical_diagnostics, save_training_curves


def train_model(model: nn.Module, data: DataBundle, settings: Any, device: torch.device) -> list[dict[str, Any]]:
    train_sampler = EpochClassMixedSampler(
        range(len(data.train)),
        labels_of(data.train),
        len(data.class_names),
        settings.batch_size,
        settings.seed,
        settings.train_samples_per_class_per_epoch,
    )
    train_loader = make_loader(data.train, settings.batch_size, settings.num_workers, False, settings.seed, train_sampler)
    validation_loader = make_loader(
        data.validation, settings.batch_size, settings.num_workers, False, settings.seed + 1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    best_top1 = -1.0
    best_macro = -1.0
    diagnostic_images = _first_images(validation_loader, device)

    for epoch in range(1, settings.epochs + 1):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        train_sampler.set_epoch(epoch)
        print(
            f"[sampling] epoch={epoch} samples={len(train_sampler)} "
            f"per_class={train_sampler.epoch_class_counts()} shuffled=True",
            flush=True,
        )
        model.train()
        loss_totals = {"total": 0.0, "classification": 0.0, "detector_concentration": 0.0}
        detector_fraction_total = 0.0
        logits_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        seen = 0
        for batch_index, (images, labels, _indices, _paths) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, loss, components, aux = _forward_objective(model, images, labels, criterion, settings)
            loss.backward()
            optimizer.step()
            count = len(labels)
            seen += count
            for name, value in components.items():
                loss_totals[name] += float(value.detach()) * count
            if aux is not None:
                detector_fraction_total += float(aux["detector_fraction"].detach().sum())
            logits_chunks.append(logits.detach().cpu())
            label_chunks.append(labels.detach().cpu())
            if batch_index % settings.log_interval_batches == 0 or batch_index == len(train_loader):
                running = metrics_from_logits(torch.cat(logits_chunks), torch.cat(label_chunks), data.class_names)
                detector_text = (
                    f" detector_fraction={detector_fraction_total / seen:.4f}"
                    if detector_fraction_total
                    else ""
                )
                print(
                    f"epoch {epoch}/{settings.epochs} batch {batch_index}/{len(train_loader)}\n"
                    f"loss_total={loss_totals['total'] / seen:.6f} "
                    f"loss_classification={loss_totals['classification'] / seen:.6f} "
                    f"running_top1={running['top1_accuracy']:.4f}{detector_text}\n"
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        train_time = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation = evaluate(model, validation_loader, device, data.class_names, settings)
        validation_time = time.perf_counter() - validation_started
        train_metrics = metrics_from_logits(torch.cat(logits_chunks), torch.cat(label_chunks), data.class_names)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_totals["total"] / seen,
            "train_classification_loss": loss_totals["classification"] / seen,
            "train_detector_concentration_loss": loss_totals["detector_concentration"] / seen,
            "train_mean_detector_energy_fraction": detector_fraction_total / seen if detector_fraction_total else None,
            "train_top1_accuracy": train_metrics["top1_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation["metrics"]["loss"],
            "validation_classification_loss": validation["metrics"]["classification_loss"],
            "validation_detector_concentration_loss": validation["metrics"]["detector_concentration_loss"],
            "validation_mean_detector_energy_fraction": validation["metrics"].get("mean_detector_energy_fraction"),
            "validation_mean_target_region_energy_fraction": validation["metrics"].get(
                "mean_target_region_energy_fraction"
            ),
            "validation_top1_accuracy": validation["metrics"]["top1_accuracy"],
            "validation_top5_accuracy": validation["metrics"]["top5_accuracy"],
            "validation_macro_f1": validation["metrics"]["macro_f1"],
            "validation_balanced_accuracy": validation["metrics"]["balanced_accuracy"],
            "epoch_time_sec": time.perf_counter() - epoch_started,
            "train_time_sec": train_time,
            "validation_time_sec": validation_time,
        }
        history.append(row)
        write_csv(settings.output_dir / "metrics" / "training_history.csv", history, list(row))
        write_json(settings.output_dir / "metrics" / "training_latest.json", row)
        if epoch % settings.save_predictions_interval_epochs == 0:
            _write_predictions(
                settings.output_dir / "metrics" / f"validation_predictions_epoch_{epoch:04d}.csv",
                validation,
                data.class_names,
            )
        improved = row["validation_top1_accuracy"] > best_top1 or row["validation_macro_f1"] > best_macro
        if improved:
            best_top1 = max(best_top1, row["validation_top1_accuracy"])
            best_macro = max(best_macro, row["validation_macro_f1"])
            _save_checkpoint(settings.output_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, epoch, row)
            write_json(settings.output_dir / "metrics" / "best_validation.json", row)
        _save_checkpoint(settings.output_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, row)
        if isinstance(model, StandardD2NNTimeOfDayClassifier) and (
            epoch == 1 or epoch % settings.save_interval_epochs == 0 or epoch == settings.epochs
        ):
            save_optical_diagnostics(model, diagnostic_images, settings.output_dir / "figures", epoch)
        save_training_curves(history, settings.output_dir / "figures" / "training_curves.png")
        scheduler.step()
        print(
            f"[epoch {epoch:03d}] val_top1={row['validation_top1_accuracy']:.4f} "
            f"val_macro_f1={row['validation_macro_f1']:.4f} time={row['epoch_time_sec']:.1f}s",
            flush=True,
        )
    return history


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: Any, device: torch.device, class_names: Sequence[str], settings: Any
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_totals = {"total": 0.0, "classification": 0.0, "detector_concentration": 0.0}
    logits_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    indices_all: list[int] = []
    paths_all: list[str] = []
    detector_fractions: list[torch.Tensor] = []
    target_fractions: list[torch.Tensor] = []
    for images, labels, indices, paths in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits, _loss, components, aux = _forward_objective(model, images, labels_device, criterion, settings)
        count = len(labels)
        for name, value in components.items():
            loss_totals[name] += float(value.detach()) * count
        logits_chunks.append(logits.cpu())
        label_chunks.append(labels)
        indices_all.extend(indices.tolist())
        paths_all.extend(paths)
        if aux is not None:
            detector_fractions.append(aux["detector_fraction"].detach().cpu())
            target_fractions.append(aux["region_fractions"].detach().cpu().gather(1, labels[:, None]).squeeze(1))
    logits_all = torch.cat(logits_chunks)
    labels_all = torch.cat(label_chunks)
    metrics = metrics_from_logits(logits_all, labels_all, class_names)
    samples = max(1, len(labels_all))
    metrics.update(
        {
            "loss": loss_totals["total"] / samples,
            "classification_loss": loss_totals["classification"] / samples,
            "detector_concentration_loss": loss_totals["detector_concentration"] / samples,
        }
    )
    if detector_fractions:
        detector_values = torch.cat(detector_fractions)
        target_values = torch.cat(target_fractions)
        metrics.update(
            {
                "mean_detector_energy_fraction": float(detector_values.mean()),
                "mean_target_region_energy_fraction": float(target_values.mean()),
                "per_class_mean_target_region_energy_fraction": {
                    name: float(target_values[labels_all.eq(index)].mean()) if labels_all.eq(index).any() else 0.0
                    for index, name in enumerate(class_names)
                },
            }
        )
    return {"metrics": metrics, "logits": logits_all, "labels": labels_all, "indices": indices_all, "paths": paths_all}


def test_model(model: nn.Module, data: DataBundle, settings: Any, device: torch.device) -> dict[str, Any]:
    checkpoint = settings.output_dir / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint missing: {checkpoint}. Run train first.")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    result = evaluate(
        model,
        make_loader(data.test, settings.batch_size, settings.num_workers, False, settings.seed + 2),
        device,
        data.class_names,
        settings,
    )
    write_json(settings.output_dir / "metrics" / "test_metrics.json", result["metrics"])
    write_json(settings.output_dir / "metrics" / "per_class_metrics.json", result["metrics"]["per_class"])
    write_confusion_csv(
        settings.output_dir / "metrics" / "confusion_matrix.csv",
        result["metrics"]["confusion_matrix"],
        data.class_names,
    )
    _write_predictions(settings.output_dir / "metrics" / "test_predictions.csv", result, data.class_names)
    save_confusion_matrix(
        result["metrics"]["confusion_matrix"],
        data.class_names,
        settings.output_dir / "figures" / "confusion_matrix.png",
    )
    return result["metrics"]


def _forward_objective(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    settings: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    if isinstance(model, StandardD2NNTimeOfDayClassifier):
        logits, aux = model(images, return_aux=True)
        classification = criterion(logits, labels)
        concentration = -torch.log(aux["detector_fraction"].clamp_min(1e-8)).mean()
        total = classification + settings.detector_concentration_loss_weight * concentration
        return (
            logits,
            total,
            {
                "total": total,
                "classification": classification,
                "detector_concentration": concentration,
            },
            aux,
        )
    logits = model(images)
    classification = criterion(logits, labels)
    zero = classification.new_zeros(())
    return (
        logits,
        classification,
        {"total": classification, "classification": classification, "detector_concentration": zero},
        None,
    )


def _write_predictions(path: Path, result: dict[str, Any], names: Sequence[str]) -> None:
    predictions = result["logits"].argmax(1).tolist()
    rows = []
    for index, image_path, truth, pred, values in zip(
        result["indices"], result["paths"], result["labels"].tolist(), predictions, result["logits"].tolist()
    ):
        row = {
            "sample_index": index,
            "image_path": image_path,
            "true_label": truth,
            "true_name": names[truth],
            "pred_label": pred,
            "pred_name": names[pred],
            "correct": truth == pred,
        }
        row.update({f"logit_{name}": value for name, value in zip(names, values)})
        rows.append(row)
    write_csv(path, rows, list(rows[0]) if rows else None)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": row,
        },
        path,
    )


def _first_images(loader: Any, device: torch.device) -> torch.Tensor:
    try:
        images = next(iter(loader))[0][:8].to(device)
    except StopIteration:
        return torch.empty(0, 1, 1, 1, device=device)
    return images
