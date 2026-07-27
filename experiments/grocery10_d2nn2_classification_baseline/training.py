from __future__ import annotations

import csv
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    write_csv,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    GroceryRetrievalBundle,
    GroceryRetrievalDataset,
    GrocerySample,
    collate_grocery,
)

from .modeling import TwoPlaneD2NNClassifier, pil_images_to_amplitude
from .settings import Settings
from .visualization import (
    save_confusion_matrix,
    save_debug_examples,
    save_history_csv,
    save_phase_masks,
    save_training_curves,
)


def detector_region_cross_entropy(
    region_energies: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if region_energies.ndim != 2 or region_energies.shape[1] != 10:
        raise ValueError("D2NN detector energies must have shape [B,10]")
    if torch.any(region_energies < 0):
        raise ValueError("D2NN detector energies cannot be negative")
    probabilities = region_energies.float() / region_energies.float().sum(
        1, keepdim=True
    ).clamp_min(1e-12)
    return F.nll_loss(
        torch.log(probabilities.clamp_min(1e-12)),
        labels.long(),
    )


def detector_plane_mse_loss(
    intensity: torch.Tensor,
    target_plane: torch.Tensor,
    *,
    scale: float,
    normalize: bool,
    eps: float,
) -> torch.Tensor:
    """Full-plane MSE with optional per-sample total-energy matching."""

    if intensity.shape != target_plane.shape or intensity.ndim != 3:
        raise ValueError("Detector intensity and target plane must share [B,H,W]")
    if float(scale) <= 0 or float(eps) <= 0:
        raise ValueError("Detector-plane MSE scale and eps must be positive")
    prediction = intensity.float()
    target = target_plane.float()
    if normalize:
        prediction_energy = prediction.sum((-2, -1), keepdim=True)
        target_energy = target.sum((-2, -1), keepdim=True)
        prediction = prediction * target_energy / prediction_energy.clamp_min(eps)
    return float(scale) * F.mse_loss(prediction, target)


def _active_detector_targets(
    model: TwoPlaneD2NNClassifier, labels: torch.Tensor
) -> torch.Tensor:
    masks = model.detector.masks[
        labels.long(),
        model.active_start : model.active_end,
        model.active_start : model.active_end,
    ]
    return masks.to(device=labels.device, dtype=torch.float32)


def _forward_loss(
    model: TwoPlaneD2NNClassifier,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    settings = model.settings
    if settings.loss_type == "detector_plane_mse":
        energies, intensity = model(images, return_detector_intensity=True)
        loss = detector_plane_mse_loss(
            intensity,
            _active_detector_targets(model, labels),
            scale=settings.detector_plane_mse_scale,
            normalize=settings.normalize_detector_plane_mse,
            eps=settings.detector_plane_mse_normalization_eps,
        )
        return energies, loss
    energies = model(images)
    return energies, detector_region_cross_entropy(energies, labels)


def _labels(samples: Sequence[GrocerySample], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [int(sample.sku_index) for sample in samples],
        dtype=torch.long,
        device=device,
    )


def _loader(
    samples: Sequence[GrocerySample],
    settings: Settings,
    *,
    train: bool,
) -> DataLoader:
    dataset = GroceryRetrievalDataset(
        samples,
        settings.image_size,
        augment=train and settings.augmentation_enabled,
        crop_scale_min=settings.crop_scale_min,
        brightness_jitter=settings.brightness_jitter,
        contrast_jitter=settings.contrast_jitter,
        rotation_degrees=settings.rotation_degrees,
    )
    sampler = None
    shuffle = train
    if train and settings.class_balanced_sampling:
        counts = Counter(int(sample.sku_index) for sample in samples)
        weights = torch.tensor(
            [1.0 / counts[int(sample.sku_index)] for sample in samples],
            dtype=torch.double,
        )
        generator = torch.Generator().manual_seed(settings.random_seed)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(samples),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=settings.batch_size if train else settings.inference_batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=True,
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_grocery,
    )


def _confusion(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    flat = labels.long() * 10 + predictions.long()
    return torch.bincount(flat, minlength=100).reshape(10, 10)


def _classification_metrics(
    labels: torch.Tensor, logits: torch.Tensor, loss: float
) -> dict[str, Any]:
    predictions = logits.argmax(1)
    top3 = logits.topk(3, 1).indices.eq(labels[:, None]).any(1)
    matrix = _confusion(labels, predictions)
    per_class: list[dict[str, Any]] = []
    recalls = []
    f1_values = []
    for index in range(10):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        f1_values.append(f1)
        per_class.append(
            {
                "class_index": index,
                "support": support,
                "correct": tp,
                "accuracy": recall,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "loss": float(loss),
        "top1_accuracy": float(predictions.eq(labels).float().mean()),
        "top3_accuracy": float(top3.float().mean()),
        "balanced_accuracy": float(sum(recalls) / len(recalls)),
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "sample_count": int(len(labels)),
    }


@torch.no_grad()
def evaluate(
    model: TwoPlaneD2NNClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    logits_all = []
    labels_all = []
    records: list[dict[str, Any]] = []
    total_loss = 0.0
    count = 0
    for batch in loader:
        images = pil_images_to_amplitude(
            batch["images"], model.settings.input_encoding
        ).to(device, non_blocking=True)
        labels = _labels(batch["samples"], device)
        logits, loss = _forward_loss(model, images, labels)
        probabilities = logits.float() / logits.float().sum(1, keepdim=True).clamp_min(
            1e-12
        )
        predictions = logits.argmax(1)
        total_loss += float(loss) * len(labels)
        count += len(labels)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        for offset, sample in enumerate(batch["samples"]):
            row = {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "true_label": int(labels[offset]),
                "true_name": sample.sku_name,
                "predicted_label": int(predictions[offset]),
                "predicted_name": model.settings.selected_skus[
                    int(predictions[offset])
                ],
                "correct": int(predictions[offset]) == int(labels[offset]),
            }
            for class_index, class_name in enumerate(model.settings.selected_skus):
                row[f"probability_{class_name}"] = float(
                    probabilities[offset, class_index]
                )
            records.append(row)
    if not count:
        raise RuntimeError("Evaluation loader is empty")
    labels = torch.cat(labels_all)
    logits = torch.cat(logits_all)
    return _classification_metrics(labels, logits, total_loss / count), records


def _optimizer(
    model: TwoPlaneD2NNClassifier, settings: Settings
) -> torch.optim.Optimizer:
    implementation = (
        torch.optim.AdamW if settings.optimizer == "adamw" else torch.optim.Adam
    )
    return implementation(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, settings: Settings
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if settings.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.epochs, eta_min=settings.min_learning_rate
    )


def save_checkpoint(
    path: Path,
    model: TwoPlaneD2NNClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    train_loss: float,
    manifest_digest: str,
) -> None:
    payload = {
        "checkpoint_version": 1,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metadata": {
            **model.parameter_report(),
            "manifest_sha256": manifest_digest,
            "selected_skus": list(model.settings.selected_skus),
            "loss_type": model.settings.loss_type,
            "selection_criterion": (
                f"minimum_training_{model.settings.loss_type}"
            ),
            "test_metrics_used_for_selection": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, model: TwoPlaneD2NNClassifier
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"D2NN checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return payload


def train(
    model: TwoPlaneD2NNClassifier,
    bundle: GroceryRetrievalBundle,
    settings: Settings,
    device: torch.device,
) -> dict[str, Any]:
    training_samples = list(bundle.train_samples)
    if settings.include_gallery_in_training:
        for _ in range(settings.gallery_repeat_factor):
            training_samples.extend(bundle.gallery_samples)
    train_loader = _loader(training_samples, settings, train=True)
    test_loader = _loader(bundle.test_samples, settings, train=False)
    optimizer = _optimizer(model, settings)
    scheduler = _scheduler(optimizer, settings)
    rows: list[dict[str, Any]] = []
    best_train_loss = math.inf
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, settings.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        correct = 0
        sample_count = 0
        for batch_index, batch in enumerate(train_loader, 1):
            images = pil_images_to_amplitude(
                batch["images"], settings.input_encoding
            ).to(device, non_blocking=True)
            labels = _labels(batch["samples"], device)
            optimizer.zero_grad(set_to_none=True)
            logits, loss = _forward_loss(model, images, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite D2NN loss at epoch={epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int(logits.detach().argmax(1).eq(labels).sum())
            sample_count += len(labels)
            if (
                batch_index % settings.log_interval_batches == 0
                or batch_index == len(train_loader)
            ):
                print(
                    f"d2nn2 epoch {epoch:03d}/{settings.epochs:03d} "
                    f"batch {batch_index:03d}/{len(train_loader):03d} "
                    f"loss={loss_sum / sample_count:.5f} "
                    f"train_top1={correct / sample_count:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        train_loss = loss_sum / sample_count
        train_top1 = correct / sample_count
        test_metrics = None
        if settings.evaluate_test_each_epoch:
            test_metrics, _ = evaluate(model, test_loader, device)
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_top1_accuracy": train_top1,
            "test_loss": test_metrics["loss"] if test_metrics else None,
            "test_top1_accuracy": (
                test_metrics["top1_accuracy"] if test_metrics else None
            ),
            "test_top3_accuracy": (
                test_metrics["top3_accuracy"] if test_metrics else None
            ),
            "test_macro_f1": test_metrics["macro_f1"] if test_metrics else None,
            "epoch_time_sec": time.perf_counter() - epoch_started,
            "samples_this_epoch": sample_count,
        }
        rows.append(row)
        save_history_csv(rows, settings.output_dir / "metrics" / "training_history.csv")
        save_history_csv(rows, settings.output_dir / "train_log.csv")
        write_json(settings.output_dir / "metrics" / "training_latest.json", row)
        save_checkpoint(
            settings.output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            bundle.manifest_digest,
        )
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            save_checkpoint(
                settings.output_dir / "checkpoints" / "best_train_loss.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                bundle.manifest_digest,
            )
            write_json(
                settings.output_dir / "metrics" / "best_train_loss.json",
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "selection_criterion": f"minimum_training_{settings.loss_type}",
                    "test_metrics_used_for_selection": False,
                },
            )
        if epoch % settings.save_visualization_interval_epochs == 0:
            save_phase_masks(
                model, settings.output_dir / "figures" / "phase_masks", epoch
            )
            save_training_curves(
                rows, settings.output_dir / "figures" / "training_curves.png"
            )
        print(
            f"d2nn2 epoch {epoch:03d} complete "
            f"train_top1={train_top1:.4f} "
            f"test_top1={test_metrics['top1_accuracy']:.4f} "
            f"best_train_loss={best_train_loss:.5f}",
            flush=True,
        )
    write_json(
        settings.output_dir / "metrics" / "training_summary.json",
        {
            "epochs": settings.epochs,
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "total_time_sec": time.perf_counter() - started,
            "selection_criterion": f"minimum_training_{settings.loss_type}",
            "natural_training_samples": len(bundle.train_samples),
            "gallery_training_samples_per_epoch": (
                len(bundle.gallery_samples) * settings.gallery_repeat_factor
                if settings.include_gallery_in_training
                else 0
            ),
        },
    )
    save_training_curves(rows, settings.output_dir / "figures" / "training_curves.png")
    return {"best_epoch": best_epoch, "best_train_loss": best_train_loss}


def evaluate_and_save(
    model: TwoPlaneD2NNClassifier,
    bundle: GroceryRetrievalBundle,
    settings: Settings,
    device: torch.device,
    *,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = (
        settings.output_dir / "checkpoints" / "best_train_loss.pt"
        if checkpoint_path is None
        else checkpoint_path
    )
    payload = load_checkpoint(checkpoint_path, model)
    model.to(device)
    loader = _loader(bundle.test_samples, settings, train=False)
    metrics, records = evaluate(model, loader, device)
    metrics.update(
        {
            "class_names": list(bundle.class_names),
            "manifest_sha256": bundle.manifest_digest,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(payload["epoch"]),
            "selection_criterion": f"minimum_training_{settings.loss_type}",
            "task_type": "closed_set_detector_region_classification",
            "retrieval_similarity_used": False,
            "teacher_or_distillation_used": False,
            "moe_used": False,
            "intermediate_oeo_nonlinearity_used": False,
        }
    )
    for entry, class_name in zip(metrics["per_class"], bundle.class_names):
        entry["class_name"] = class_name
    metrics_dir = settings.output_dir / "metrics"
    write_json(metrics_dir / "test_metrics.json", metrics)
    fieldnames = list(records[0]) if records else []
    write_csv(metrics_dir / "test_predictions.csv", records, fieldnames)
    per_class_rows = metrics["per_class"]
    write_csv(
        metrics_dir / "per_class_metrics.csv",
        per_class_rows,
        list(per_class_rows[0]),
    )
    matrix_rows = []
    for index, row in enumerate(metrics["confusion_matrix"]):
        matrix_rows.append(
            {"true_name": bundle.class_names[index]}
            | {
                f"pred_{bundle.class_names[column]}": int(value)
                for column, value in enumerate(row)
            }
        )
    write_csv(
        metrics_dir / "confusion_matrix.csv",
        matrix_rows,
        list(matrix_rows[0]),
    )
    save_confusion_matrix(
        metrics["confusion_matrix"],
        bundle.class_names,
        settings.output_dir / "figures" / "confusion_matrix.png",
    )
    save_phase_masks(
        model, settings.output_dir / "figures" / "phase_masks", "best"
    )
    debug_dataset = GroceryRetrievalDataset(
        bundle.test_samples, settings.image_size, augment=False
    )
    save_debug_examples(
        model,
        debug_dataset,
        bundle.class_names,
        device,
        settings.output_dir / "figures" / "debug_examples",
        settings.visualization_sample_count,
    )
    print(
        f"D2NN2 test: Top-1={metrics['top1_accuracy']:.4f} "
        f"Top-3={metrics['top3_accuracy']:.4f} "
        f"macro-F1={metrics['macro_f1']:.4f}",
        flush=True,
    )
    return metrics
