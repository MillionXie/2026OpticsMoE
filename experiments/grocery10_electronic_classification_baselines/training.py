from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from .modeling import (
    ElectronicTenClassClassifier,
    parameter_report,
    preprocess_pil_images,
)
from .settings import Settings


def _loader(
    samples: Sequence[GrocerySample],
    settings: Settings,
    *,
    train: bool,
    device: torch.device,
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
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(samples),
            replacement=True,
            generator=torch.Generator().manual_seed(settings.random_seed),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=settings.batch_size if train else settings.inference_batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_grocery,
    )


def _labels(samples: Sequence[GrocerySample], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [int(sample.sku_index) for sample in samples],
        dtype=torch.long,
        device=device,
    )


def _confusion(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    return torch.bincount(
        labels.long() * 10 + predictions.long(), minlength=100
    ).reshape(10, 10)


def classification_metrics(
    labels: torch.Tensor, logits: torch.Tensor, average_loss: float
) -> dict[str, Any]:
    predictions = logits.argmax(1)
    top3 = logits.topk(3, 1).indices.eq(labels[:, None]).any(1)
    matrix = _confusion(labels, predictions)
    per_class = []
    recalls = []
    f1_values = []
    for index in range(10):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_count = int(matrix[:, index].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted_count if predicted_count else 0.0
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
        "loss": float(average_loss),
        "top1_accuracy": float(predictions.eq(labels).float().mean()),
        "top3_accuracy": float(top3.float().mean()),
        "balanced_accuracy": float(sum(recalls) / len(recalls)),
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "sample_count": int(len(labels)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate_split(
    model: ElectronicTenClassClassifier,
    samples: Sequence[GrocerySample],
    settings: Settings,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = _loader(samples, settings, train=False, device=device)
    model.eval()
    use_amp = settings.amp_enabled and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if settings.amp_dtype == "bfloat16" else torch.float16
    )
    labels_all = []
    logits_all = []
    loss_sum = 0.0
    count = 0
    records = []
    for batch in loader:
        images = preprocess_pil_images(batch["images"], device)
        labels = _labels(batch["samples"], device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), labels)
        probabilities = logits.float().softmax(1)
        predictions = probabilities.argmax(1)
        loss_sum += float(loss) * len(labels)
        count += len(labels)
        labels_all.append(labels.cpu())
        logits_all.append(logits.float().cpu())
        for offset, sample in enumerate(batch["samples"]):
            prediction = int(predictions[offset])
            row = {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "true_label": int(labels[offset]),
                "true_name": sample.sku_name,
                "predicted_label": prediction,
                "predicted_name": settings.selected_skus[prediction],
                "correct": prediction == int(labels[offset]),
            }
            for class_index, class_name in enumerate(settings.selected_skus):
                row[f"probability_{class_name}"] = float(
                    probabilities[offset, class_index]
                )
            records.append(row)
    if count == 0:
        raise RuntimeError("Electronic classification evaluation split is empty")
    return (
        classification_metrics(
            torch.cat(labels_all), torch.cat(logits_all), loss_sum / count
        ),
        records,
    )


def _optimizer(
    model: ElectronicTenClassClassifier, settings: Settings
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    backbone = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    if backbone:
        groups.append(
            {
                "params": backbone,
                "lr": settings.backbone_learning_rate,
                "group_name": "backbone",
            }
        )
    groups.append(
        {
            "params": list(model.feature_norm.parameters())
            + list(model.classifier.parameters()),
            "lr": settings.head_learning_rate,
            "group_name": "classification_head",
        }
    )
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


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
    model: ElectronicTenClassClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    train_loss: float,
    settings: Settings,
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
            **parameter_report(model),
            "selected_skus": list(settings.selected_skus),
            "manifest_sha256": manifest_digest,
            "selection_criterion": "minimum_training_cross_entropy",
            "test_metrics_used_for_selection": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, model: ElectronicTenClassClassifier
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Classification checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("model_name") not in {None, model.model_name}:
        raise RuntimeError("Classification checkpoint backbone mismatch")
    if metadata.get("num_classes") not in {None, model.num_classes}:
        raise RuntimeError("Classification checkpoint class-count mismatch")
    model.load_state_dict(payload["model"])
    return payload


def train(
    model: ElectronicTenClassClassifier,
    bundle: GroceryRetrievalBundle,
    settings: Settings,
    device: torch.device,
) -> dict[str, Any]:
    loader = _loader(bundle.train_samples, settings, train=True, device=device)
    optimizer = _optimizer(model, settings)
    scheduler = _scheduler(optimizer, settings)
    use_amp = settings.amp_enabled and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if settings.amp_dtype == "bfloat16" else torch.float16
    )
    scaler_enabled = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    rows: list[dict[str, Any]] = []
    best_train_loss = math.inf
    best_epoch = 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        started = time.perf_counter()
        loss_sum = 0.0
        correct = 0
        count = 0
        for batch_index, batch in enumerate(loader, 1):
            images = preprocess_pil_images(batch["images"], device)
            labels = _labels(batch["samples"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                logits = model(images)
                loss = F.cross_entropy(logits.float(), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite classification loss at epoch={epoch}, "
                    f"batch={batch_index}"
                )
            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int(logits.detach().argmax(1).eq(labels).sum())
            count += len(labels)
            if (
                batch_index % settings.log_interval_batches == 0
                or batch_index == len(loader)
            ):
                print(
                    f"{settings.model_name}-classification "
                    f"epoch {epoch:03d}/{settings.epochs:03d} "
                    f"batch {batch_index:03d}/{len(loader):03d} "
                    f"loss={loss_sum/count:.5f} train_top1={correct/count:.4f}",
                    flush=True,
                )
        average_loss = loss_sum / count
        test_metrics = {}
        if settings.evaluate_test_each_epoch:
            test_metrics, _ = evaluate_split(
                model, bundle.test_samples, settings, device
            )
        row = {
            "epoch": epoch,
            "backbone_learning_rate": next(
                (
                    group["lr"]
                    for group in optimizer.param_groups
                    if group.get("group_name") == "backbone"
                ),
                0.0,
            ),
            "head_learning_rate": next(
                group["lr"]
                for group in optimizer.param_groups
                if group.get("group_name") == "classification_head"
            ),
            "train_loss": average_loss,
            "train_top1_accuracy": correct / count,
            "test_loss": test_metrics.get("loss"),
            "test_top1_accuracy": test_metrics.get("top1_accuracy"),
            "test_top3_accuracy": test_metrics.get("top3_accuracy"),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "samples": count,
            "epoch_time_sec": time.perf_counter() - started,
            "checkpoint_selected_by": "minimum_training_cross_entropy",
        }
        rows.append(row)
        write_csv(
            settings.output_dir / "training_history.csv", rows, list(rows[0])
        )
        write_json(settings.output_dir / "metrics" / "training_latest.json", row)
        save_checkpoint(
            settings.output_dir / "last_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            average_loss,
            settings,
            bundle.manifest_digest,
        )
        if average_loss < best_train_loss:
            best_train_loss = average_loss
            best_epoch = epoch
            save_checkpoint(
                settings.output_dir / "best_train_loss_checkpoint.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                average_loss,
                settings,
                bundle.manifest_digest,
            )
        if scheduler is not None:
            scheduler.step()
        print(
            f"{settings.model_name}-classification epoch {epoch:03d} complete "
            f"train_top1={row['train_top1_accuracy']:.4f} "
            f"test_top1={row['test_top1_accuracy'] if row['test_top1_accuracy'] is not None else float('nan'):.4f} "
            f"best_train_loss={best_train_loss:.5f}",
            flush=True,
        )
    plot_training_curves(rows, settings.output_dir / "figures" / "training_curves.png")
    summary = {
        "epochs": settings.epochs,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "selection_criterion": "minimum training cross-entropy; test not used",
    }
    write_json(settings.output_dir / "metrics" / "training_summary.json", summary)
    return summary


def evaluate_and_save(
    model: ElectronicTenClassClassifier,
    bundle: GroceryRetrievalBundle,
    settings: Settings,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path or (
        settings.output_dir / "best_train_loss_checkpoint.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path, model)
    model.to(device)
    metrics, records = evaluate_split(
        model, bundle.test_samples, settings, device
    )
    metrics.update(
        {
            "class_names": list(bundle.class_names),
            "manifest_sha256": bundle.manifest_digest,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_selection": "minimum training cross-entropy; test not used",
            "task_type": "closed_set_direct_classification",
            "retrieval_similarity_used": False,
            "model": parameter_report(model),
        }
    )
    for item, class_name in zip(metrics["per_class"], bundle.class_names):
        item["class_name"] = class_name
    metrics_dir = settings.output_dir / "metrics"
    write_json(metrics_dir / "test_metrics.json", metrics)
    write_csv(
        metrics_dir / "test_predictions.csv",
        records,
        list(records[0]) if records else [],
    )
    write_csv(
        metrics_dir / "per_class_metrics.csv",
        metrics["per_class"],
        list(metrics["per_class"][0]),
    )
    matrix_rows = [
        {"true_name": bundle.class_names[row_index]}
        | {
            f"pred_{bundle.class_names[column]}": int(value)
            for column, value in enumerate(row)
        }
        for row_index, row in enumerate(metrics["confusion_matrix"])
    ]
    write_csv(
        metrics_dir / "confusion_matrix.csv",
        matrix_rows,
        list(matrix_rows[0]),
    )
    plot_confusion(
        metrics["confusion_matrix"],
        bundle.class_names,
        settings.output_dir / "figures" / "confusion_matrix.png",
        settings.model_name,
    )
    return metrics


def plot_training_curves(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train CE")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Training loss")
    axes[1].plot(
        epochs,
        [100.0 * row["train_top1_accuracy"] for row in rows],
        label="train",
    )
    if rows[0]["test_top1_accuracy"] is not None:
        axes[1].plot(
            epochs,
            [100.0 * row["test_top1_accuracy"] for row in rows],
            label="test",
        )
    axes[1].set(
        xlabel="Epoch", ylabel="Top-1 accuracy [%]", title="Direct classification"
    )
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_confusion(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: Path,
    model_name: str,
) -> None:
    values = torch.tensor(matrix).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Sample count")
    short = [name[:16] for name in class_names]
    axis.set_xticks(range(10), short, rotation=45, ha="right")
    axis.set_yticks(range(10), short)
    axis.set(
        xlabel="Predicted SKU",
        ylabel="True SKU",
        title=f"{model_name} direct classification",
    )
    for y in range(10):
        for x in range(10):
            axis.text(x, y, str(int(values[y, x])), ha="center", va="center", fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)
