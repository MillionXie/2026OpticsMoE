from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

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
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.retrieval_metrics import (
    RetrievalEvaluation,
    evaluate_embeddings,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    PKBatchSampler,
    gallery_retrieval_logits,
    retrieval_ranking_sums,
    select_gallery_items_for_queries,
    supervised_contrastive_loss,
)

from .modeling import (
    ElectronicRetrievalEncoder,
    parameter_report,
    preprocess_pil_images,
)
from .settings import Settings


def _optimizer(
    model: ElectronicRetrievalEncoder, settings: Settings
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head = list(model.feature_norm.parameters()) + list(model.projection.parameters())
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
            "params": head,
            "lr": settings.projection_learning_rate,
            "group_name": "retrieval_head",
        }
    )
    return torch.optim.AdamW(
        groups,
        weight_decay=settings.weight_decay,
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, settings: Settings
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if settings.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=settings.epochs,
        eta_min=settings.min_learning_rate,
    )


def save_checkpoint(
    path: Path,
    model: ElectronicRetrievalEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    train_loss: float,
    settings: Settings,
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
            "gallery_aggregation": settings.gallery_aggregation,
            "selection_criterion": "minimum_training_total_loss",
            "test_metrics_used_for_selection": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: ElectronicRetrievalEncoder,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Electronic baseline checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("model_name") not in {None, model.model_name}:
        raise RuntimeError(
            f"Checkpoint model={metadata.get('model_name')} does not match {model.model_name}"
        )
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def train(
    model: ElectronicRetrievalEncoder,
    bundle: GroceryRetrievalBundle,
    settings: Settings,
    device: torch.device,
) -> dict[str, Any]:
    train_dataset = GroceryRetrievalDataset(
        bundle.train_samples,
        settings.image_size,
        augment=settings.augmentation_enabled,
        crop_scale_min=settings.crop_scale_min,
        brightness_jitter=settings.brightness_jitter,
        contrast_jitter=settings.contrast_jitter,
        rotation_degrees=settings.rotation_degrees,
    )
    sampler = PKBatchSampler(
        bundle.train_samples,
        settings.pk_skus_per_batch,
        settings.pk_images_per_sku,
        settings.random_seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_grocery,
    )
    gallery_dataset = GroceryRetrievalDataset(
        bundle.gallery_samples, settings.image_size, augment=False
    )
    gallery_items = [gallery_dataset[index] for index in range(len(gallery_dataset))]
    if {
        int(item["sample"].sku_index) for item in gallery_items
    } != set(range(len(bundle.class_names))):
        raise RuntimeError("Electronic baseline gallery does not cover every SKU")

    optimizer = _optimizer(model, settings)
    scheduler = _scheduler(optimizer, settings)
    use_amp = settings.amp_enabled and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rows: list[dict[str, Any]] = []
    fieldnames = [
        "epoch",
        "backbone_learning_rate",
        "projection_learning_rate",
        "total_loss",
        "retrieval_loss",
        "gallery_loss",
        "train_top1",
        "train_top3",
        "train_mrr",
        "samples",
        "model_forward_samples",
        "epoch_time_sec",
        "test_top1",
        "test_top3",
        "test_mrr",
        "checkpoint_selected_by",
    ]
    best_train_loss = math.inf
    for epoch in range(1, settings.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        totals = {
            "total": 0.0,
            "retrieval": 0.0,
            "gallery": 0.0,
            "top1": 0.0,
            "top3": 0.0,
            "rr": 0.0,
            "samples": 0,
            "forward_samples": 0,
        }
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, 1):
            query_count = len(batch["samples"])
            gallery_batch = collate_grocery(
                select_gallery_items_for_queries(gallery_items, batch["samples"])
            )
            combined_images = batch["images"] + gallery_batch["images"]
            combined_samples = batch["samples"] + gallery_batch["samples"]
            query_labels = torch.tensor(
                [sample.sku_index for sample in batch["samples"]],
                dtype=torch.long,
                device=device,
            )
            gallery_labels = torch.tensor(
                [sample.sku_index for sample in gallery_batch["samples"]],
                dtype=torch.long,
                device=device,
            )
            labels = torch.cat((query_labels, gallery_labels))
            images = preprocess_pil_images(combined_images, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                embeddings = model(images)
                retrieval_loss = supervised_contrastive_loss(
                    embeddings, labels, settings.temperature
                )
                gallery_logits, gallery_targets = gallery_retrieval_logits(
                    embeddings[:query_count],
                    query_labels,
                    embeddings[query_count:],
                    gallery_labels,
                    settings.gallery_temperature,
                    stop_gradient_on_gallery=(
                        settings.gallery_prototype_stop_gradient
                    ),
                )
                gallery_loss = F.cross_entropy(gallery_logits, gallery_targets)
                total_loss = (
                    settings.lambda_ret * retrieval_loss
                    + settings.lambda_gallery * gallery_loss
                )
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch} batch={batch_index}"
                )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = query_count
            totals["total"] += float(total_loss.detach()) * count
            totals["retrieval"] += float(retrieval_loss.detach()) * count
            totals["gallery"] += float(gallery_loss.detach()) * count
            totals["samples"] += count
            totals["forward_samples"] += len(combined_samples)
            ranking = retrieval_ranking_sums(gallery_logits, gallery_targets)
            totals["top1"] += ranking["top1_correct"]
            totals["top3"] += ranking["top3_correct"]
            totals["rr"] += ranking["reciprocal_rank_sum"]
            if (
                batch_index % settings.log_interval_batches == 0
                or batch_index == len(loader)
            ):
                print(
                    f"{settings.model_name} epoch {epoch:03d}/{settings.epochs:03d} "
                    f"batch {batch_index:03d}/{len(loader):03d} "
                    f"loss={totals['total']/totals['samples']:.5f} "
                    f"train_top1={totals['top1']/totals['samples']:.4f}"
                )
        sample_count = int(totals["samples"])
        test_metrics: dict[str, Any] = {}
        if settings.evaluate_test_each_epoch:
            test_metrics = evaluate_split(
                model,
                bundle.test_samples,
                bundle.gallery_samples,
                bundle.class_names,
                settings,
                device,
            ).metrics
        backbone_lr = next(
            (
                group["lr"]
                for group in optimizer.param_groups
                if group.get("group_name") == "backbone"
            ),
            0.0,
        )
        projection_lr = next(
            group["lr"]
            for group in optimizer.param_groups
            if group.get("group_name") == "retrieval_head"
        )
        average_loss = totals["total"] / sample_count
        row = {
            "epoch": epoch,
            "backbone_learning_rate": backbone_lr,
            "projection_learning_rate": projection_lr,
            "total_loss": average_loss,
            "retrieval_loss": totals["retrieval"] / sample_count,
            "gallery_loss": totals["gallery"] / sample_count,
            "train_top1": totals["top1"] / sample_count,
            "train_top3": totals["top3"] / sample_count,
            "train_mrr": totals["rr"] / sample_count,
            "samples": sample_count,
            "model_forward_samples": int(totals["forward_samples"]),
            "epoch_time_sec": time.perf_counter() - started,
            "test_top1": test_metrics.get("top1_retrieval_accuracy"),
            "test_top3": test_metrics.get("top3_retrieval_accuracy"),
            "test_mrr": test_metrics.get("mrr"),
            "checkpoint_selected_by": "training_total_loss",
        }
        rows.append(row)
        write_csv(settings.output_dir / "train_log.csv", rows, fieldnames)
        write_json(settings.output_dir / "metrics" / "training_latest.json", row)
        save_checkpoint(
            settings.output_dir / "last_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            average_loss,
            settings,
        )
        if average_loss < best_train_loss:
            best_train_loss = average_loss
            save_checkpoint(
                settings.output_dir / "best_train_loss_checkpoint.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                average_loss,
                settings,
            )
            write_json(
                settings.output_dir / "metrics" / "best_train_loss.json",
                {
                    "epoch": epoch,
                    "train_total_loss": average_loss,
                    "selection_criterion": "minimum_training_total_loss",
                    "test_was_not_used_for_selection": True,
                },
            )
        if scheduler is not None:
            scheduler.step()
        print(
            f"{settings.model_name} epoch {epoch:03d} complete "
            f"train_top1={row['train_top1']:.4f} "
            f"test_top1={test_metrics.get('top1_retrieval_accuracy', float('nan')):.4f} "
            f"best_train_loss={best_train_loss:.5f}"
        )
    plot_training_curves(settings.output_dir / "train_log.csv", settings.output_dir)
    return {
        "epochs": settings.epochs,
        "best_train_loss": best_train_loss,
        "selection_criterion": "minimum training total loss; test not used",
    }


@torch.no_grad()
def encode_samples(
    model: ElectronicRetrievalEncoder,
    samples: Sequence[GrocerySample],
    settings: Settings,
    device: torch.device,
) -> torch.Tensor:
    dataset = GroceryRetrievalDataset(samples, settings.image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=collate_grocery,
    )
    model.eval()
    use_amp = settings.amp_enabled and device.type == "cuda"
    chunks: list[torch.Tensor] = []
    for batch in loader:
        images = preprocess_pil_images(batch["images"], device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            values = model(images)
        chunks.append(values.detach().cpu())
    output = torch.cat(chunks)
    if output.shape != (len(samples), settings.embedding_dim):
        raise RuntimeError(f"Encoded baseline shape {tuple(output.shape)} is invalid")
    return output


def evaluate_split(
    model: ElectronicRetrievalEncoder,
    query_samples: Sequence[GrocerySample],
    gallery_samples: Sequence[GrocerySample],
    class_names: Sequence[str],
    settings: Settings,
    device: torch.device,
) -> RetrievalEvaluation:
    gallery = encode_samples(model, gallery_samples, settings, device)
    query = encode_samples(model, query_samples, settings, device)
    return evaluate_embeddings(
        query,
        query_samples,
        gallery,
        gallery_samples,
        class_names,
        settings.gallery_aggregation,
        system_name=f"electronic_{settings.model_name}_query_vs_gallery",
    )


def evaluate(
    model: ElectronicRetrievalEncoder,
    bundle: GroceryRetrievalBundle,
    settings: Settings,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> RetrievalEvaluation:
    checkpoint_path = (
        checkpoint_path or settings.output_dir / "best_train_loss_checkpoint.pt"
    )
    checkpoint = load_checkpoint(checkpoint_path, model)
    result = evaluate_split(
        model,
        bundle.test_samples,
        bundle.gallery_samples,
        bundle.class_names,
        settings,
        device,
    )
    metrics = {
        **result.metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_selection": "minimum training total loss; test not used",
        "model": parameter_report(model),
        "manifest_sha256": bundle.manifest_digest,
    }
    write_json(settings.output_dir / "metrics" / "test_metrics.json", metrics)
    write_csv(
        settings.output_dir / "metrics" / "retrieval_results.csv",
        result.rows,
        list(result.rows[0]),
    )
    per_sku_rows = [
        {"sku": name, **values}
        for name, values in result.metrics["per_sku"].items()
    ]
    write_csv(
        settings.output_dir / "metrics" / "per_sku_metrics.csv",
        per_sku_rows,
        ["sku", "query_count", "top1_accuracy", "top3_accuracy"],
    )
    plot_confusion(
        result.confusion,
        bundle.class_names,
        settings.output_dir / "figures" / "confusion_matrix.png",
        settings.model_name,
    )
    _write_optical_comparison(settings, metrics)
    return result


def _write_optical_comparison(
    settings: Settings, baseline_metrics: dict[str, Any]
) -> None:
    path = settings.optical_metrics_path
    if path is None or not path.is_file():
        return
    import json

    optical = json.loads(path.read_text(encoding="utf-8"))
    write_json(
        settings.output_dir / "metrics" / "comparison_with_optical.json",
        {
            "electronic_model": settings.model_name,
            "electronic": {
                "top1": baseline_metrics["top1_retrieval_accuracy"],
                "top3": baseline_metrics["top3_retrieval_accuracy"],
                "mrr": baseline_metrics["mrr"],
                "parameters": baseline_metrics["model"]["parameters"],
                "trainable_parameters": baseline_metrics["model"][
                    "trainable_parameters"
                ],
            },
            "optical": {
                "top1": optical["top1_retrieval_accuracy"],
                "top3": optical["top3_retrieval_accuracy"],
                "mrr": optical["mrr"],
                "checkpoint": optical.get("checkpoint"),
            },
            "electronic_minus_optical": {
                "top1": baseline_metrics["top1_retrieval_accuracy"]
                - optical["top1_retrieval_accuracy"],
                "top3": baseline_metrics["top3_retrieval_accuracy"]
                - optical["top3_retrieval_accuracy"],
                "mrr": baseline_metrics["mrr"] - optical["mrr"],
            },
        },
    )


def plot_training_curves(csv_path: Path, output_dir: Path) -> None:
    if not csv_path.is_file():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(epochs, [float(row["total_loss"]) for row in rows], label="total")
    axes[0].plot(
        epochs, [float(row["retrieval_loss"]) for row in rows], label="contrastive"
    )
    axes[0].plot(
        epochs, [float(row["gallery_loss"]) for row in rows], label="gallery"
    )
    axes[0].set(title="Training losses", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(
        epochs, [float(row["train_top1"]) for row in rows], label="train Top-1"
    )
    axes[1].plot(
        epochs, [float(row["test_top1"]) for row in rows], label="test Top-1"
    )
    axes[1].set(title="Retrieval accuracy", xlabel="Epoch", ylabel="Top-1")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    path = output_dir / "figures" / "training_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_confusion(
    matrix: torch.Tensor,
    class_names: Sequence[str],
    path: Path,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 10), constrained_layout=True)
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis, label="Query count")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set(xlabel="Retrieved SKU", ylabel="True SKU", title=f"{model_name} retrieval")
    for y in range(len(class_names)):
        for x in range(len(class_names)):
            axis.text(x, y, str(int(matrix[y, x])), ha="center", va="center", fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)
