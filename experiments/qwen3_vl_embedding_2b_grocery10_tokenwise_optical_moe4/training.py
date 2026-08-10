from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.cache_teacher_embeddings import (
    TeacherEmbeddingStore,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    move_inputs,
    preprocess_images,
    validate_token_budgets,
)
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
    embedding_distillation_loss,
    supervised_contrastive_loss,
)

from .modeling import (
    LoadedBackbone,
    TokenwiseVisionReplacement,
    student_embeddings,
    trainable_parameter_report,
)


def _loader(
    samples: Sequence[GrocerySample],
    settings: Any,
    *,
    training: bool,
) -> tuple[DataLoader, PKBatchSampler | None]:
    dataset = GroceryRetrievalDataset(
        samples,
        settings.image_size,
        augment=training and settings.augmentation_enabled,
        crop_scale_min=settings.crop_scale_min,
        brightness_jitter=settings.brightness_jitter,
        contrast_jitter=settings.contrast_jitter,
        rotation_degrees=settings.rotation_degrees,
    )
    if training:
        sampler = PKBatchSampler(
            samples,
            settings.pk_skus_per_batch,
            settings.pk_images_per_sku,
            settings.random_seed,
            settings.optimizer_steps_per_epoch,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=settings.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=settings.num_workers > 0,
            collate_fn=collate_grocery,
        )
        return loader, sampler
    loader = DataLoader(
        dataset,
        batch_size=settings.inference_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=settings.num_workers > 0,
        collate_fn=collate_grocery,
    )
    return loader, None


def _inputs(loaded: LoadedBackbone, batch: dict[str, Any], settings: Any) -> dict[str, torch.Tensor]:
    inputs = preprocess_images(loaded.processor, batch["images"], settings.instruction)
    validate_token_budgets(inputs, settings)
    return move_inputs(inputs, loaded.device)


def _optimizer(replacement: TokenwiseVisionReplacement, settings: Any) -> torch.optim.Optimizer:
    cores = [replacement.vision_surrogate]
    if replacement.language_surrogate is not None:
        cores.append(replacement.language_surrogate.optical_core)
    router_ids = {id(p) for core in cores for p in core.router.parameters()}
    phase_ids = {
        id(parameter)
        for core in cores
        for name, parameter in core.named_parameters()
        if name.endswith("raw_phase")
    }
    router, phase, other = [], [], []
    for parameter in replacement.trainable_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in router_ids:
            router.append(parameter)
        elif id(parameter) in phase_ids:
            phase.append(parameter)
        else:
            other.append(parameter)
    groups = [
        {"params": router, "lr": settings.router_learning_rate, "name": "router"},
        {"params": phase, "lr": settings.phase_learning_rate, "name": "phase"},
    ]
    if other:
        groups.append({"params": other, "lr": settings.learning_rate, "name": "other"})
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def _batch_retrieval_accuracy(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    if len(embeddings) < 2:
        return 0.0
    similarity = F.normalize(embeddings.float(), dim=-1) @ F.normalize(embeddings.float(), dim=-1).T
    similarity.fill_diagonal_(-torch.inf)
    predicted = labels[similarity.argmax(dim=1)]
    return float(predicted.eq(labels).float().mean())


def train(
    loaded: LoadedBackbone,
    replacement: TokenwiseVisionReplacement,
    bundle: GroceryRetrievalBundle,
    teacher_store: TeacherEmbeddingStore,
    settings: Any,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    report = trainable_parameter_report(loaded, replacement)
    write_json(settings.output_dir / "metrics" / "model_parameters.json", report)
    print(
        f"token-wise panel active={settings.active_height}x{settings.active_width} "
        f"canvas={settings.canvas_size} max_tokens={settings.max_tokens} "
        f"trainable={report['trainable_parameters']:,}",
        flush=True,
    )
    for row in report["trainable_parameter_list"]:
        print(f"  {row['name']} shape={row['shape']} params={row['parameters']:,}", flush=True)
    loader, sampler = _loader(bundle.train_samples, settings, training=True)
    optimizer = _optimizer(replacement, settings)
    start_epoch = 1
    if resume_checkpoint is not None:
        payload = load_checkpoint(replacement, resume_checkpoint, loaded.device)
        if settings.weight_decay == payload.get("settings", {}).get("weight_decay"):
            optimizer_state = payload.get("optimizer")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
        start_epoch = int(payload.get("epoch", 0)) + 1
        print(f"resumed {resume_checkpoint} at epoch={start_epoch}", flush=True)
    history: list[dict[str, Any]] = []
    best_train_loss = float("inf")
    best_observed_test_top1 = float("-inf")
    best_observed_test_epoch = -1
    history_path = settings.output_dir / "metrics" / "training_history.csv"
    history_path.unlink(missing_ok=True)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=settings.amp_enabled and loaded.device.type == "cuda"
    )
    for epoch in range(start_epoch, settings.epochs + 1):
        assert sampler is not None
        sampler.set_epoch(epoch)
        replacement.use_student()
        replacement.set_student_train_mode()
        sums = defaultdict(float)
        seen = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, 1):
            inputs = _inputs(loaded, batch, settings)
            teacher = teacher_store.lookup(batch["samples"]).to(loaded.device).float()
            labels = torch.tensor(
                [sample.sku_index for sample in batch["samples"]],
                device=loaded.device,
                dtype=torch.long,
            )
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast(
                device_type=loaded.device.type,
                dtype=torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16,
                enabled=settings.amp_enabled and loaded.device.type == "cuda",
            )
            with autocast:
                student = student_embeddings(
                    loaded, replacement, inputs, settings.embedding_dim
                )
                kd = embedding_distillation_loss(student, teacher)
                retrieval = supervised_contrastive_loss(student, labels, settings.temperature)
                router = replacement.router_losses()
                dc = (
                    replacement.phase_dc_loss()
                    if settings.phase_dc_enabled and settings.phase_dc_weight > 0.0
                    else student.sum() * 0.0
                )
                loss = (
                    settings.lambda_kd * kd
                    + settings.lambda_ret * retrieval
                    + settings.lambda_router_balance * router["balance"]
                    + settings.lambda_router_importance * router["importance"]
                    + settings.phase_dc_weight * dc
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} batch={batch_index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(replacement.trainable_parameters()), settings.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["samples"])
            seen += count
            values = {
                "loss": loss,
                "kd_loss": kd,
                "retrieval_loss": retrieval,
                "router_balance_loss": router["balance"],
                "router_importance_loss": router["importance"],
                "phase_dc_loss": dc,
                "train_top1": torch.tensor(_batch_retrieval_accuracy(student.detach(), labels)),
                "vision_router_entropy": replacement.vision_surrogate.last_routing["normalized_entropy"],
            }
            if replacement.language_surrogate is not None:
                values["language_router_entropy"] = (
                    replacement.language_surrogate.optical_core.last_routing["normalized_entropy"]
                )
            for name, value in values.items():
                sums[name] += float(value.detach()) * count
            if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
                print(
                    f"epoch {epoch:03d}/{settings.epochs:03d} batch {batch_index:04d}/{len(loader):04d} "
                    f"loss={sums['loss']/seen:.5f} kd={sums['kd_loss']/seen:.5f} "
                    f"ret={sums['retrieval_loss']/seen:.5f} train_top1={sums['train_top1']/seen:.4f} "
                    f"balance={sums['router_balance_loss']/seen:.4f} "
                    f"v_entropy={sums['vision_router_entropy']/seen:.4f} "
                    + (
                        f"l_entropy={sums['language_router_entropy']/seen:.4f}"
                        if replacement.language_surrogate is not None else ""
                    ),
                    flush=True,
                )
        row = {"epoch": epoch, **{name: value / seen for name, value in sums.items()}}
        row["epoch_time_sec"] = time.perf_counter() - started
        row["samples_this_epoch"] = seen
        if settings.evaluate_test_each_epoch:
            evaluation = evaluate_student(
                loaded, replacement, bundle.test_samples, bundle.gallery_samples,
                bundle.class_names, settings, system_name="student_epoch",
            )
            row.update({
                "test_top1": evaluation.metrics["top1_retrieval_accuracy"],
                "test_top3": evaluation.metrics["top3_retrieval_accuracy"],
                "test_mrr": evaluation.metrics["mrr"],
            })
        history.append(row)
        _append_csv(history_path, row)
        save_checkpoint(
            replacement,
            optimizer,
            settings.output_dir / "checkpoints" / "last_checkpoint.pt",
            epoch,
            settings,
        )
        if row["loss"] < best_train_loss:
            best_train_loss = row["loss"]
            save_checkpoint(
                replacement,
                optimizer,
                settings.output_dir / "checkpoints" / "best_train_loss_checkpoint.pt",
                epoch,
                settings,
            )
        if "test_top1" in row and row["test_top1"] > best_observed_test_top1:
            best_observed_test_top1 = row["test_top1"]
            best_observed_test_epoch = epoch
            save_checkpoint(
                replacement,
                optimizer,
                settings.output_dir / "checkpoints" / "best_observed_test_top1_checkpoint.pt",
                epoch,
                settings,
            )
            write_json(
                settings.output_dir / "metrics" / "best_observed_test.json",
                {
                    "selection_biased": True,
                    "warning": "This checkpoint was selected by repeatedly observing the test set.",
                    "epoch": epoch,
                    "test_top1": row["test_top1"],
                    "test_top3": row["test_top3"],
                    "test_mrr": row["test_mrr"],
                },
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == settings.epochs:
            save_phase_preview(replacement, settings.output_dir / "figures", epoch)
        print(
            f"epoch {epoch:03d} complete train_loss={row['loss']:.5f} "
            f"train_top1={row['train_top1']:.4f} "
            + (f"test_top1={row['test_top1']:.4f} " if "test_top1" in row else "")
            + f"best_train_loss={best_train_loss:.5f} "
            + (
                f"best_observed_test_top1={best_observed_test_top1:.4f}@{best_observed_test_epoch}"
                if best_observed_test_epoch >= 0 else ""
            ),
            flush=True,
        )
    save_training_curves(history, settings.output_dir / "figures" / "training_curves.png")
    return {
        "best_train_loss": best_train_loss,
        "best_observed_test_top1": best_observed_test_top1,
        "best_observed_test_epoch": best_observed_test_epoch,
        "epochs": settings.epochs,
    }


@torch.no_grad()
def encode_samples(
    loaded: LoadedBackbone,
    replacement: TokenwiseVisionReplacement,
    samples: Sequence[GrocerySample],
    settings: Any,
) -> torch.Tensor:
    loader, _ = _loader(samples, settings, training=False)
    replacement.use_student()
    loaded.model.eval()
    replacement.vision_surrogate.eval()
    if replacement.language_surrogate is not None:
        replacement.language_surrogate.eval()
    chunks = []
    for batch in loader:
        inputs = _inputs(loaded, batch, settings)
        chunks.append(
            student_embeddings(loaded, replacement, inputs, settings.embedding_dim).cpu()
        )
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def evaluate_student(
    loaded: LoadedBackbone,
    replacement: TokenwiseVisionReplacement,
    queries: Sequence[GrocerySample],
    gallery: Sequence[GrocerySample],
    class_names: Sequence[str],
    settings: Any,
    *,
    system_name: str,
) -> RetrievalEvaluation:
    gallery_embeddings = encode_samples(loaded, replacement, gallery, settings)
    query_embeddings = encode_samples(loaded, replacement, queries, settings)
    return evaluate_embeddings(
        query_embeddings,
        queries,
        gallery_embeddings,
        gallery,
        class_names,
        settings.gallery_aggregation,
        system_name=system_name,
    )


def evaluate(
    loaded: LoadedBackbone,
    replacement: TokenwiseVisionReplacement,
    bundle: GroceryRetrievalBundle,
    teacher_store: TeacherEmbeddingStore,
    settings: Any,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    if checkpoint is None:
        filename = (
            "best_observed_test_top1_checkpoint.pt"
            if settings.evaluation_checkpoint == "best_observed_test"
            else "best_train_loss_checkpoint.pt"
        )
        checkpoint = settings.output_dir / "checkpoints" / filename
    payload = load_checkpoint(replacement, checkpoint, loaded.device)
    student = evaluate_student(
        loaded,
        replacement,
        bundle.test_samples,
        bundle.gallery_samples,
        bundle.class_names,
        settings,
        system_name="tokenwise_optical_student",
    )
    teacher = evaluate_embeddings(
        teacher_store.lookup(bundle.test_samples),
        bundle.test_samples,
        teacher_store.lookup(bundle.gallery_samples),
        bundle.gallery_samples,
        bundle.class_names,
        settings.gallery_aggregation,
        system_name="frozen_qwen_teacher",
    )
    root = settings.output_dir / "metrics"
    write_json(root / "student_metrics.json", student.metrics)
    write_json(root / "teacher_metrics.json", teacher.metrics)
    write_csv(
        root / "student_retrieval_results.csv",
        student.rows,
        list(student.rows[0]) if student.rows else [],
    )
    comparison = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "teacher": teacher.metrics,
        "student": student.metrics,
        "top1_retention_ratio": (
            student.metrics["top1_retrieval_accuracy"]
            / max(teacher.metrics["top1_retrieval_accuracy"], 1e-12)
        ),
        "manifest_digest": bundle.manifest_digest,
        "architecture": replacement.architecture_report(),
    }
    write_json(root / "comparison.json", comparison)
    save_confusion(student.confusion, bundle.class_names, settings.output_dir / "figures" / "confusion_matrix.png")
    save_routing_summary(replacement, root / "routing_summary.json")
    print(
        "Teacher retrieval: "
        f"Top-1={teacher.metrics['top1_retrieval_accuracy']:.4f} "
        f"Top-3={teacher.metrics['top3_retrieval_accuracy']:.4f} MRR={teacher.metrics['mrr']:.4f}"
    )
    print(
        "Token-wise optical retrieval: "
        f"Top-1={student.metrics['top1_retrieval_accuracy']:.4f} "
        f"Top-3={student.metrics['top3_retrieval_accuracy']:.4f} MRR={student.metrics['mrr']:.4f}"
    )
    return comparison


def save_checkpoint(
    replacement: TokenwiseVisionReplacement,
    optimizer: torch.optim.Optimizer,
    path: Path,
    epoch: int,
    settings: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        **replacement.state_dict(),
        "optimizer": optimizer.state_dict(),
        "settings": settings.to_dict(),
        "architecture": replacement.architecture_report(),
    }, path)


def load_checkpoint(
    replacement: TokenwiseVisionReplacement,
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("vision_tokenwise_optical") is None:
        raise RuntimeError("Checkpoint does not contain vision_tokenwise_optical")
    replacement.load_state_dict(payload)
    return payload


def save_phase_preview(replacement: TokenwiseVisionReplacement, root: Path, epoch: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _save_core_phase_preview(
        replacement.vision_surrogate, root / f"vision_phase_epoch_{epoch:03d}.png"
    )
    if replacement.language_surrogate is not None:
        _save_core_phase_preview(
            replacement.language_surrogate.optical_core,
            root / f"language_phase_epoch_{epoch:03d}.png",
        )


def _save_core_phase_preview(core: Any, path: Path) -> None:
    first = core.first_expert_phase.phase().detach().cpu()
    second = core.second_phase.phase().detach().cpu()
    if first.ndim == 4:
        first = first[0]
    columns = max(first.shape[0], 2)
    figure, axes = plt.subplots(2, columns, figsize=(3.2 * columns, 6.4), squeeze=False)
    for expert in range(first.shape[0]):
        image = axes[0, expert].imshow(first[expert], cmap="twilight", vmin=0, vmax=2 * torch.pi)
        axes[0, expert].set_title(f"stage1 expert {expert}")
        figure.colorbar(image, ax=axes[0, expert], fraction=0.046)
    for index in range(first.shape[0], columns):
        axes[0, index].axis("off")
    if core.settings.second_plane_mode == "global":
        image = axes[1, 0].imshow(second, cmap="twilight", vmin=0, vmax=2 * torch.pi)
        axes[1, 0].set_title("stage2 global")
        figure.colorbar(image, ax=axes[1, 0], fraction=0.046)
        for index in range(1, columns):
            axes[1, index].axis("off")
    else:
        if second.ndim == 4:
            second = second[0]
        for expert in range(second.shape[0]):
            image = axes[1, expert].imshow(second[expert], cmap="twilight", vmin=0, vmax=2 * torch.pi)
            axes[1, expert].set_title(f"stage2 expert {expert}")
            figure.colorbar(image, ax=axes[1, expert], fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_training_curves(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, [row["loss"] for row in history], label="total")
    axes[0].plot(epochs, [row["kd_loss"] for row in history], label="KD")
    axes[0].plot(epochs, [row["retrieval_loss"] for row in history], label="retrieval")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Training losses")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_top1"] for row in history], label="train batch top-1")
    if "test_top1" in history[0]:
        axes[1].plot(epochs, [row["test_top1"] for row in history], label="test top-1")
    axes[1].set(xlabel="epoch", ylabel="accuracy", ylim=(0, 1), title="Retrieval accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_confusion(matrix: torch.Tensor, names: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    axis.set_xticks(range(len(names)), labels=names, rotation=60, ha="right")
    axis.set_yticks(range(len(names)), labels=names)
    axis.set_xlabel("Predicted SKU")
    axis.set_ylabel("True SKU")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_routing_summary(replacement: TokenwiseVisionReplacement, path: Path) -> None:
    def summary(core: Any) -> dict[str, Any]:
        routing = core.last_routing
        return {
            "importance": routing.get("importance", torch.empty(0)).detach().cpu().tolist(),
            "load": routing.get("load", torch.empty(0)).detach().cpu().tolist(),
            "normalized_entropy": float(
                routing.get("normalized_entropy", torch.tensor(0.0))
            ),
            "token_counts": core.last_token_counts,
        }

    payload: dict[str, Any] = {
        "scope": "last evaluated batch; routing is independently computed per token",
        "vision": summary(replacement.vision_surrogate),
    }
    if replacement.language_surrogate is not None:
        payload["language"] = summary(replacement.language_surrogate.optical_core)
    write_json(path, payload)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
