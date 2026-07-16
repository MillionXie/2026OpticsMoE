from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from .datasets import indexed_collate, labels_of, make_indexed_loader, stratified_split_indices
from .features import move_inputs, pool_token_groups, preprocess_images, run_visual
from .io_utils import write_csv, write_json
from .metrics import metrics_from_logits
from .modeling import NormalizedLinearHead, build_head
from .sampling import EpochClassMixedSampler
from .teacher_cache import TeacherCacheStore, load_teacher_logits, pooled_teacher_features, write_teacher_logits
from .visualization import save_confusion_matrix, save_phase_masks, save_training_curves


def train_teacher_head(train_store: TeacherCacheStore, test_store: TeacherCacheStore, settings: Any,
                       class_names: Sequence[str], device: torch.device) -> NormalizedLinearHead:
    features, labels = pooled_teacher_features(train_store)
    train_indices, validation_indices = _stratified_tensor_split(labels, settings.validation_fraction, settings.seed)
    head = build_head(settings, features.shape[1], len(class_names)).to(device)
    optimizer_class = torch.optim.AdamW if settings.optimizer_type == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(head.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs) if settings.scheduler_type == "cosine" else None
    history: list[dict[str, Any]] = []
    best = -1.0
    for epoch in range(1, settings.epochs + 1):
        head.train()
        total_loss = 0.0
        loader = DataLoader(TensorDataset(features[train_indices], labels[train_indices]), batch_size=settings.head_batch_size, shuffle=True)
        for batch_features, batch_labels in loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_labels)
        validation_logits = _head_logits(head, features[validation_indices], settings.head_batch_size, device)
        metrics = metrics_from_logits(validation_logits, labels[validation_indices], class_names)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_indices),
               "validation_top1_accuracy": metrics["top1_accuracy"], "validation_top5_accuracy": metrics["top5_accuracy"],
               "validation_macro_f1": metrics["macro_f1"], "validation_balanced_accuracy": metrics["balanced_accuracy"]}
        history.append(row)
        write_csv(settings.output_dir / "metrics" / "teacher_training_history.csv", history, list(row))
        if scheduler is not None:
            scheduler.step()
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            save_head(head, settings.output_dir / "checkpoints" / "teacher_head.pt", settings, len(class_names))
    head = load_head(settings.output_dir / "checkpoints" / "teacher_head.pt", settings, device)
    teacher_inference(head, test_store, settings, class_names, device)
    return head


def generate_teacher_logits(head: nn.Module, stores: dict[str, TeacherCacheStore], settings: Any,
                            device: torch.device) -> None:
    for split, store in stores.items():
        features, labels = pooled_teacher_features(store)
        write_teacher_logits(settings.output_dir, split, _head_logits(head, features, settings.head_batch_size, device), labels)


def teacher_inference(head: nn.Module, store: TeacherCacheStore, settings: Any,
                      class_names: Sequence[str], device: torch.device) -> dict[str, Any]:
    features, labels = pooled_teacher_features(store)
    report = metrics_from_logits(_head_logits(head, features, settings.head_batch_size, device), labels, class_names)
    report.update({"model": "complete_electronic_qwen3_vl_2b_vision_stack", "feature_pooling": "valid_visual_token_mean",
                   "head": "LayerNorm(vision_hidden_size) -> Linear(num_classes)"})
    write_json(settings.output_dir / "metrics" / "teacher_inference.json", report)
    return report


class CachedStudentDataset(Dataset[Any]):
    def __init__(self, images: Dataset[Any], store: TeacherCacheStore, logits: torch.Tensor) -> None:
        self.images = images
        self.store = store
        self.logits = logits

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image, label = self.images[index]
        target = self.store.get(index)
        if int(target["label"]) != int(label):
            raise RuntimeError(f"Teacher cache label mismatch at sample {index}")
        return image, int(label), index, target["image_grid_thw"], target["visual_token_count"], target["teacher_vision_stack_output"], self.logits[index]


def cached_collate(batch: Sequence[Any]):
    images, labels, indices, grids, counts, hidden, logits = zip(*batch)
    return list(images), torch.tensor(labels), torch.tensor(indices), torch.stack(grids), torch.stack(counts).long(), list(hidden), torch.stack(logits)


def train_student(model: nn.Module, processor: Any, replacement: Any, head: NormalizedLinearHead,
                  train_dataset: Dataset[Any], train_store: TeacherCacheStore, settings: Any,
                  class_names: Sequence[str], device: torch.device) -> None:
    teacher_logits = load_teacher_logits(settings.output_dir / "teacher_cache" / "train_teacher_logits.pt")
    train_indices, validation_indices = stratified_split_indices(train_dataset, settings.validation_fraction, settings.seed)
    cached = CachedStudentDataset(train_dataset, train_store, teacher_logits)
    sampler = EpochClassMixedSampler(train_indices, labels_of(train_dataset), len(class_names), settings.student_batch_size,
                                     settings.seed, settings.train_samples_per_class_per_epoch,
                                     settings.teacher_cache_shard_size)
    loader = DataLoader(cached, batch_size=settings.student_batch_size, sampler=sampler,
                        num_workers=0, collate_fn=cached_collate, pin_memory=torch.cuda.is_available())
    validation_loader = make_indexed_loader(Subset(train_dataset, validation_indices), settings.inference_batch_size,
                                            settings.num_workers, False, settings.seed)
    model.requires_grad_(False).eval()
    replacement.use_student()
    replacement.surrogate.requires_grad_(True).train()
    head.requires_grad_(True).train()
    parameters = [*replacement.surrogate.parameters(), *head.parameters()]
    optimizer_class = torch.optim.AdamW if settings.optimizer_type == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs) if settings.scheduler_type == "cosine" else None
    history: list[dict[str, Any]] = []
    best_top1 = -1.0
    first_shapes_written = False
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter()
        sampler.set_epoch(epoch)
        phase_dropout_active = settings.phase_dropout_enabled and epoch >= settings.phase_dropout_start_epoch
        replacement.surrogate.set_phase_dropout_active(phase_dropout_active)
        replacement.surrogate.train()
        head.train()
        totals = {name: 0.0 for name in ("total", "hidden", "kd", "ce", "balance", "importance")}
        seen = 0
        train_logits: list[torch.Tensor] = []
        train_labels: list[torch.Tensor] = []
        selection = torch.zeros(settings.num_experts)
        weight_sum = torch.zeros(settings.num_experts)
        last_log_time = time.perf_counter()
        print(f"[sampling] epoch={epoch} samples={len(sampler)} per_class={sampler.epoch_class_counts()} "
              f"phase_dropout={'on' if phase_dropout_active else 'off'}", flush=True)
        for batch_index, (images, labels, _indices, cached_grids, cached_counts, teacher_hidden, teacher_batch_logits) in enumerate(loader, start=1):
            cpu_inputs = preprocess_images(processor, images)
            if not torch.equal(cpu_inputs["image_grid_thw"].cpu(), cached_grids.cpu()):
                raise RuntimeError("Current image_grid_thw differs from teacher cache; regenerate teacher_precompute")
            inputs = move_inputs(cpu_inputs, device)
            labels = labels.to(device)
            teacher_batch_logits = teacher_batch_logits.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            run_visual(model, inputs)
            counts = replacement.surrogate.last_token_counts
            if counts != cached_counts.tolist():
                raise RuntimeError("Current visual token counts differ from teacher cache; regenerate teacher_precompute")
            student_groups = list(replacement.surrogate.last_output.split(counts, dim=0))
            pooled = pool_token_groups(student_groups)
            logits = head(pooled)
            if not first_shapes_written:
                write_json(settings.output_dir / "metrics" / "first_batch_shapes.json", {
                    "pixel_values": list(cpu_inputs["pixel_values"].shape),
                    "image_grid_thw": cpu_inputs["image_grid_thw"].tolist(),
                    "visual_token_counts": counts,
                    "teacher_vision_hidden": [list(value.shape) for value in teacher_hidden],
                    "optical_input_fields": list(replacement.surrogate.last_input_fields.shape),
                    "full_detector_intensity": list(replacement.surrogate.last_detector_intensity.shape),
                    "detector_readout": list(replacement.surrogate.last_detector_readout.shape),
                    "student_vision_hidden": [list(value.shape) for value in student_groups],
                    "student_logits": list(logits.shape),
                })
                first_shapes_written = True
            hidden_losses = []
            for student, teacher in zip(student_groups, teacher_hidden):
                teacher = teacher.float().to(device)
                hidden_losses.append(F.mse_loss(F.layer_norm(student.float(), (student.shape[-1],)),
                                                F.layer_norm(teacher, (teacher.shape[-1],))))
            loss_hidden = torch.stack(hidden_losses).mean()
            temperature = settings.distill_temperature
            loss_kd = temperature ** 2 * F.kl_div(F.log_softmax(logits / temperature, dim=1),
                                                  F.softmax(teacher_batch_logits / temperature, dim=1), reduction="batchmean")
            loss_ce = F.cross_entropy(logits, labels)
            loss_balance, loss_importance = replacement.surrogate.router_losses()
            loss_total = (settings.loss_hidden_weight * loss_hidden + settings.loss_kd_weight * loss_kd +
                          settings.loss_ce_weight * loss_ce + settings.router_balance_weight * loss_balance +
                          settings.router_importance_weight * loss_importance)
            loss_total.backward()
            optimizer.step()
            batch_size = len(labels)
            seen += batch_size
            for name, value in (("total", loss_total), ("hidden", loss_hidden), ("kd", loss_kd), ("ce", loss_ce),
                                ("balance", loss_balance), ("importance", loss_importance)):
                totals[name] += float(value.detach()) * batch_size
            train_logits.append(logits.detach().cpu())
            train_labels.append(labels.detach().cpu())
            routing = replacement.surrogate.last_routing
            selection += routing["selected_mask"].detach().cpu().float().sum(0)
            weight_sum += routing["weights"].detach().cpu().sum(0)
            now = time.perf_counter()
            should_log = (batch_index % settings.log_interval_batches == 0 or batch_index == len(loader) or
                          now - last_log_time >= settings.log_interval_seconds)
            if should_log:
                running = metrics_from_logits(torch.cat(train_logits), torch.cat(train_labels), class_names)
                print(f"epoch {epoch}/{settings.epochs} batch {batch_index}/{len(loader)} "
                      f"loss={totals['total']/seen:.5f} hidden={totals['hidden']/seen:.5f} "
                      f"kd={totals['kd']/seen:.5f} ce={totals['ce']/seen:.5f} "
                      f"balance={totals['balance']/seen:.5f} top1={running['top1_accuracy']:.4f}", flush=True)
                last_log_time = now
        train_report = metrics_from_logits(torch.cat(train_logits), torch.cat(train_labels), class_names)
        validation = evaluate_student(model, processor, replacement, head, validation_loader, class_names, device)
        if scheduler is not None:
            scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"loss_{name}": value / seen for name, value in totals.items()},
            "train_top1_accuracy": train_report["top1_accuracy"], "train_macro_f1": train_report["macro_f1"],
            "validation_top1_accuracy": validation["top1_accuracy"], "validation_top5_accuracy": validation["top5_accuracy"],
            "validation_macro_f1": validation["macro_f1"], "validation_balanced_accuracy": validation["balanced_accuracy"],
            "epoch_time_sec": time.perf_counter() - started,
            "phase_dropout_active": phase_dropout_active,
            "samples_this_epoch": len(sampler),
        }
        for expert in range(settings.num_experts):
            row[f"expert_{expert}_selection_rate"] = float(selection[expert] / seen)
            row[f"expert_{expert}_mean_routing_weight"] = float(weight_sum[expert] / seen)
        history.append(row)
        write_csv(settings.output_dir / "metrics" / "student_training_history.csv", history, list(row))
        write_json(settings.output_dir / "metrics" / "student_training_latest.json", row)
        if settings.visualization_enabled and settings.save_training_curves:
            save_training_curves(history, settings.output_dir / "figures" / "student_training_curves.png")
        save_student_parts(settings.output_dir, replacement, head, "last", epoch, row)
        if epoch % settings.checkpoint_interval_epochs == 0:
            save_student_parts(settings.output_dir, replacement, head, f"epoch_{epoch:04d}", epoch, row)
        if validation["top1_accuracy"] > best_top1:
            best_top1 = validation["top1_accuracy"]
            save_student_parts(settings.output_dir, replacement, head, "best", epoch, row)
            write_json(settings.output_dir / "metrics" / "best_validation.json", row)
            if settings.visualization_enabled and settings.save_phase_masks:
                save_phase_masks(replacement.surrogate, settings.output_dir / "figures" / "phase_masks_best.png", f"Best epoch {epoch}")
        if (settings.visualization_enabled and settings.save_phase_masks and
                epoch % settings.visualization_interval_epochs == 0):
            save_phase_masks(replacement.surrogate, settings.output_dir / "figures" / f"phase_masks_epoch_{epoch:04d}.png", f"Epoch {epoch}")
    write_json(settings.output_dir / "metrics" / "student_training.json", {"epochs": settings.epochs, "best_validation_top1": best_top1})


@torch.inference_mode()
def evaluate_student(model: nn.Module, processor: Any, replacement: Any, head: nn.Module, loader: Any,
                     class_names: Sequence[str], device: torch.device, save_predictions: Path | None = None) -> dict[str, Any]:
    replacement.use_student()
    replacement.surrogate.eval()
    head.eval()
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    indices_all: list[torch.Tensor] = []
    routing_weight_sum = torch.zeros(replacement.surrogate.geometry.num_experts)
    selection_sum = torch.zeros(replacement.surrogate.geometry.num_experts)
    sample_count = 0
    for images, labels, indices in loader:
        run_visual(model, move_inputs(preprocess_images(processor, images), device))
        groups = list(replacement.surrogate.last_output.split(replacement.surrogate.last_token_counts, dim=0))
        logits_all.append(head(pool_token_groups(groups)).cpu())
        labels_all.append(labels.cpu())
        indices_all.append(indices.cpu())
        routing = replacement.surrogate.last_routing
        routing_weight_sum += routing["weights"].detach().cpu().sum(0)
        selection_sum += routing["selected_mask"].detach().cpu().float().sum(0)
        sample_count += len(labels)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    report = metrics_from_logits(logits, labels, class_names)
    report["mean_routing_weights"] = (routing_weight_sum / max(1, sample_count)).tolist()
    report["expert_selection_rates"] = (selection_sum / max(1, sample_count)).tolist()
    if save_predictions is not None:
        predictions = logits.argmax(1)
        rows = []
        for index, truth, prediction, values in zip(torch.cat(indices_all).tolist(), labels.tolist(), predictions.tolist(), logits.tolist()):
            row = {"sample_index": index, "true_label": truth, "true_name": class_names[truth], "pred_label": prediction,
                   "pred_name": class_names[prediction], "correct": truth == prediction}
            row.update({f"logit_{name}": value for name, value in zip(class_names, values)})
            rows.append(row)
        write_csv(save_predictions, rows, list(rows[0]))
    return report


def save_student_inference(report: dict[str, Any], settings: Any, replacement: Any) -> None:
    report = {**report, "model": "qwen3_vl_2b_vision_homogeneous_moe9x5", "language_model_used": False,
              "detector": "full 480x480 intensity plane; no class regions",
              "detector_readout": "AvgPool4 -> non-affine LayerNorm(120x120) -> ReLU -> first T rows",
              "router_balance_weight": settings.router_balance_weight}
    write_json(settings.output_dir / "metrics" / "student_inference.json", report)
    matrix = report["confusion_matrix"]
    rows = [{"true_class": index, **{f"pred_{column}": value for column, value in enumerate(row)}} for index, row in enumerate(matrix)]
    write_csv(settings.output_dir / "metrics" / "student_confusion_matrix.csv", rows, list(rows[0]))
    if settings.visualization_enabled and settings.save_confusion_matrix:
        save_confusion_matrix(matrix, ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
                              settings.output_dir / "figures" / "student_confusion_matrix.png")


def save_head(head: NormalizedLinearHead, path: Path, settings: Any, num_classes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "head": head.specification(), "num_classes": num_classes,
                "vision_hidden_size": settings.vision_hidden_size}, path)


def load_head(path: Path, settings: Any, device: torch.device) -> NormalizedLinearHead:
    if not path.is_file():
        raise FileNotFoundError(f"Teacher head checkpoint missing: {path}. Run --phase teacher_train first.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("head", {}).get("type") != "normalized_linear":
        raise RuntimeError("Teacher checkpoint is not the required normalized_linear head")
    head = build_head(settings, int(payload["vision_hidden_size"]), int(payload["num_classes"])).to(device)
    head.load_state_dict(payload["state_dict"])
    return head


def save_student_parts(output_dir: Path, replacement: Any, head: nn.Module, tag: str, epoch: int, metrics: dict[str, Any]) -> None:
    metadata = {"epoch": epoch, "metrics": metrics, "replacement": "complete_vision_stack_homogeneous_moe9x5",
                "head": head.specification()}
    torch.save({"state_dict": replacement.surrogate.state_dict(), "metadata": metadata}, output_dir / "checkpoints" / f"vision_homogeneous_moe_{tag}.pt")
    torch.save({"state_dict": head.state_dict(), "metadata": metadata}, output_dir / "checkpoints" / f"student_head_{tag}.pt")


def load_student_parts(output_dir: Path, replacement: Any, head: nn.Module, tag: str) -> None:
    surrogate_path = output_dir / "checkpoints" / f"vision_homogeneous_moe_{tag}.pt"
    head_path = output_dir / "checkpoints" / f"student_head_{tag}.pt"
    if not surrogate_path.is_file() or not head_path.is_file():
        raise FileNotFoundError(f"Student {tag} checkpoints are missing; run --phase student_train first")
    replacement.surrogate.load_state_dict(torch.load(surrogate_path, map_location="cpu", weights_only=True)["state_dict"])
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True)["state_dict"])


@torch.inference_mode()
def _head_logits(head: nn.Module, features: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    head.eval()
    chunks = []
    for start in range(0, len(features), batch_size):
        chunks.append(head(features[start:start + batch_size].to(device)).cpu())
    return torch.cat(chunks)


def _stratified_tensor_split(labels: torch.Tensor, fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train: list[int] = []
    validation: list[int] = []
    for class_index in range(int(labels.max()) + 1):
        indices = torch.where(labels == class_index)[0]
        order = indices[torch.randperm(len(indices), generator=generator)].tolist()
        count = min(max(int(round(len(order) * fraction)), 1), len(order) - 1) if len(order) > 1 else 0
        validation.extend(order[:count])
        train.extend(order[count:])
    return torch.tensor(train), torch.tensor(validation)
