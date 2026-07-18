from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .datasets import make_indexed_loader, sample_metadata
from .features import move_inputs, pool_token_groups, preprocess_images, run_visual
from .io_utils import write_csv, write_json
from .metrics import regression_metrics
from .modeling import NormalizedLinearRegressionHead, build_head
from .sampling import EpochRotatingSampler
from .teacher_cache import (TeacherCacheStore, load_teacher_predictions, pooled_teacher_features,
                            write_teacher_predictions)
from .visualization import save_debug_example, save_phase_masks, save_scatter, save_training_curves


def _score_metrics(predictions_normalized: torch.Tensor, targets_normalized: torch.Tensor) -> dict[str, float]:
    return regression_metrics((targets_normalized.float() * 100.0).tolist(),
                              (predictions_normalized.float() * 100.0).tolist())


def _split_indices(samples: int, fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.randperm(samples, generator=torch.Generator().manual_seed(seed))
    validation_count = min(max(int(round(samples * fraction)), 1), samples - 1)
    return order[validation_count:], order[:validation_count]


def train_teacher_head(train_store: TeacherCacheStore, test_store: TeacherCacheStore, settings: Any,
                       device: torch.device) -> NormalizedLinearRegressionHead:
    features, targets = pooled_teacher_features(train_store)
    train_indices, validation_indices = _split_indices(len(targets), settings.validation_fraction, settings.seed)
    head = build_head(settings, features.shape[1]).to(device)
    optimizer_class = torch.optim.AdamW if settings.optimizer_type == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(head.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs) if settings.scheduler_type == "cosine" else None
    history: list[dict[str, Any]] = []
    best_srcc = float("-inf")
    for epoch in range(1, settings.epochs + 1):
        head.train()
        total_loss = 0.0
        loader = DataLoader(TensorDataset(features[train_indices], targets[train_indices]),
                            batch_size=settings.head_batch_size, shuffle=True)
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(batch_features)
            loss = F.smooth_l1_loss(prediction, batch_targets, beta=settings.smooth_l1_beta)
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(batch_targets)
        validation_predictions = _head_predictions(head, features[validation_indices], settings.head_batch_size, device)
        metrics = _score_metrics(validation_predictions, targets[validation_indices])
        row = {"epoch": epoch, "train_loss": total_loss / len(train_indices),
               **{f"validation_{name}": value for name, value in metrics.items()}}
        history.append(row)
        write_csv(settings.output_dir / "metrics" / "teacher_training_history.csv", history, list(row))
        write_json(settings.output_dir / "metrics" / "teacher_training_latest.json", row)
        if metrics["srcc"] > best_srcc:
            best_srcc = metrics["srcc"]
            save_head(head, settings.output_dir / "checkpoints" / "teacher_head.pt", settings)
            write_json(settings.output_dir / "metrics" / "teacher_best_validation.json", row)
        if scheduler is not None:
            scheduler.step()
    head = load_head(settings.output_dir / "checkpoints" / "teacher_head.pt", settings, device)
    teacher_inference(head, test_store, settings, device)
    return head


def generate_teacher_predictions(head: nn.Module, stores: dict[str, TeacherCacheStore],
                                 settings: Any, device: torch.device) -> None:
    for split, store in stores.items():
        features, targets = pooled_teacher_features(store)
        predictions = _head_predictions(head, features, settings.head_batch_size, device)
        write_teacher_predictions(settings.output_dir, split, predictions, targets)


def teacher_inference(head: nn.Module, store: TeacherCacheStore, settings: Any,
                      device: torch.device) -> dict[str, Any]:
    features, targets = pooled_teacher_features(store)
    predictions = _head_predictions(head, features, settings.head_batch_size, device)
    report: dict[str, Any] = _score_metrics(predictions, targets)
    report.update({"dataset": "SPAQ", "task": "MOS", "score_scale": [0.0, 100.0],
                   "model": "complete_electronic_qwen3_vl_2b_vision_stack",
                   "feature_pooling": "valid_visual_token_mean", "language_model_used": False,
                   "head": head.specification() if hasattr(head, "specification") else {}})
    write_json(settings.output_dir / "metrics" / "teacher_inference.json", report)
    return report


class CachedStudentDataset(Dataset[Any]):
    def __init__(self, images: Dataset[Any], store: TeacherCacheStore, predictions: torch.Tensor) -> None:
        self.images = images; self.store = store; self.predictions = predictions

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image, target = self.images[index]
        cached = self.store.get(index)
        if abs(float(cached["target"]) - float(target)) > 1e-6:
            raise RuntimeError(f"Teacher cache MOS target mismatch at sample {index}")
        return (image, float(target), index, cached["image_grid_thw"], cached["visual_token_count"],
                cached["teacher_vision_stack_output"], self.predictions[index])


def cached_collate(batch: Sequence[Any]):
    images, targets, indices, grids, counts, hidden, predictions = zip(*batch)
    return (list(images), torch.tensor(targets, dtype=torch.float32), torch.tensor(indices),
            torch.stack(grids), torch.stack(counts).long(), list(hidden), torch.stack(predictions).float())


def _is_better(metric: str, current: float, best: float) -> bool:
    return current < best if metric == "mae" else current > best


def train_student(model: nn.Module, processor: Any, replacement: Any, head: NormalizedLinearRegressionHead,
                  train_dataset: Dataset[Any], test_dataset: Dataset[Any], train_store: TeacherCacheStore,
                  test_store: TeacherCacheStore, settings: Any, device: torch.device) -> None:
    teacher_predictions = load_teacher_predictions(settings.output_dir / "teacher_cache" / "train_teacher_predictions.pt")
    cached = CachedStudentDataset(train_dataset, train_store, teacher_predictions)
    sampler = EpochRotatingSampler(len(train_dataset), settings.train_samples_per_epoch, settings.seed)
    loader = DataLoader(cached, batch_size=settings.student_batch_size, sampler=sampler, num_workers=0,
                        collate_fn=cached_collate, pin_memory=torch.cuda.is_available())
    test_loader = make_indexed_loader(test_dataset, settings.inference_batch_size, settings.num_workers, False, settings.seed)
    model.requires_grad_(False).eval(); replacement.use_student()
    replacement.surrogate.requires_grad_(True).train(); head.requires_grad_(True).train()
    router_parameters = list(replacement.surrogate.prompt.router.parameters())
    router_ids = {id(parameter) for parameter in router_parameters}
    main_parameters = [parameter for parameter in [*replacement.surrogate.parameters(), *head.parameters()]
                       if id(parameter) not in router_ids]
    optimizer_class = torch.optim.AdamW if settings.optimizer_type == "adamw" else torch.optim.Adam
    optimizer = optimizer_class([
        {"params": main_parameters, "lr": settings.learning_rate, "group_name": "main"},
        {"params": router_parameters, "lr": settings.router_learning_rate, "group_name": "router"},
    ], weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs) if settings.scheduler_type == "cosine" else None
    history: list[dict[str, Any]] = []
    best_value = float("inf") if settings.student_selection_metric == "mae" else float("-inf")
    write_json(settings.output_dir / "metrics" / "student_selection_protocol.json", {
        "selection_split": settings.student_selection_split, "metric": settings.student_selection_metric,
        "evaluate_test_every_epoch": True,
        "warning": "The test split selects the best checkpoint; this best-test result is selection-biased.",
    })
    first_shapes_written = False
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter(); sampler.set_epoch(epoch)
        dropout_active = settings.phase_dropout_enabled and epoch >= settings.phase_dropout_start_epoch
        replacement.surrogate.set_phase_dropout_active(dropout_active)
        replacement.surrogate.train(); head.train()
        totals = {name: 0.0 for name in ("total", "hidden", "prediction_distill", "regression", "balance", "importance")}
        seen = 0; train_predictions: list[torch.Tensor] = []; train_targets: list[torch.Tensor] = []
        selection = torch.zeros(settings.num_experts); weight_sum = torch.zeros(settings.num_experts)
        print(f"[sampling] epoch={epoch} samples={len(sampler)}/{len(train_dataset)} phase_dropout={'on' if dropout_active else 'off'}", flush=True)
        for batch_index, (images, targets, _indices, cached_grids, cached_counts, teacher_hidden,
                          teacher_batch_predictions) in enumerate(loader, start=1):
            cpu_inputs = preprocess_images(processor, images)
            if not torch.equal(cpu_inputs["image_grid_thw"].cpu(), cached_grids.cpu()):
                raise RuntimeError("Current image_grid_thw differs from teacher cache; regenerate it")
            inputs = move_inputs(cpu_inputs, device); targets = targets.to(device)
            teacher_batch_predictions = teacher_batch_predictions.to(device)
            optimizer.zero_grad(set_to_none=True); run_visual(model, inputs)
            counts = replacement.surrogate.last_token_counts
            if counts != cached_counts.tolist():
                raise RuntimeError("Current visual token counts differ from teacher cache; regenerate it")
            student_groups = list(replacement.surrogate.last_output.split(counts, dim=0))
            predictions = head(pool_token_groups(student_groups))
            if not first_shapes_written:
                write_json(settings.output_dir / "metrics" / "first_batch_shapes.json", {
                    "pixel_values": list(cpu_inputs["pixel_values"].shape), "image_grid_thw": cpu_inputs["image_grid_thw"].tolist(),
                    "visual_token_counts": counts, "teacher_vision_hidden": [list(value.shape) for value in teacher_hidden],
                    "optical_input_fields": list(replacement.surrogate.last_input_fields.shape),
                    "full_detector_intensity": list(replacement.surrogate.last_detector_intensity.shape),
                    "detector_readout": list(replacement.surrogate.last_detector_readout.shape),
                    "student_vision_hidden": [list(value.shape) for value in student_groups],
                    "student_predictions": list(predictions.shape), "target_scale": [0.0, 1.0],
                }); first_shapes_written = True
            hidden_losses = [F.mse_loss(F.layer_norm(student.float(), (student.shape[-1],)),
                                        F.layer_norm(teacher.float().to(device), (teacher.shape[-1],)))
                             for student, teacher in zip(student_groups, teacher_hidden)]
            loss_hidden = torch.stack(hidden_losses).mean()
            loss_prediction_distill = F.smooth_l1_loss(predictions, teacher_batch_predictions,
                                                        beta=settings.smooth_l1_beta)
            loss_regression = F.smooth_l1_loss(predictions, targets, beta=settings.smooth_l1_beta)
            loss_balance, loss_importance = replacement.surrogate.router_losses()
            loss_total = (settings.loss_hidden_weight * loss_hidden +
                          settings.loss_prediction_distill_weight * loss_prediction_distill +
                          settings.loss_regression_weight * loss_regression +
                          settings.router_balance_weight * loss_balance +
                          settings.router_importance_weight * loss_importance)
            loss_total.backward(); optimizer.step()
            batch_size = len(targets); seen += batch_size
            for name, value in (("total", loss_total), ("hidden", loss_hidden),
                                ("prediction_distill", loss_prediction_distill), ("regression", loss_regression),
                                ("balance", loss_balance), ("importance", loss_importance)):
                totals[name] += float(value.detach()) * batch_size
            train_predictions.append(predictions.detach().cpu()); train_targets.append(targets.detach().cpu())
            routing = replacement.surrogate.last_routing
            selection += routing["selected_mask"].detach().cpu().float().sum(0)
            weight_sum += routing["weights"].detach().cpu().sum(0)
            if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
                current = _score_metrics(torch.cat(train_predictions), torch.cat(train_targets))
                expert_status = " ".join(f"e{i}:sel={float(selection[i]/seen):.3f},w|sel={float(weight_sum[i]/selection[i].clamp_min(1)):.3f}"
                                         for i in range(settings.num_experts))
                print(f"epoch {epoch}/{settings.epochs} batch {batch_index}/{len(loader)} loss={totals['total']/seen:.5f} "
                      f"hidden={totals['hidden']/seen:.5f} distill={totals['prediction_distill']/seen:.5f} "
                      f"mos={totals['regression']/seen:.5f} MAE={current['mae']:.3f} SRCC={current['srcc']:.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.3e} router_lr={optimizer.param_groups[1]['lr']:.3e} "
                      f"experts=[{expert_status}]", flush=True)
        train_report = _score_metrics(torch.cat(train_predictions), torch.cat(train_targets))
        test_report = evaluate_student(model, processor, replacement, head, test_loader, device, test_dataset)
        if scheduler is not None: scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "router_learning_rate": optimizer.param_groups[1]["lr"],
            **{f"loss_{name}": value / seen for name, value in totals.items()},
            **{f"train_{name}": value for name, value in train_report.items()},
            **{f"test_{name}": value for name, value in test_report.items() if isinstance(value, (int, float))},
            "epoch_time_sec": time.perf_counter() - started, "phase_dropout_active": dropout_active,
            "samples_this_epoch": len(sampler), "checkpoint_selection_split": settings.student_selection_split,
            "checkpoint_selection_metric": settings.student_selection_metric,
        }
        for expert in range(settings.num_experts):
            row[f"expert_{expert}_selection_rate"] = float(selection[expert] / seen)
            row[f"expert_{expert}_mean_routing_weight"] = float(weight_sum[expert] / seen)
            row[f"expert_{expert}_mean_selected_weight"] = float(weight_sum[expert] / selection[expert].clamp_min(1))
        history.append(row)
        write_csv(settings.output_dir / "metrics" / "student_training_history.csv", history, list(row))
        write_json(settings.output_dir / "metrics" / "student_training_latest.json", row)
        if settings.visualization_enabled and settings.save_training_curves:
            save_training_curves(history, settings.output_dir / "figures" / "student_training_curves.png")
        save_student_parts(settings.output_dir, replacement, head, "last", epoch, row)
        if epoch % settings.checkpoint_interval_epochs == 0:
            save_student_parts(settings.output_dir, replacement, head, f"epoch_{epoch:04d}", epoch, row)
        current_value = float(test_report[settings.student_selection_metric])
        if _is_better(settings.student_selection_metric, current_value, best_value):
            best_value = current_value
            save_student_parts(settings.output_dir, replacement, head, "best", epoch, row)
            write_json(settings.output_dir / "metrics" / "best_test.json", row)
            if settings.visualization_enabled and settings.save_phase_masks:
                save_phase_masks(replacement.surrogate, settings.output_dir / "figures" / "phase_masks_best.png", f"Best epoch {epoch}")
        if settings.visualization_enabled and settings.save_phase_masks and epoch % settings.visualization_interval_epochs == 0:
            save_phase_masks(replacement.surrogate, settings.output_dir / "figures" / f"phase_masks_epoch_{epoch:04d}.png", f"Epoch {epoch}")
        debug_saved = 0
        if settings.visualization_enabled and settings.save_intermediate_fields and epoch % settings.visualization_interval_epochs == 0:
            debug_saved = save_epoch_debug_examples(model, processor, replacement, head, test_dataset, test_store,
                                                      settings, device, epoch)
        print(f"epoch {epoch:03d} complete train_MAE={train_report['mae']:.3f} train_SRCC={train_report['srcc']:.4f} "
              f"test_MAE={test_report['mae']:.3f} test_SRCC={test_report['srcc']:.4f} "
              f"best_{settings.student_selection_metric}={best_value:.4f} debug_examples={debug_saved}", flush=True)
    write_json(settings.output_dir / "metrics" / "student_training.json", {
        "epochs": settings.epochs, "best_metric": settings.student_selection_metric, "best_value": best_value,
        "checkpoint_selection_split": settings.student_selection_split, "test_selection_bias_warning": True,
    })


@torch.inference_mode()
def save_epoch_debug_examples(model: nn.Module, processor: Any, replacement: Any, head: nn.Module,
                              dataset: Dataset[Any], store: TeacherCacheStore, settings: Any,
                              device: torch.device, epoch: int) -> int:
    count = min(settings.visualization_sample_count, len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(settings.seed + 1009 * epoch))[:count].tolist()
    replacement.use_student(); replacement.surrogate.eval(); replacement.surrogate.set_debug_capture(True); head.eval()
    try:
        for index in indices:
            image, target = dataset[index]; cached = store.get(index)
            inputs = move_inputs(preprocess_images(processor, [image]), device); run_visual(model, inputs)
            token_count = replacement.surrogate.last_token_counts[0]
            student_hidden = replacement.surrogate.last_output[:token_count].detach().cpu()
            teacher_hidden = cached["teacher_vision_stack_output"].detach().cpu().float()
            prediction = float(head(student_hidden.float().to(device).mean(0, keepdim=True))[0]) * 100.0
            save_debug_example(settings.output_dir / "figures" / "debug_examples" / f"epoch_{epoch:04d}" / f"sample_{index:05d}",
                               image, index, float(target) * 100.0, prediction,
                               replacement.surrogate.last_input_fields[0].detach().cpu(), replacement.surrogate.last_routing,
                               replacement.surrogate.last_detector_intensity[0].detach().cpu(), student_hidden, teacher_hidden, epoch)
        return count
    finally:
        replacement.surrogate.set_debug_capture(False)


@torch.inference_mode()
def evaluate_student(model: nn.Module, processor: Any, replacement: Any, head: nn.Module, loader: Any,
                     device: torch.device, dataset: Dataset[Any] | None = None,
                     predictions_path: Path | None = None) -> dict[str, Any]:
    replacement.use_student(); replacement.surrogate.eval(); replacement.surrogate.set_phase_dropout_active(False); head.eval()
    predictions_all: list[torch.Tensor] = []; targets_all: list[torch.Tensor] = []; indices_all: list[torch.Tensor] = []
    selection = torch.zeros(replacement.surrogate.geometry.num_experts); weight_sum = torch.zeros_like(selection); seen = 0
    for images, targets, indices in loader:
        run_visual(model, move_inputs(preprocess_images(processor, images), device))
        groups = list(replacement.surrogate.last_output.split(replacement.surrogate.last_token_counts, dim=0))
        predictions_all.append(head(pool_token_groups(groups)).cpu()); targets_all.append(targets.cpu()); indices_all.append(indices.cpu())
        routing = replacement.surrogate.last_routing; seen += len(targets)
        selection += routing["selected_mask"].cpu().float().sum(0); weight_sum += routing["weights"].cpu().sum(0)
    predictions = torch.cat(predictions_all); targets = torch.cat(targets_all); indices = torch.cat(indices_all)
    report: dict[str, Any] = _score_metrics(predictions, targets)
    report.update({"dataset": "SPAQ", "task": "MOS", "score_scale": [0.0, 100.0],
                   "model": "qwen3_vl_2b_vision_homogeneous_optical_moe9x5", "language_model_used": False,
                   "routing": {f"expert_{i}": {"selection_rate": float(selection[i] / max(seen, 1)),
                                                 "mean_weight": float(weight_sum[i] / max(seen, 1))}
                               for i in range(len(selection))}})
    if predictions_path is not None:
        rows = []
        for index, target, prediction in zip(indices.tolist(), targets.tolist(), predictions.tolist()):
            metadata = sample_metadata(dataset, int(index)) if dataset is not None else {"sample_index": int(index)}
            rows.append({**metadata, "sample_index": int(index), "true_score": float(target) * 100.0,
                         "predicted_score": float(prediction) * 100.0,
                         "absolute_error": abs(float(prediction - target)) * 100.0})
        if rows: write_csv(predictions_path, rows, list(rows[0]))
    return report


def save_student_inference(report: dict[str, Any], settings: Any, replacement: Any,
                           predictions_path: Path | None = None) -> None:
    report = {**report, "head_output_activation": settings.head_output_activation,
              "phase_dropout_disabled_for_inference": True,
              "parameter_breakdown": replacement.surrogate.parameter_breakdown()}
    write_json(settings.output_dir / "metrics" / "student_inference.json", report)
    if settings.visualization_enabled and settings.save_scatter_plot and predictions_path and predictions_path.is_file():
        import csv
        with predictions_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        save_scatter([float(row["true_score"]) for row in rows], [float(row["predicted_score"]) for row in rows],
                     settings.output_dir / "figures" / "student_mos_scatter.png", "SPAQ MOS optical student")


def save_head(head: NormalizedLinearRegressionHead, path: Path, settings: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "head": head.specification(), "task": "MOS",
                "target_scale": [0.0, 1.0], "smooth_l1_beta": settings.smooth_l1_beta}, path)


def load_head(path: Path, settings: Any, device: torch.device) -> NormalizedLinearRegressionHead:
    if not path.is_file():
        raise FileNotFoundError(f"Teacher head checkpoint missing: {path}. Run --phase teacher_train first.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    head = build_head(settings, settings.vision_hidden_size).to(device)
    saved = payload.get("head", {})
    if saved and saved.get("output_activation") != settings.head_output_activation:
        raise RuntimeError("Teacher head output activation differs from current config; use a new output_dir or retrain it")
    head.load_state_dict(payload["state_dict"]); return head


def save_student_parts(output_dir: Path, replacement: Any, head: nn.Module, tag: str,
                       epoch: int, metrics: dict[str, Any]) -> None:
    checkpoint = output_dir / "checkpoints"; checkpoint.mkdir(parents=True, exist_ok=True)
    metadata = {"epoch": epoch, "metrics": metrics, "task": "MOS",
                "head": head.specification() if hasattr(head, "specification") else {}}
    torch.save({"state_dict": replacement.surrogate.state_dict(), **metadata}, checkpoint / f"vision_moe_{tag}.pt")
    torch.save({"state_dict": head.state_dict(), **metadata}, checkpoint / f"student_head_{tag}.pt")


def load_student_parts(output_dir: Path, replacement: Any, head: nn.Module, tag: str) -> None:
    checkpoint = output_dir / "checkpoints"
    surrogate_path = checkpoint / f"vision_moe_{tag}.pt"; head_path = checkpoint / f"student_head_{tag}.pt"
    if not surrogate_path.is_file() or not head_path.is_file():
        raise FileNotFoundError(f"Student checkpoint '{tag}' is incomplete under {checkpoint}")
    replacement.surrogate.load_state_dict(torch.load(surrogate_path, map_location="cpu", weights_only=True)["state_dict"])
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True)["state_dict"])


@torch.inference_mode()
def _head_predictions(head: nn.Module, features: torch.Tensor, batch_size: int,
                      device: torch.device) -> torch.Tensor:
    head.eval(); values = []
    for start in range(0, len(features), batch_size):
        values.append(head(features[start:start + batch_size].to(device)).cpu())
    return torch.cat(values)
