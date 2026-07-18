from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .datasets import make_indexed_loader, sample_metadata, targets_of
from .features import move_inputs, pool_token_groups, preprocess_images, run_visual
from .io_utils import write_csv, write_json
from .metrics import regression_metrics
from .modeling import NormalizedLinearRegressionHead, build_head
from .processor_cache import ProcessorCacheStore
from .sampling import EpochRotatingSampler
from .teacher_cache import (TeacherCacheStore, load_teacher_predictions, pooled_teacher_features,
                            write_teacher_predictions)
from .visualization import save_debug_example, save_phase_masks, save_scatter, save_training_curves


def _score_metrics(predictions_normalized: torch.Tensor, targets_normalized: torch.Tensor) -> dict[str, float]:
    predictions = predictions_normalized.float().reshape(-1)
    report = regression_metrics((targets_normalized.float().reshape(-1) * 100.0).tolist(),
                                (predictions * 100.0).tolist())
    report.update({
        "prediction_min_normalized": float(predictions.min()),
        "prediction_max_normalized": float(predictions.max()),
        "prediction_mean_normalized": float(predictions.mean()),
        "prediction_std_normalized": float(predictions.std(unbiased=False)),
        "prediction_out_of_range_ratio": float(((predictions < 0.0) | (predictions > 1.0)).float().mean()),
        "prediction_boundary_ratio": float(((predictions <= 1e-6) | (predictions >= 1.0 - 1e-6)).float().mean()),
    })
    return report


def _split_indices(samples: int, fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.randperm(samples, generator=torch.Generator().manual_seed(seed))
    validation_count = min(max(int(round(samples * fraction)), 1), samples - 1)
    return order[validation_count:], order[:validation_count]


def train_teacher_head(train_store: TeacherCacheStore, test_store: TeacherCacheStore,
                       train_dataset: Dataset[Any], test_dataset: Dataset[Any], settings: Any,
                       device: torch.device) -> NormalizedLinearRegressionHead:
    features, targets = pooled_teacher_features(train_store, targets_of(train_dataset))
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
    teacher_inference(head, test_store, test_dataset, settings, device)
    return head


def generate_teacher_predictions(head: nn.Module, stores: dict[str, TeacherCacheStore],
                                 datasets: dict[str, Dataset[Any]], settings: Any,
                                 device: torch.device) -> None:
    for split, store in stores.items():
        features, targets = pooled_teacher_features(store, targets_of(datasets[split]))
        predictions = _head_predictions(head, features, settings.head_batch_size, device)
        write_teacher_predictions(settings.output_dir, split, predictions, targets, head.specification())


def teacher_inference(head: nn.Module, store: TeacherCacheStore, dataset: Dataset[Any], settings: Any,
                      device: torch.device) -> dict[str, Any]:
    features, targets = pooled_teacher_features(store, targets_of(dataset))
    predictions = _head_predictions(head, features, settings.head_batch_size, device)
    report: dict[str, Any] = _score_metrics(predictions, targets)
    report.update({"dataset": "SPAQ", "task": settings.task_name, "score_scale": [0.0, 100.0],
                   "model": "complete_electronic_qwen3_vl_2b_vision_stack",
                   "feature_pooling": "valid_visual_token_mean", "language_model_used": False,
                   "head": head.specification() if hasattr(head, "specification") else {}})
    write_json(settings.output_dir / "metrics" / "teacher_inference.json", report)
    return report


class CachedStudentDataset(Dataset[Any]):
    def __init__(self, images: Dataset[Any], store: TeacherCacheStore, input_store: ProcessorCacheStore,
                 predictions: torch.Tensor) -> None:
        self.images = images; self.store = store; self.input_store = input_store; self.predictions = predictions

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        target = _cached_target(self.images, index)
        cached = self.store.get(index)
        processor_input = self.input_store.get(index)
        # The electronic visual hidden cache is task-independent. Its legacy
        # cached target may be MOS when source_cache_run_dir points at the MOS
        # experiment, so the configured attribute target always comes from
        # the current dataset instead.
        if not torch.equal(cached["image_grid_thw"], processor_input["image_grid_thw"]):
            raise RuntimeError(f"Processor and teacher cache image_grid_thw mismatch at sample {index}")
        if int(cached["visual_token_count"]) != int(processor_input["visual_token_count"]):
            raise RuntimeError(f"Processor and teacher cache token count mismatch at sample {index}")
        return (processor_input["pixel_values"], float(target), index, cached["image_grid_thw"], cached["visual_token_count"],
                cached["teacher_vision_stack_output"], self.predictions[index])


def cached_collate(batch: Sequence[Any]):
    pixel_values, targets, indices, grids, counts, hidden, predictions = zip(*batch)
    processor_inputs = {
        "pixel_values": torch.cat(pixel_values, dim=0).float(),
        "image_grid_thw": torch.stack(grids),
    }
    return (processor_inputs, torch.tensor(targets, dtype=torch.float32), torch.tensor(indices),
            torch.stack(grids), torch.stack(counts).long(), list(hidden), torch.stack(predictions).float())


class CachedEvaluationDataset(Dataset[Any]):
    def __init__(self, images: Dataset[Any], input_store: ProcessorCacheStore) -> None:
        self.images = images
        self.input_store = input_store

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        cached = self.input_store.get(index)
        return cached["pixel_values"], _cached_target(self.images, index), index, cached["image_grid_thw"]


def cached_evaluation_collate(batch: Sequence[Any]):
    pixel_values, targets, indices, grids = zip(*batch)
    return ({"pixel_values": torch.cat(pixel_values, dim=0).float(), "image_grid_thw": torch.stack(grids)},
            torch.tensor(targets, dtype=torch.float32), torch.tensor(indices))


def make_cached_evaluation_loader(dataset: Dataset[Any], input_store: ProcessorCacheStore,
                                  batch_size: int) -> DataLoader[Any]:
    # Sequential sample order is already shard-local for the test split.
    return DataLoader(CachedEvaluationDataset(dataset, input_store), batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=cached_evaluation_collate,
                      pin_memory=torch.cuda.is_available())


def _cached_target(dataset: Dataset[Any], index: int) -> float:
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return float(targets[index])
    # Compatibility fallback for a generic Dataset. SPAQSingleAttributeDataset takes the
    # fast path above and therefore does not open a JPEG here.
    _image, target = dataset[index]
    return float(target)


def _is_better(metric: str, current: float, best: float) -> bool:
    return current < best if metric == "mae" else current > best


def train_student(model: nn.Module, processor: Any, replacement: Any, head: NormalizedLinearRegressionHead,
                  train_dataset: Dataset[Any], test_dataset: Dataset[Any], train_store: TeacherCacheStore,
                  test_store: TeacherCacheStore, train_input_store: ProcessorCacheStore,
                  test_input_store: ProcessorCacheStore, settings: Any, device: torch.device) -> None:
    teacher_predictions = load_teacher_predictions(
        settings.output_dir / "teacher_cache" / "train_teacher_predictions.pt",
        settings.head_output_activation,
    )
    cached = CachedStudentDataset(train_dataset, train_store, train_input_store, teacher_predictions)
    sampler = EpochRotatingSampler(len(train_dataset), settings.train_samples_per_epoch, settings.seed,
                                   settings.teacher_cache_shard_size)
    loader = DataLoader(cached, batch_size=settings.student_batch_size, sampler=sampler, num_workers=0,
                        collate_fn=cached_collate, pin_memory=torch.cuda.is_available())
    test_loader = make_cached_evaluation_loader(test_dataset, test_input_store, settings.inference_batch_size)
    model.requires_grad_(False).eval(); replacement.use_student()
    replacement.surrogate.requires_grad_(True).train(); head.requires_grad_(True).train()
    router_parameters = list(replacement.surrogate.prompt.router.parameters())
    router_ids = {id(parameter) for parameter in router_parameters}
    main_parameters = [parameter for parameter in replacement.surrogate.parameters()
                       if id(parameter) not in router_ids]
    head_parameters = list(head.parameters())
    optimizer_class = torch.optim.AdamW if settings.optimizer_type == "adamw" else torch.optim.Adam
    optimizer = optimizer_class([
        {"params": main_parameters, "lr": settings.learning_rate, "group_name": "main"},
        {"params": head_parameters, "lr": settings.student_head_learning_rate, "group_name": "student_head"},
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
        train_store.reset_stats(); train_input_store.reset_stats(); test_input_store.reset_stats()
        dropout_active = settings.phase_dropout_enabled and epoch >= settings.phase_dropout_start_epoch
        replacement.surrogate.set_phase_dropout_active(dropout_active)
        replacement.surrogate.train(); head.train()
        totals = {name: 0.0 for name in ("total", "hidden", "prediction_distill", "regression", "balance", "importance")}
        seen = 0; train_predictions: list[torch.Tensor] = []; train_targets: list[torch.Tensor] = []
        selection = torch.zeros(settings.num_experts); weight_sum = torch.zeros(settings.num_experts)
        print(f"[sampling] epoch={epoch} samples={len(sampler)}/{len(train_dataset)} phase_dropout={'on' if dropout_active else 'off'}", flush=True)
        data_wait_sec = 0.0; train_compute_sec = 0.0; previous_batch_finished = time.perf_counter()
        for batch_index, (cpu_inputs, targets, _indices, cached_grids, cached_counts, teacher_hidden,
                          teacher_batch_predictions) in enumerate(loader, start=1):
            batch_ready = time.perf_counter(); data_wait_sec += batch_ready - previous_batch_finished
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
            loss_total.backward()
            if epoch == 1 and batch_index == 1:
                initial_metrics = _score_metrics(predictions.detach(), targets.detach())
                head_gradient_norm = sum(float(parameter.grad.detach().float().square().sum())
                                         for parameter in head.parameters() if parameter.grad is not None) ** 0.5
                write_json(settings.output_dir / "metrics" / "student_first_batch_diagnostics.json", {
                    "student_output_activation": getattr(head, "output_activation", "unknown"),
                    "student_head_fresh_initialization": True,
                    "prediction_min_normalized": initial_metrics["prediction_min_normalized"],
                    "prediction_max_normalized": initial_metrics["prediction_max_normalized"],
                    "prediction_std_normalized": initial_metrics["prediction_std_normalized"],
                    "prediction_boundary_ratio": initial_metrics["prediction_boundary_ratio"],
                    "prediction_out_of_range_ratio": initial_metrics["prediction_out_of_range_ratio"],
                    "student_head_gradient_l2_norm": head_gradient_norm,
                    "zero_head_gradient_on_first_batch": head_gradient_norm == 0.0,
                    "prediction_requires_grad": bool(predictions.requires_grad),
                    "loss_hidden": float(loss_hidden.detach()),
                    "loss_prediction_distill": float(loss_prediction_distill.detach()),
                    "loss_regression": float(loss_regression.detach()),
                    "finite_loss": bool(torch.isfinite(loss_total)),
                })
                if not torch.isfinite(loss_total) or not predictions.requires_grad:
                    raise RuntimeError("Student first batch has a non-finite loss or disconnected prediction graph")
                if head_gradient_norm == 0.0:
                    print(
                        "WARNING: student head gradient is exactly zero on the first batch. This can be a transient "
                        "cancellation between teacher-prediction and ground-truth regression terms; training will "
                        "continue and epoch-level prediction variance must be checked.",
                        flush=True,
                    )
            optimizer.step()
            train_compute_sec += time.perf_counter() - batch_ready
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
                learning_rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
                print(f"epoch {epoch}/{settings.epochs} batch {batch_index}/{len(loader)} loss={totals['total']/seen:.5f} "
                      f"hidden={totals['hidden']/seen:.5f} distill={totals['prediction_distill']/seen:.5f} "
                      f"attribute={totals['regression']/seen:.5f} "
                      f"balance={totals['balance']/seen:.5f} balance_term={settings.router_balance_weight * totals['balance']/seen:.5f} "
                      f"importance={totals['importance']/seen:.5f} importance_term={settings.router_importance_weight * totals['importance']/seen:.5f} "
                      f"MAE={current['mae']:.3f} SRCC={current['srcc']:.4f} "
                      f"pred=[{current['prediction_min_normalized']:.3f},{current['prediction_max_normalized']:.3f}] "
                      f"pred_std={current['prediction_std_normalized']:.4f} out_of_range={current['prediction_out_of_range_ratio']:.3f} "
                      f"lr={learning_rates['main']:.3e} head_lr={learning_rates['student_head']:.3e} "
                      f"router_lr={learning_rates['router']:.3e} "
                      f"experts=[{expert_status}]", flush=True)
            previous_batch_finished = time.perf_counter()
        train_report = _score_metrics(torch.cat(train_predictions), torch.cat(train_targets))
        test_started = time.perf_counter()
        test_report = evaluate_student(model, processor, replacement, head, test_loader, device, test_dataset,
                                       inputs_are_cached=True)
        test_time_sec = time.perf_counter() - test_started
        if scheduler is not None: scheduler.step()
        learning_rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        row: dict[str, Any] = {
            "epoch": epoch, "learning_rate": learning_rates["main"],
            "student_head_learning_rate": learning_rates["student_head"],
            "router_learning_rate": learning_rates["router"],
            **{f"loss_{name}": value / seen for name, value in totals.items()},
            **{f"train_{name}": value for name, value in train_report.items()},
            **{f"test_{name}": value for name, value in test_report.items() if isinstance(value, (int, float))},
            "epoch_time_sec": time.perf_counter() - started, "phase_dropout_active": dropout_active,
            "train_data_wait_sec": data_wait_sec, "train_compute_sec": train_compute_sec,
            "test_time_sec": test_time_sec,
            "teacher_cache_hits": train_store.stats()["hits"],
            "teacher_cache_misses": train_store.stats()["misses"],
            "teacher_cache_hit_rate": train_store.stats()["hit_rate"],
            "processor_cache_hits": train_input_store.stats()["hits"],
            "processor_cache_misses": train_input_store.stats()["misses"],
            "processor_cache_hit_rate": train_input_store.stats()["hit_rate"],
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
              f"test_pred=[{test_report['prediction_min_normalized']:.3f},{test_report['prediction_max_normalized']:.3f}] "
              f"test_pred_std={test_report['prediction_std_normalized']:.4f} "
              f"best_{settings.student_selection_metric}={best_value:.4f} debug_examples={debug_saved} "
              f"data_wait={data_wait_sec:.1f}s compute={train_compute_sec:.1f}s test={test_time_sec:.1f}s "
              f"teacher_cache_hit={train_store.stats()['hit_rate']:.3f} "
              f"processor_cache_hit={train_input_store.stats()['hit_rate']:.3f}", flush=True)
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
                               replacement.surrogate.last_detector_intensity[0].detach().cpu(), student_hidden,
                               teacher_hidden, epoch, settings.task_name)
        return count
    finally:
        replacement.surrogate.set_debug_capture(False)


@torch.inference_mode()
def evaluate_student(model: nn.Module, processor: Any, replacement: Any, head: nn.Module, loader: Any,
                     device: torch.device, dataset: Dataset[Any] | None = None,
                     predictions_path: Path | None = None, inputs_are_cached: bool = False) -> dict[str, Any]:
    replacement.use_student(); replacement.surrogate.eval(); replacement.surrogate.set_phase_dropout_active(False); head.eval()
    predictions_all: list[torch.Tensor] = []; targets_all: list[torch.Tensor] = []; indices_all: list[torch.Tensor] = []
    selection = torch.zeros(replacement.surrogate.geometry.num_experts); weight_sum = torch.zeros_like(selection); seen = 0
    for batch_inputs, targets, indices in loader:
        cpu_inputs = batch_inputs if inputs_are_cached else preprocess_images(processor, batch_inputs)
        run_visual(model, move_inputs(cpu_inputs, device))
        groups = list(replacement.surrogate.last_output.split(replacement.surrogate.last_token_counts, dim=0))
        predictions_all.append(head(pool_token_groups(groups)).cpu()); targets_all.append(targets.cpu()); indices_all.append(indices.cpu())
        routing = replacement.surrogate.last_routing; seen += len(targets)
        selection += routing["selected_mask"].cpu().float().sum(0); weight_sum += routing["weights"].cpu().sum(0)
    predictions = torch.cat(predictions_all); targets = torch.cat(targets_all); indices = torch.cat(indices_all)
    report: dict[str, Any] = _score_metrics(predictions, targets)
    report.update({"dataset": "SPAQ", "task": getattr(dataset, "task_name", "unknown"), "score_scale": [0.0, 100.0],
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
              "predictions_clamped_for_metrics": False,
              "phase_dropout_disabled_for_inference": True,
              "parameter_breakdown": replacement.surrogate.parameter_breakdown()}
    write_json(settings.output_dir / "metrics" / "student_inference.json", report)
    if settings.visualization_enabled and settings.save_scatter_plot and predictions_path and predictions_path.is_file():
        import csv
        with predictions_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        save_scatter([float(row["true_score"]) for row in rows], [float(row["predicted_score"]) for row in rows],
                     settings.output_dir / "figures" / f"student_{settings.task_name.lower()}_scatter.png",
                     f"SPAQ {settings.task_name} optical student", settings.task_name)


def save_head(head: NormalizedLinearRegressionHead, path: Path, settings: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "head": head.specification(), "task": settings.task_name,
                "target_scale": [0.0, 1.0], "smooth_l1_beta": settings.smooth_l1_beta}, path)


def load_head(path: Path, settings: Any, device: torch.device) -> NormalizedLinearRegressionHead:
    if not path.is_file():
        raise FileNotFoundError(f"Teacher head checkpoint missing: {path}. Run --phase teacher_train first.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    saved_task = payload.get("task")
    if saved_task is not None and saved_task != settings.task_name:
        raise RuntimeError(
            f"Teacher head task differs from current config: saved={saved_task}, current={settings.task_name}. "
            "Use the task-specific output_dir or rerun teacher_train."
        )
    head = build_head(settings, settings.vision_hidden_size).to(device)
    saved = payload.get("head", {})
    if saved and saved.get("output_activation") != settings.head_output_activation:
        raise RuntimeError("Teacher head output activation differs from current config; use a new output_dir or retrain it")
    head.load_state_dict(payload["state_dict"]); return head


def save_student_parts(output_dir: Path, replacement: Any, head: nn.Module, tag: str,
                       epoch: int, metrics: dict[str, Any]) -> None:
    checkpoint = output_dir / "checkpoints"; checkpoint.mkdir(parents=True, exist_ok=True)
    task_name = getattr(head, "task_name", None)
    metadata = {"epoch": epoch, "metrics": metrics, "task": task_name,
                "head": head.specification() if hasattr(head, "specification") else {}}
    torch.save({"state_dict": replacement.surrogate.state_dict(), **metadata}, checkpoint / f"vision_moe_{tag}.pt")
    torch.save({"state_dict": head.state_dict(), **metadata}, checkpoint / f"student_head_{tag}.pt")


def load_student_parts(output_dir: Path, replacement: Any, head: nn.Module, tag: str) -> None:
    checkpoint = output_dir / "checkpoints"
    surrogate_path = checkpoint / f"vision_moe_{tag}.pt"; head_path = checkpoint / f"student_head_{tag}.pt"
    if not surrogate_path.is_file() or not head_path.is_file():
        raise FileNotFoundError(f"Student checkpoint '{tag}' is incomplete under {checkpoint}")
    surrogate_payload = torch.load(surrogate_path, map_location="cpu", weights_only=True)
    head_payload = torch.load(head_path, map_location="cpu", weights_only=True)
    saved_task = head_payload.get("task")
    current_task = getattr(head, "task_name", None)
    if saved_task is not None and current_task is not None and saved_task != current_task:
        raise RuntimeError(
            f"Student checkpoint '{tag}' is for task={saved_task}, but current config uses {current_task}."
        )
    saved_head = head_payload.get("head", {})
    current_head = head.specification() if hasattr(head, "specification") else {}
    if saved_head and saved_head.get("output_activation") != current_head.get("output_activation"):
        raise RuntimeError(
            f"Student checkpoint '{tag}' uses output_activation={saved_head.get('output_activation')}, "
            f"but current config requests {current_head.get('output_activation')}. Retrain student_train; "
            "the electronic teacher cache and teacher_head.pt can still be reused."
        )
    replacement.surrogate.load_state_dict(surrogate_payload["state_dict"])
    head.load_state_dict(head_payload["state_dict"])


@torch.inference_mode()
def _head_predictions(head: nn.Module, features: torch.Tensor, batch_size: int,
                      device: torch.device) -> torch.Tensor:
    head.eval(); values = []
    for start in range(0, len(features), batch_size):
        values.append(head(features[start:start + batch_size].to(device)).cpu())
    return torch.cat(values)
