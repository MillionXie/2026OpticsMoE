from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .datasets import sample_metadata, targets_of
from .features import move_inputs, multimodal_forward_features, pool_answer_hidden_state
from .io_utils import write_csv, write_json
from .metrics import regression_metrics
from .modeling import NormalizedLinearRegressionHead, build_head
from .processor_cache import ProcessorCacheStore, collate_processor_samples
from .sampling import EpochRotatingSampler
from .teacher_cache import (TeacherCacheStore, cached_answer_features, load_teacher_predictions,
                            write_teacher_predictions)
from .visualization import save_phase_masks, save_scatter, save_training_curves


def score_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predictions = predictions.float().reshape(-1); targets = targets.float().reshape(-1)
    report = regression_metrics((targets * 100).tolist(), (predictions * 100).tolist())
    report.update({"prediction_min_normalized": float(predictions.min()),
                   "prediction_max_normalized": float(predictions.max()),
                   "prediction_mean_normalized": float(predictions.mean()),
                   "prediction_std_normalized": float(predictions.std(unbiased=False)),
                   "prediction_out_of_range_ratio": float(((predictions < 0) | (predictions > 1)).float().mean())})
    return report


def _split_indices(samples: int, fraction: float, seed: int):
    order = torch.randperm(samples, generator=torch.Generator().manual_seed(seed))
    count = min(max(round(samples * fraction), 1), samples - 1)
    return order[count:], order[:count]


def _head_predictions(head: nn.Module, features: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    head.eval(); chunks = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size): chunks.append(head(features[start:start + batch_size].to(device)).cpu())
    return torch.cat(chunks)


def train_teacher_head(train_store: TeacherCacheStore, test_store: TeacherCacheStore,
                       train_dataset: Dataset[Any], test_dataset: Dataset[Any], settings: Any,
                       device: torch.device) -> NormalizedLinearRegressionHead:
    features, targets = cached_answer_features(train_store, targets_of(train_dataset))
    train_indices, validation_indices = _split_indices(len(targets), settings.validation_fraction, settings.seed)
    head = build_head(settings, features.shape[1]).to(device)
    optimizer_cls = torch.optim.AdamW if settings.optimizer_type == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(head.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs) if settings.scheduler_type == "cosine" else None
    history = []; best = float("-inf")
    for epoch in range(1, settings.epochs + 1):
        head.train(); total = 0.0
        loader = DataLoader(TensorDataset(features[train_indices], targets[train_indices]),
                            batch_size=settings.head_batch_size, shuffle=True)
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device); batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True); loss = F.smooth_l1_loss(head(batch_features), batch_targets,
                                                                          beta=settings.smooth_l1_beta)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(batch_targets)
        predictions = _head_predictions(head, features[validation_indices], settings.head_batch_size, device)
        metrics = score_metrics(predictions, targets[validation_indices]); row = {"epoch": epoch,
            "train_loss": total / len(train_indices), **{f"validation_{k}": v for k, v in metrics.items()}}
        history.append(row); write_csv(settings.output_dir / "metrics" / "teacher_training_history.csv", history, list(row))
        write_json(settings.output_dir / "metrics" / "teacher_training_latest.json", row)
        if metrics["srcc"] > best:
            best = metrics["srcc"]; save_head(head, settings.output_dir / "checkpoints" / "teacher_head.pt", settings)
            write_json(settings.output_dir / "metrics" / "teacher_best_validation.json", row)
        if scheduler: scheduler.step()
    head = load_head(settings.output_dir / "checkpoints" / "teacher_head.pt", settings, device)
    teacher_inference(head, test_store, test_dataset, settings, device); return head


def generate_teacher_predictions(head: nn.Module, stores: dict[str, TeacherCacheStore], settings: Any,
                                 device: torch.device) -> None:
    for split, store in stores.items():
        features = torch.cat([shard["teacher_answer_hidden"].float() for shard in store.iter_shards()])
        predictions = _head_predictions(head, features, settings.head_batch_size, device)
        write_teacher_predictions(settings.output_dir, split, predictions, settings.head_output_activation)


def teacher_inference(head: nn.Module, store: TeacherCacheStore, dataset: Dataset[Any], settings: Any,
                      device: torch.device) -> dict[str, Any]:
    features, targets = cached_answer_features(store, targets_of(dataset)); predictions = _head_predictions(
        head, features, settings.head_batch_size, device)
    report = {**score_metrics(predictions, targets), "dataset": "SPAQ", "task": settings.task_name,
              "model": "full_electronic_qwen3_vl_2b_multimodal_regressor", "prompt": settings.classification_prompt}
    write_json(settings.output_dir / "metrics" / "teacher_inference.json", report); return report


class CachedStudentDataset(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any], teacher: TeacherCacheStore, inputs: ProcessorCacheStore,
                 predictions: torch.Tensor) -> None:
        self.dataset = dataset; self.teacher = teacher; self.inputs = inputs; self.predictions = predictions
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index: int):
        return self.inputs.get(index), float(targets_of(self.dataset)[index]), index, self.teacher.get(index), self.predictions[index]


def _cached_target(dataset: Dataset[Any], index: int) -> float:
    targets = getattr(dataset, "targets", None)
    if targets is not None: return float(targets[index])
    return float(dataset[index][1])


def cached_student_collate(batch: Sequence[Any], metadata: dict[str, Any]):
    inputs, targets, indices, teachers, predictions = zip(*batch)
    return collate_processor_samples(inputs, metadata), torch.tensor(targets), torch.tensor(indices), list(teachers), torch.stack(predictions)


class CachedEvaluationDataset(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any], inputs: ProcessorCacheStore): self.dataset = dataset; self.inputs = inputs
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index): return self.inputs.get(index), _cached_target(self.dataset, index), index


def make_evaluation_loader(dataset: Dataset[Any], inputs: ProcessorCacheStore, batch_size: int):
    return DataLoader(CachedEvaluationDataset(dataset, inputs), batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=lambda batch: (collate_processor_samples([row[0] for row in batch], inputs.metadata),
                                                torch.tensor([row[1] for row in batch]),
                                                torch.tensor([row[2] for row in batch])))


def train_student(model: nn.Module, replacement: Any, head: nn.Module, train_dataset: Dataset[Any],
                  test_dataset: Dataset[Any], train_store: TeacherCacheStore, test_store: TeacherCacheStore,
                  train_inputs: ProcessorCacheStore, test_inputs: ProcessorCacheStore,
                  settings: Any, device: torch.device) -> None:
    teacher_predictions = load_teacher_predictions(settings.output_dir / "teacher_cache" / "train_teacher_predictions.pt",
                                                     settings.head_output_activation)
    cached = CachedStudentDataset(train_dataset, train_store, train_inputs, teacher_predictions)
    sampler = EpochRotatingSampler(len(train_dataset), settings.train_samples_per_epoch, settings.seed,
                                   settings.teacher_cache_shard_size)
    loader = DataLoader(cached, batch_size=settings.student_batch_size, sampler=sampler, num_workers=0,
                        collate_fn=lambda batch: cached_student_collate(batch, train_inputs.metadata), pin_memory=True)
    test_loader = make_evaluation_loader(test_dataset, test_inputs, settings.inference_batch_size)
    replacement.use_student(); model.requires_grad_(False).eval()
    replacement.vision_surrogate.requires_grad_(True)
    if settings.student_language_mode == "optical_moe": replacement.language_surrogate.requires_grad_(True)
    head.requires_grad_(True)
    routers = list(replacement.vision_surrogate.core.prompt.router.parameters())
    if settings.student_language_mode == "optical_moe": routers += list(replacement.language_surrogate.core.prompt.router.parameters())
    router_ids = {id(parameter) for parameter in routers}
    main = [parameter for parameter in replacement.trainable_parameters() if id(parameter) not in router_ids]
    optimizer_cls = torch.optim.AdamW if settings.optimizer_type == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls([{"params": main, "lr": settings.learning_rate, "group_name": "optical"},
                               {"params": head.parameters(), "lr": settings.student_head_learning_rate, "group_name": "head"},
                               {"params": routers, "lr": settings.router_learning_rate, "group_name": "routers"}],
                              weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs) if settings.scheduler_type == "cosine" else None
    history = []; best = float("-inf") if settings.student_selection_metric != "mae" else float("inf")
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter(); sampler.set_epoch(epoch); dropout = settings.phase_dropout_enabled and epoch >= settings.phase_dropout_start_epoch
        replacement.set_phase_dropout_active(dropout); replacement.vision_surrogate.train()
        if settings.student_language_mode == "optical_moe": replacement.language_surrogate.train()
        head.train(); totals = {name: 0.0 for name in ("total", "vision", "answer", "distill", "regression",
                                                       "vision_balance", "language_balance")}
        predictions_all = []; targets_all = []; seen = 0
        v_selection = torch.zeros(settings.num_experts); l_selection = torch.zeros(settings.num_experts)
        print(f"[sampling] epoch={epoch} samples={len(sampler)}/{len(train_dataset)} mode={settings.student_language_mode}", flush=True)
        for batch_index, (cpu_inputs, targets, _indices, teachers, teacher_predictions_batch) in enumerate(loader, 1):
            inputs = move_inputs(cpu_inputs, device); targets = targets.to(device); teacher_predictions_batch = teacher_predictions_batch.to(device)
            replacement.prepare_student_batch(inputs["attention_mask"]); optimizer.zero_grad(set_to_none=True)
            hidden = multimodal_forward_features(model, inputs); answer, _ = pool_answer_hidden_state(hidden, inputs["attention_mask"])
            predictions = head(answer)
            student_taps = [*replacement.vision_surrogate.tap_outputs, replacement.vision_surrogate.last_output]
            vision_losses = []
            for tap_number, student_packed in enumerate(student_taps):
                groups = student_packed.split(replacement.vision_surrogate.last_token_counts)
                for group, teacher in zip(groups, teachers):
                    target = teacher["teacher_vision_taps"][tap_number].float().to(device)
                    vision_losses.append(F.mse_loss(F.layer_norm(group.float(), (group.shape[-1],)),
                                                    F.layer_norm(target, (target.shape[-1],))))
            loss_vision = torch.stack(vision_losses).mean()
            teacher_answer = torch.stack([row["teacher_answer_hidden"] for row in teachers]).float().to(device)
            loss_answer = F.mse_loss(F.layer_norm(answer, (answer.shape[-1],)),
                                     F.layer_norm(teacher_answer, (teacher_answer.shape[-1],)))
            loss_distill = F.smooth_l1_loss(predictions, teacher_predictions_batch, beta=settings.smooth_l1_beta)
            loss_regression = F.smooth_l1_loss(predictions, targets, beta=settings.smooth_l1_beta)
            router = replacement.router_losses()
            loss_total = (settings.loss_hidden_weight * loss_vision + settings.loss_answer_weight * loss_answer +
                          settings.loss_prediction_distill_weight * loss_distill +
                          settings.loss_regression_weight * loss_regression +
                          settings.router_balance_weight * (router["vision_balance"] + router["language_balance"]) +
                          settings.router_importance_weight * (router["vision_importance"] + router["language_importance"]))
            loss_total.backward(); optimizer.step(); batch_size = len(targets); seen += batch_size
            for name, value in (("total", loss_total), ("vision", loss_vision), ("answer", loss_answer),
                                ("distill", loss_distill), ("regression", loss_regression),
                                ("vision_balance", router["vision_balance"]), ("language_balance", router["language_balance"])):
                totals[name] += float(value.detach()) * batch_size
            predictions_all.append(predictions.detach().cpu()); targets_all.append(targets.detach().cpu())
            v_selection += replacement.vision_surrogate.core.last_routing["selected_mask"].detach().cpu().float().sum(0)
            if settings.student_language_mode == "optical_moe":
                l_selection += replacement.language_surrogate.core.last_routing["selected_mask"].detach().cpu().float().sum(0)
            if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
                current = score_metrics(torch.cat(predictions_all), torch.cat(targets_all))
                print(f"epoch {epoch}/{settings.epochs} batch {batch_index}/{len(loader)} total={totals['total']/seen:.5f} "
                      f"vision={totals['vision']/seen:.5f} answer={totals['answer']/seen:.5f} "
                      f"distill={totals['distill']/seen:.5f} gt={totals['regression']/seen:.5f} "
                      f"v_balance={totals['vision_balance']/seen:.5f} l_balance={totals['language_balance']/seen:.5f} "
                      f"MAE={current['mae']:.3f} SRCC={current['srcc']:.4f} "
                      f"vision_sel={[round(float(x/seen),3) for x in v_selection]} "
                      f"language_sel={[round(float(x/seen),3) for x in l_selection]}", flush=True)
        train_report = score_metrics(torch.cat(predictions_all), torch.cat(targets_all))
        test_report = evaluate_student(model, replacement, head, test_loader, settings, device, test_dataset)
        if scheduler: scheduler.step()
        row = {"epoch": epoch, **{f"loss_{key}": value / seen for key, value in totals.items()},
               **{f"train_{key}": value for key, value in train_report.items()},
               **{f"test_{key}": value for key, value in test_report.items() if isinstance(value, (int, float))},
               "epoch_time_sec": time.perf_counter() - started, "samples_this_epoch": len(sampler),
               "student_language_mode": settings.student_language_mode, "phase_dropout_active": dropout}
        for expert in range(settings.num_experts):
            row[f"vision_expert_{expert}_selection_rate"] = float(v_selection[expert] / seen)
            row[f"language_expert_{expert}_selection_rate"] = float(l_selection[expert] / seen)
        history.append(row); write_csv(settings.output_dir / "metrics" / "student_training_history.csv", history, list(row))
        write_json(settings.output_dir / "metrics" / "student_training_latest.json", row)
        save_student_parts(settings.output_dir, replacement, head, "last", epoch, row)
        if epoch % settings.checkpoint_interval_epochs == 0: save_student_parts(settings.output_dir, replacement, head, f"epoch_{epoch:04d}", epoch, row)
        value = float(test_report[settings.student_selection_metric]); improved = value < best if settings.student_selection_metric == "mae" else value > best
        if improved:
            best = value; save_student_parts(settings.output_dir, replacement, head, "best", epoch, row)
            write_json(settings.output_dir / "metrics" / "best_test.json", row)
        if settings.visualization_enabled and settings.save_training_curves: save_training_curves(history, settings.output_dir / "figures" / "student_training_curves.png")
        if settings.visualization_enabled and settings.save_phase_masks and epoch % settings.visualization_interval_epochs == 0:
            save_phase_masks(replacement.vision_surrogate.core, settings.output_dir / "figures" / f"vision_phase_masks_epoch_{epoch:04d}.png", f"Vision epoch {epoch}")
            if settings.student_language_mode == "optical_moe":
                save_phase_masks(replacement.language_surrogate.core, settings.output_dir / "figures" / f"language_phase_masks_epoch_{epoch:04d}.png", f"Language epoch {epoch}")
        print(f"epoch {epoch:03d} complete train_MAE={train_report['mae']:.3f} train_SRCC={train_report['srcc']:.4f} "
              f"test_MAE={test_report['mae']:.3f} test_SRCC={test_report['srcc']:.4f} best={best:.4f}", flush=True)
    write_json(settings.output_dir / "metrics" / "student_training.json", {"epochs": settings.epochs,
               "best_metric": settings.student_selection_metric, "best_value": best,
               "student_language_mode": settings.student_language_mode})


@torch.inference_mode()
def evaluate_student(model: nn.Module, replacement: Any, head: nn.Module, loader: Any, settings: Any,
                     device: torch.device, dataset: Dataset[Any] | None = None, predictions_path: Path | None = None):
    replacement.use_student(); replacement.set_phase_dropout_active(False); model.eval(); head.eval()
    predictions_all = []; targets_all = []; indices_all = []
    for cpu_inputs, targets, indices in loader:
        inputs = move_inputs(cpu_inputs, device); replacement.prepare_student_batch(inputs["attention_mask"])
        hidden = multimodal_forward_features(model, inputs); answer, _ = pool_answer_hidden_state(hidden, inputs["attention_mask"])
        predictions_all.append(head(answer).cpu()); targets_all.append(targets); indices_all.append(indices)
    predictions = torch.cat(predictions_all); targets = torch.cat(targets_all); indices = torch.cat(indices_all)
    report = {**score_metrics(predictions, targets), "dataset": "SPAQ", "task": settings.task_name,
              "model": f"vision_optical_moe_language_{settings.student_language_mode}",
              "language_model_used": True, "prompt": settings.classification_prompt}
    if predictions_path:
        rows = []
        for index, target, prediction in zip(indices.tolist(), targets.tolist(), predictions.tolist()):
            metadata = sample_metadata(dataset, index) if dataset is not None else {"sample_index": index}
            rows.append({**metadata, "sample_index": index, "true_score": target * 100,
                         "predicted_score": prediction * 100, "absolute_error": abs(prediction-target)*100})
        if rows: write_csv(predictions_path, rows, list(rows[0]))
    return report


def save_head(head: NormalizedLinearRegressionHead, path: Path, settings: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "head": head.specification(), "task": settings.task_name}, path)


def load_head(path: Path, settings: Any, device: torch.device):
    if not path.is_file(): raise FileNotFoundError(f"Teacher head missing: {path}. Run teacher_train.")
    payload = torch.load(path, map_location="cpu", weights_only=True); head = build_head(settings, settings.text_hidden_size).to(device)
    if payload.get("task") not in (None, settings.task_name): raise RuntimeError("Teacher head task mismatch")
    head.load_state_dict(payload["state_dict"]); return head


def save_student_parts(output_dir: Path, replacement: Any, head: nn.Module, tag: str, epoch: int, metrics: dict[str, Any]):
    root = output_dir / "checkpoints"; root.mkdir(parents=True, exist_ok=True); metadata = {"epoch": epoch, "metrics": metrics}
    torch.save({"state_dict": replacement.vision_surrogate.state_dict(), **metadata}, root / f"vision_moe_{tag}.pt")
    if replacement.language_mode == "optical_moe":
        torch.save({"state_dict": replacement.language_surrogate.state_dict(), **metadata}, root / f"language_moe_{tag}.pt")
    torch.save({"state_dict": head.state_dict(), "head": head.specification(), **metadata}, root / f"student_head_{tag}.pt")


def load_student_parts(output_dir: Path, replacement: Any, head: nn.Module, tag: str):
    root = output_dir / "checkpoints"; vision = root / f"vision_moe_{tag}.pt"; head_path = root / f"student_head_{tag}.pt"
    if not vision.is_file() or not head_path.is_file(): raise FileNotFoundError(f"Incomplete student checkpoint {tag}")
    replacement.vision_surrogate.load_state_dict(torch.load(vision, map_location="cpu", weights_only=True)["state_dict"])
    if replacement.language_mode == "optical_moe":
        language = root / f"language_moe_{tag}.pt"
        if not language.is_file(): raise FileNotFoundError(f"Missing language MoE checkpoint: {language}")
        replacement.language_surrogate.load_state_dict(torch.load(language, map_location="cpu", weights_only=True)["state_dict"])
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True)["state_dict"])


def save_student_inference(report: dict[str, Any], settings: Any, replacement: Any, predictions_path: Path | None):
    report["vision_parameter_breakdown"] = replacement.vision_surrogate.parameter_breakdown()
    report["language_parameter_breakdown"] = (replacement.language_surrogate.parameter_breakdown()
                                                if replacement.language_mode == "optical_moe" else None)
    write_json(settings.output_dir / "metrics" / "student_inference.json", report)
    if settings.visualization_enabled and settings.save_scatter_plot and predictions_path and predictions_path.is_file():
        import csv
        with predictions_path.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
        save_scatter([float(row["true_score"]) for row in rows], [float(row["predicted_score"]) for row in rows],
                     settings.output_dir / "figures" / "student_scatter.png", "SPAQ optical multimodal student", settings.task_name)
