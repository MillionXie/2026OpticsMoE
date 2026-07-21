from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .datasets import sample_metadata, targets_of
from .features import move_inputs, multimodal_forward_features, pool_answer_hidden_state
from .io_utils import write_csv, write_json
from .metrics import binary_classification_metrics, probabilities_from_logits
from .modeling import NormalizedBinaryClassificationHead, build_head
from .processor_cache import ProcessorCacheStore, collate_processor_samples
from .sampling import EpochRotatingSampler
from .teacher_cache import (TeacherCacheStore, cached_answer_features, load_teacher_logits,
                            write_teacher_logits)
from .visualization import save_confusion_matrix, save_phase_masks, save_training_curves


def score_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, Any]:
    return binary_classification_metrics(labels.detach().float().cpu().numpy(),
                                         logits.detach().float().cpu().numpy(), threshold)


def _head_logits(head: nn.Module, features: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    head.eval(); chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            chunks.append(head(features[start:start + batch_size].to(device)).float().cpu())
    return torch.cat(chunks)


def train_teacher_head(train_store: TeacherCacheStore, test_store: TeacherCacheStore,
                       train_dataset: Dataset[Any], test_dataset: Dataset[Any], settings: Any,
                       device: torch.device) -> NormalizedBinaryClassificationHead:
    train_features, train_labels = cached_answer_features(train_store)
    test_features, test_labels = cached_answer_features(test_store)
    if train_labels.tolist() != targets_of(train_dataset): raise RuntimeError("Teacher train cache labels do not match pair manifest")
    if test_labels.tolist() != targets_of(test_dataset): raise RuntimeError("Teacher test cache labels do not match pair manifest")
    head = build_head(settings, train_features.shape[1]).to(device)
    optimizer_cls = torch.optim.AdamW if settings.optimizer_type == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(head.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs) if settings.scheduler_type == "cosine" else None
    history: list[dict[str, Any]] = []; best = float("-inf")
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter(); head.train(); total = 0.0
        loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=settings.head_batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(settings.seed + epoch))
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device); batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features); loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(batch_labels)
        train_logits = _head_logits(head, train_features, settings.head_batch_size, device)
        test_logits = _head_logits(head, test_features, settings.head_batch_size, device)
        train_report = score_metrics(train_logits, train_labels, settings.classification_threshold)
        test_report = score_metrics(test_logits, test_labels, settings.classification_threshold)
        row = {"epoch": epoch, "train_optimization_bce": total / len(train_labels),
               **{f"train_{key}": value for key, value in train_report.items() if isinstance(value, (int, float))},
               **{f"test_{key}": value for key, value in test_report.items() if isinstance(value, (int, float))},
               "epoch_time_sec": time.perf_counter() - started,
               "selection_split": "test", "selection_metric": "auroc"}
        history.append(row); write_csv(settings.output_dir / "metrics" / "teacher_training_history.csv", history, list(row))
        write_json(settings.output_dir / "metrics" / "teacher_training_latest.json", row)
        if float(test_report["auroc"]) > best:
            best = float(test_report["auroc"]); save_head(head, settings.output_dir / "checkpoints" / "teacher_head.pt", settings)
            write_json(settings.output_dir / "metrics" / "teacher_best_test.json", row)
        if scheduler: scheduler.step()
        print(f"teacher epoch {epoch:03d}/{settings.epochs} train_acc={train_report['accuracy']:.4f} "
              f"test_acc={test_report['accuracy']:.4f} test_AUROC={test_report['auroc']:.4f} best={best:.4f}", flush=True)
    head = load_head(settings.output_dir / "checkpoints" / "teacher_head.pt", settings, device)
    teacher_inference(head, test_store, test_dataset, settings, device)
    return head


def generate_teacher_logits(head: nn.Module, stores: dict[str, TeacherCacheStore], settings: Any,
                            device: torch.device) -> None:
    for split, store in stores.items():
        features, _labels = cached_answer_features(store)
        logits = _head_logits(head, features, settings.head_batch_size, device)
        write_teacher_logits(settings.output_dir, split, logits, settings)


def teacher_inference(head: nn.Module, store: TeacherCacheStore, dataset: Dataset[Any], settings: Any,
                      device: torch.device) -> dict[str, Any]:
    features, labels = cached_answer_features(store); logits = _head_logits(head, features, settings.head_batch_size, device)
    report = {**score_metrics(logits, labels, settings.classification_threshold),
              "dataset": settings.dataset, "model": "full_electronic_qwen3_vl_2b_binary_head",
              "head": head.specification(), "prompt_template": settings.prompt_template,
              "pair_manifest_digest": (settings.pair_manifest_digests or {}).get("test"),
              "checkpoint_selection": "best test AUROC (user-requested; selection-biased)"}
    prediction_path = settings.output_dir / "metrics" / "teacher_predictions.csv"
    _write_predictions(prediction_path, dataset, torch.arange(len(labels)), labels, logits,
                       settings.classification_threshold)
    report["predictions_csv"] = str(prediction_path)
    write_json(settings.output_dir / "metrics" / "teacher_inference.json", report)
    if settings.visualization_enabled and settings.save_confusion_matrix:
        save_confusion_matrix(report["confusion_matrix"], settings.output_dir / "figures" / "teacher_confusion_matrix.png",
                              "Electronic Qwen3-VL teacher")
    return report


class CachedStudentDataset(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any], teacher: TeacherCacheStore, inputs: ProcessorCacheStore,
                 logits: torch.Tensor) -> None:
        self.dataset = dataset; self.teacher = teacher; self.inputs = inputs; self.logits = logits
        self.targets = targets_of(dataset)

    def __len__(self) -> int: return len(self.dataset)
    def __getitem__(self, index: int) -> tuple[Any, float, int, Any, torch.Tensor]:
        teacher = self.teacher.get(index)
        if float(teacher["label"]) != float(self.targets[index]): raise RuntimeError(f"Teacher cache label mismatch at pair {index}")
        return self.inputs.get(index), float(self.targets[index]), index, teacher, self.logits[index]


def cached_student_collate(batch: Sequence[Any], metadata: dict[str, Any]):
    inputs, labels, indices, teachers, logits = zip(*batch)
    return (collate_processor_samples(inputs, metadata), torch.tensor(labels, dtype=torch.float32),
            torch.tensor(indices, dtype=torch.long), list(teachers), torch.stack(logits).float())


class CachedEvaluationDataset(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any], inputs: ProcessorCacheStore) -> None:
        self.dataset = dataset; self.inputs = inputs; self.targets = targets_of(dataset)
    def __len__(self) -> int: return len(self.dataset)
    def __getitem__(self, index: int): return self.inputs.get(index), self.targets[index], index


def make_evaluation_loader(dataset: Dataset[Any], inputs: ProcessorCacheStore, batch_size: int) -> DataLoader[Any]:
    return DataLoader(CachedEvaluationDataset(dataset, inputs), batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=lambda batch: (collate_processor_samples([row[0] for row in batch], inputs.metadata),
                                                torch.tensor([row[1] for row in batch], dtype=torch.float32),
                                                torch.tensor([row[2] for row in batch], dtype=torch.long)))


def train_student(model: nn.Module, replacement: Any, head: nn.Module, train_dataset: Dataset[Any],
                  test_dataset: Dataset[Any], train_store: TeacherCacheStore, test_store: TeacherCacheStore,
                  train_inputs: ProcessorCacheStore, test_inputs: ProcessorCacheStore,
                  settings: Any, device: torch.device) -> None:
    del test_store  # test teacher hidden targets are intentionally not used during parameter updates.
    if settings.initialize_student_head_from_teacher:
        teacher_head = load_head(settings.output_dir / "checkpoints" / "teacher_head.pt", settings, device)
        head.load_state_dict(teacher_head.state_dict())
    teacher_logits = load_teacher_logits(settings.output_dir / "teacher_cache" / "train_teacher_logits.pt", settings, "train")
    cached = CachedStudentDataset(train_dataset, train_store, train_inputs, teacher_logits)
    sampler = EpochRotatingSampler(len(train_dataset), settings.train_samples_per_epoch, settings.seed,
                                   settings.teacher_cache_shard_size)
    loader = DataLoader(cached, batch_size=settings.student_batch_size, sampler=sampler, num_workers=0,
                        collate_fn=lambda batch: cached_student_collate(batch, train_inputs.metadata), pin_memory=True)
    test_loader = make_evaluation_loader(test_dataset, test_inputs, settings.inference_batch_size)
    replacement.use_student(); model.requires_grad_(False).eval(); replacement.vision_surrogate.requires_grad_(True)
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
    history: list[dict[str, Any]] = []; best = float("-inf")
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter(); sampler.set_epoch(epoch)
        dropout_active = settings.phase_dropout_enabled and epoch >= settings.phase_dropout_start_epoch
        replacement.set_phase_dropout_active(dropout_active); replacement.vision_surrogate.train()
        if settings.student_language_mode == "optical_moe": replacement.language_surrogate.train()
        head.train(); logits_all: list[torch.Tensor] = []; labels_all: list[torch.Tensor] = []; seen = 0
        names = ("total", "vision", "answer", "logit", "classification", "vision_balance", "language_balance",
                 "vision_importance", "language_importance")
        totals = torch.zeros(len(names), device=device, dtype=torch.float64)
        v_selection = torch.zeros(settings.num_experts, device=device); l_selection = torch.zeros_like(v_selection)
        v_weights = torch.zeros_like(v_selection); l_weights = torch.zeros_like(v_selection)
        print(f"[sampling] epoch={epoch} pairs={len(sampler)}/{len(train_dataset)} mode={settings.student_language_mode}", flush=True)
        for batch_index, (cpu_inputs, labels, _indices, teachers, teacher_batch_logits) in enumerate(loader, 1):
            replacement.prepare_student_batch(cpu_inputs["attention_mask"]); inputs = move_inputs(cpu_inputs, device)
            labels = labels.to(device, non_blocking=True); teacher_batch_logits = teacher_batch_logits.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            hidden = multimodal_forward_features(model, inputs); answer, _ = pool_answer_hidden_state(hidden, inputs["attention_mask"])
            logits = head(answer)
            student_taps = [*replacement.vision_surrogate.tap_outputs, replacement.vision_surrogate.last_output]
            vision_losses: list[torch.Tensor] = []
            for tap_number, student_packed in enumerate(student_taps):
                groups = student_packed.split(replacement.vision_surrogate.last_token_counts)
                for group, teacher in zip(groups, teachers):
                    target = teacher["teacher_vision_taps"][tap_number].float().to(device)
                    vision_losses.append(F.mse_loss(F.layer_norm(group.float(), (group.shape[-1],)),
                                                    F.layer_norm(target, (target.shape[-1],))))
            loss_vision = torch.stack(vision_losses).mean()
            teacher_answer = torch.stack([row["teacher_answer_hidden"] for row in teachers]).float().to(device)
            loss_answer = F.mse_loss(F.layer_norm(answer.float(), (answer.shape[-1],)),
                                     F.layer_norm(teacher_answer, (teacher_answer.shape[-1],)))
            loss_logit = F.smooth_l1_loss(logits, teacher_batch_logits, beta=settings.smooth_l1_beta)
            loss_classification = F.binary_cross_entropy_with_logits(logits, labels)
            router = replacement.router_losses()
            loss_total = (settings.loss_hidden_weight * loss_vision + settings.loss_answer_weight * loss_answer +
                          settings.loss_logit_weight * loss_logit + settings.loss_classification_weight * loss_classification +
                          settings.router_balance_weight * (router["vision_balance"] + router["language_balance"]) +
                          settings.router_importance_weight * (router["vision_importance"] + router["language_importance"]))
            loss_total.backward(); optimizer.step(); batch_size = len(labels); seen += batch_size
            values = (loss_total, loss_vision, loss_answer, loss_logit, loss_classification,
                      router["vision_balance"], router["language_balance"],
                      router["vision_importance"], router["language_importance"])
            totals += torch.stack([value.detach().double() for value in values]) * batch_size
            logits_all.append(logits.detach()); labels_all.append(labels.detach())
            vr = replacement.vision_surrogate.core.last_routing
            v_selection += vr["selected_mask"].detach().float().sum(0); v_weights += vr["weights"].detach().sum(0)
            if settings.student_language_mode == "optical_moe":
                lr = replacement.language_surrogate.core.last_routing
                l_selection += lr["selected_mask"].detach().float().sum(0); l_weights += lr["weights"].detach().sum(0)
            if batch_index % settings.log_interval_batches == 0 or batch_index == len(loader):
                report = score_metrics(torch.cat(logits_all), torch.cat(labels_all), settings.classification_threshold)
                mean_losses = dict(zip(names, (totals / seen).cpu().tolist()))
                print(f"epoch {epoch}/{settings.epochs} batch {batch_index}/{len(loader)} total={mean_losses['total']:.5f} "
                      f"vision={mean_losses['vision']:.5f} answer={mean_losses['answer']:.5f} "
                      f"logit={mean_losses['logit']:.5f} bce={mean_losses['classification']:.5f} "
                      f"v_bal={mean_losses['vision_balance']:.4f} l_bal={mean_losses['language_balance']:.4f} "
                      f"acc={report['accuracy']:.4f} AUROC={report['auroc']:.4f} "
                      f"v_sel={[round(x,3) for x in (v_selection/seen).cpu().tolist()]} "
                      f"v_w={[round(x,3) for x in (v_weights/seen).cpu().tolist()]}", flush=True)
        train_report = score_metrics(torch.cat(logits_all), torch.cat(labels_all), settings.classification_threshold)
        mean_losses = dict(zip(names, (totals / seen).cpu().tolist()))
        test_report = evaluate_student(model, replacement, head, test_loader, settings, device)
        if scheduler: scheduler.step()
        row = {"epoch": epoch, **{f"loss_{key}": value for key, value in mean_losses.items()},
               **{f"train_{key}": value for key, value in train_report.items() if isinstance(value, (int, float))},
               **{f"test_{key}": value for key, value in test_report.items() if isinstance(value, (int, float))},
               "epoch_time_sec": time.perf_counter() - started, "samples_this_epoch": len(sampler),
               "student_language_mode": settings.student_language_mode, "phase_dropout_active": dropout_active,
               "selection_split": "test", "selection_metric": "auroc"}
        for expert in range(settings.num_experts):
            row[f"vision_expert_{expert}_selection_rate"] = float((v_selection[expert] / seen).cpu())
            row[f"vision_expert_{expert}_mean_weight"] = float((v_weights[expert] / seen).cpu())
            row[f"language_expert_{expert}_selection_rate"] = float((l_selection[expert] / seen).cpu())
            row[f"language_expert_{expert}_mean_weight"] = float((l_weights[expert] / seen).cpu())
        history.append(row); write_csv(settings.output_dir / "metrics" / "student_training_history.csv", history, list(row))
        write_json(settings.output_dir / "metrics" / "student_training_latest.json", row)
        save_student_parts(settings.output_dir, replacement, head, "last", epoch, row)
        if epoch % settings.checkpoint_interval_epochs == 0:
            save_student_parts(settings.output_dir, replacement, head, f"epoch_{epoch:04d}", epoch, row)
        if float(test_report["auroc"]) > best:
            best = float(test_report["auroc"]); save_student_parts(settings.output_dir, replacement, head, "best", epoch, row)
            write_json(settings.output_dir / "metrics" / "best_test.json", row)
        if settings.visualization_enabled and settings.save_training_curves:
            save_training_curves(history, settings.output_dir / "figures" / "student_training_curves.png")
        if settings.visualization_enabled and settings.save_phase_masks and epoch % settings.visualization_interval_epochs == 0:
            save_phase_masks(replacement.vision_surrogate.core,
                             settings.output_dir / "figures" / f"vision_phase_masks_epoch_{epoch:04d}.png", f"Vision epoch {epoch}")
            if settings.student_language_mode == "optical_moe":
                save_phase_masks(replacement.language_surrogate.core,
                                 settings.output_dir / "figures" / f"language_phase_masks_epoch_{epoch:04d}.png", f"Language epoch {epoch}")
        print(f"epoch {epoch:03d} complete train_acc={train_report['accuracy']:.4f} "
              f"test_acc={test_report['accuracy']:.4f} test_AUROC={test_report['auroc']:.4f} best={best:.4f}", flush=True)
    write_json(settings.output_dir / "metrics" / "student_training.json", {
        "epochs": settings.epochs, "best_metric": "test_auroc", "best_value": best,
        "student_language_mode": settings.student_language_mode,
        "protocol_warning": "test evaluated every epoch and used for checkpoint selection at user request",
    })


@torch.inference_mode()
def evaluate_student(model: nn.Module, replacement: Any, head: nn.Module, loader: Any, settings: Any,
                     device: torch.device, dataset: Dataset[Any] | None = None,
                     predictions_path: Path | None = None) -> dict[str, Any]:
    replacement.use_student(); replacement.set_phase_dropout_active(False); model.eval(); head.eval()
    logits_all: list[torch.Tensor] = []; labels_all: list[torch.Tensor] = []; indices_all: list[torch.Tensor] = []
    for cpu_inputs, labels, indices in loader:
        replacement.prepare_student_batch(cpu_inputs["attention_mask"]); inputs = move_inputs(cpu_inputs, device)
        hidden = multimodal_forward_features(model, inputs); answer, _ = pool_answer_hidden_state(hidden, inputs["attention_mask"])
        logits_all.append(head(answer).float().cpu()); labels_all.append(labels.float().cpu()); indices_all.append(indices)
    logits = torch.cat(logits_all); labels = torch.cat(labels_all); indices = torch.cat(indices_all)
    report = {**score_metrics(logits, labels, settings.classification_threshold),
              "dataset": settings.dataset, "model": f"vision_optical_moe_language_{settings.student_language_mode}",
              "student_language_mode": settings.student_language_mode, "language_model_used": True,
              "prompt_template": settings.prompt_template,
              "pair_manifest_digest": (settings.pair_manifest_digests or {}).get("test")}
    if predictions_path is not None and dataset is not None:
        _write_predictions(predictions_path, dataset, indices, labels, logits, settings.classification_threshold)
        report["predictions_csv"] = str(predictions_path)
    return report


def save_head(head: NormalizedBinaryClassificationHead, path: Path, settings: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "head": head.specification(), "dataset": settings.dataset,
                "pair_manifest_digests": settings.pair_manifest_digests,
                "checkpoint_selection": "test_auroc"}, path)


def load_head(path: Path, settings: Any, device: torch.device) -> NormalizedBinaryClassificationHead:
    if not path.is_file(): raise FileNotFoundError(f"Teacher head missing: {path}. Run teacher_train.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("dataset") not in (None, settings.dataset): raise RuntimeError("Teacher head dataset mismatch")
    head = build_head(settings, settings.text_hidden_size).to(device); head.load_state_dict(payload["state_dict"]); return head


def save_student_parts(output_dir: Path, replacement: Any, head: nn.Module, tag: str, epoch: int,
                       metrics: dict[str, Any]) -> None:
    root = output_dir / "checkpoints"; root.mkdir(parents=True, exist_ok=True)
    metadata = {"epoch": epoch, "metrics": metrics, "language_mode": replacement.language_mode}
    torch.save({"state_dict": replacement.vision_surrogate.state_dict(), **metadata}, root / f"vision_moe_{tag}.pt")
    if replacement.language_mode == "optical_moe":
        torch.save({"state_dict": replacement.language_surrogate.state_dict(), **metadata}, root / f"language_moe_{tag}.pt")
    torch.save({"state_dict": head.state_dict(), "head": head.specification(), **metadata}, root / f"student_head_{tag}.pt")


def load_student_parts(output_dir: Path, replacement: Any, head: nn.Module, tag: str) -> None:
    root = output_dir / "checkpoints"; vision = root / f"vision_moe_{tag}.pt"; head_path = root / f"student_head_{tag}.pt"
    if not vision.is_file() or not head_path.is_file(): raise FileNotFoundError(f"Incomplete student checkpoint {tag} in {root}")
    replacement.vision_surrogate.load_state_dict(torch.load(vision, map_location="cpu", weights_only=True)["state_dict"])
    if replacement.language_mode == "optical_moe":
        language = root / f"language_moe_{tag}.pt"
        if not language.is_file(): raise FileNotFoundError(f"Missing language MoE checkpoint: {language}")
        replacement.language_surrogate.load_state_dict(torch.load(language, map_location="cpu", weights_only=True)["state_dict"])
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True)["state_dict"])


def save_student_inference(report: dict[str, Any], settings: Any, replacement: Any,
                           predictions_path: Path | None) -> None:
    report["vision_parameter_breakdown"] = replacement.vision_surrogate.parameter_breakdown()
    report["language_parameter_breakdown"] = (replacement.language_surrogate.parameter_breakdown()
                                                if replacement.language_mode == "optical_moe" else None)
    report["checkpoint_selection"] = "best test AUROC (user-requested; selection-biased)"
    write_json(settings.output_dir / "metrics" / "student_inference.json", report)
    if settings.visualization_enabled and settings.save_confusion_matrix:
        save_confusion_matrix(report["confusion_matrix"], settings.output_dir / "figures" / "student_confusion_matrix.png",
                              "Optical MoE student")


def _write_predictions(path: Path, dataset: Dataset[Any], indices: torch.Tensor, labels: torch.Tensor,
                       logits: torch.Tensor, threshold: float) -> None:
    probabilities = probabilities_from_logits(logits.numpy()); predicted = (probabilities >= threshold).astype(np.int64)
    rows: list[dict[str, Any]] = []
    for position, (index, label, logit, probability, prediction) in enumerate(
            zip(indices.tolist(), labels.tolist(), logits.tolist(), probabilities.tolist(), predicted.tolist())):
        metadata = sample_metadata(dataset, int(index))
        rows.append({
            "pair_id": metadata.get("pair_id", index), "image_id": metadata.get("image_id", ""),
            "filename": metadata.get("filename", ""), "caption": metadata.get("caption", ""),
            "caption_source_image_id": metadata.get("caption_source_image_id", ""),
            "label": int(label), "raw_logit": float(logit), "probability": float(probability),
            "predicted_label": int(prediction), "correct": bool(int(prediction) == int(label)),
        })
    if rows: write_csv(path, rows, list(rows[0]))
