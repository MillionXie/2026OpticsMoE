from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .feature_probe import load_feature_cache
from .io_utils import write_csv, write_json
from .metrics import metrics_from_logits
from .modeling import VisionFieldProbeHead
from .visualization import save_confusion_matrix, save_training_curves


class CachedVisionFieldDataset(Dataset[Any]):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.features = payload["features"].float()
        self.labels = payload["labels"].long().tolist()
        self.sample_indices = payload["sample_indices"].long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], self.sample_indices[index]


def build_probe(settings: Any, num_classes: int) -> VisionFieldProbeHead:
    return VisionFieldProbeHead(
        input_dim=settings.optical_field_size ** 2, num_classes=num_classes,
        head_type=settings.probe_head_type, hidden_dim=settings.probe_hidden_dim,
        dropout=settings.probe_dropout,
    )


def _split_indices(labels: Sequence[int], validation_fraction: float, seed: int,
                   num_classes: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    train: list[int] = []; validation: list[int] = []
    for class_index in range(num_classes):
        indices = [index for index, label in enumerate(labels) if label == class_index]
        if not indices:
            raise RuntimeError(f"Training feature cache has no samples for class {class_index}")
        order = torch.randperm(len(indices), generator=generator).tolist()
        count = min(max(round(len(indices) * validation_fraction), 1), len(indices) - 1) if len(indices) > 1 else 0
        validation.extend(indices[position] for position in order[:count])
        train.extend(indices[position] for position in order[count:])
    return sorted(train), sorted(validation)


def _loader(dataset: Dataset[Any], indices: Sequence[int] | None, batch_size: int,
            shuffle: bool, workers: int, seed: int) -> DataLoader[Any]:
    selected: Dataset[Any] = Subset(dataset, list(indices)) if indices is not None else dataset
    return DataLoader(selected, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(seed))


def train_probe(settings: Any, class_names: Sequence[str], device: torch.device,
                source_parameters: dict[str, int] | None = None) -> dict[str, Any]:
    payload = load_feature_cache(settings.output_dir / "features" / "train_vision_input_field.pt")
    if list(payload["class_names"]) != list(class_names):
        raise RuntimeError("Feature cache class names do not match the current dataset")
    dataset = CachedVisionFieldDataset(payload)
    train_indices, validation_indices = _split_indices(dataset.labels, settings.validation_fraction,
                                                        settings.seed, len(class_names))
    train_loader = _loader(dataset, train_indices, settings.head_batch_size, True,
                           settings.num_workers, settings.seed)
    validation_loader = _loader(dataset, validation_indices, settings.head_batch_size, False,
                                settings.num_workers, settings.seed)
    head = build_probe(settings, len(class_names)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    history: list[dict[str, Any]] = []
    best_macro = -1.0; best_top1 = -1.0
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter(); head.train(); seen = 0; loss_sum = 0.0
        logits_chunks: list[torch.Tensor] = []; label_chunks: list[torch.Tensor] = []
        for features, labels, _ in train_loader:
            features = features.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); logits = head(features); loss = F.cross_entropy(logits, labels)
            loss.backward(); optimizer.step()
            seen += len(labels); loss_sum += float(loss.detach()) * len(labels)
            logits_chunks.append(logits.detach().cpu()); label_chunks.append(labels.detach().cpu())
        train_metrics = metrics_from_logits(torch.cat(logits_chunks), torch.cat(label_chunks), class_names)
        validation = evaluate_probe(head, validation_loader, class_names, device)
        row = {
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(1, seen), "train_top1_accuracy": train_metrics["top1_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"], "validation_loss": validation["loss"],
            "validation_top1_accuracy": validation["metrics"]["top1_accuracy"],
            "validation_macro_f1": validation["metrics"]["macro_f1"],
            "validation_balanced_accuracy": validation["metrics"]["balanced_accuracy"],
            "epoch_time_sec": time.perf_counter() - started,
        }
        history.append(row)
        write_csv(settings.output_dir / "metrics" / "probe_training_history.csv", history, list(row))
        save_training_curves(history, settings.output_dir / "figures" / "probe_training_curves.png")
        improved = row["validation_macro_f1"] > best_macro or row["validation_top1_accuracy"] > best_top1
        if improved:
            best_macro = max(best_macro, row["validation_macro_f1"])
            best_top1 = max(best_top1, row["validation_top1_accuracy"])
            _save_probe(head, settings.output_dir / "checkpoints" / "probe_head_best.pt", settings, class_names, row)
            write_json(settings.output_dir / "metrics" / "best_validation.json", row)
        _save_probe(head, settings.output_dir / "checkpoints" / "probe_head_last.pt", settings, class_names, row)
        scheduler.step()
        print(f"[train_probe] epoch={epoch}/{settings.epochs} loss={row['train_loss']:.5f} train_top1={row['train_top1_accuracy']:.4f} val_top1={row['validation_top1_accuracy']:.4f} val_macro_f1={row['validation_macro_f1']:.4f}", flush=True)
    specification = head.specification()
    report = {
        **(source_parameters or {}), "feature_type": "vision_optical_input_field",
        "feature_dim": settings.optical_field_size ** 2, "probe_head": specification,
        "probe_head_parameters": specification["parameters"],
        "probe_total_trainable_parameters": specification["trainable_parameters"],
        "probe_head_type": settings.probe_head_type,
        "probe_hidden_dim": None if settings.probe_head_type == "linear" else settings.probe_hidden_dim,
        "finetune_vision_input_adapter": False,
    }
    write_json(settings.output_dir / "metrics" / "probe_model.json", report)
    return {"best_validation_top1": best_top1, "best_validation_macro_f1": best_macro, **report}


@torch.no_grad()
def evaluate_probe(head: nn.Module, loader: DataLoader[Any], class_names: Sequence[str],
                   device: torch.device) -> dict[str, Any]:
    head.eval(); logits_chunks = []; label_chunks = []; index_chunks = []; loss_sum = 0.0; seen = 0
    for features, labels, sample_indices in loader:
        labels_device = labels.to(device, non_blocking=True)
        logits = head(features.to(device, non_blocking=True))
        loss = F.cross_entropy(logits, labels_device)
        seen += len(labels); loss_sum += float(loss) * len(labels)
        logits_chunks.append(logits.cpu()); label_chunks.append(labels.cpu()); index_chunks.append(sample_indices.cpu())
    logits = torch.cat(logits_chunks); labels = torch.cat(label_chunks); indices = torch.cat(index_chunks)
    return {"metrics": metrics_from_logits(logits, labels, class_names), "loss": loss_sum / max(1, seen),
            "logits": logits, "labels": labels, "sample_indices": indices}


def probe_inference(settings: Any, class_names: Sequence[str], device: torch.device) -> dict[str, Any]:
    payload = load_feature_cache(settings.output_dir / "features" / "test_vision_input_field.pt")
    dataset = CachedVisionFieldDataset(payload)
    loader = _loader(dataset, None, settings.inference_batch_size, False, settings.num_workers, settings.seed)
    head = load_probe(settings.output_dir / "checkpoints" / "probe_head_best.pt", settings,
                      len(class_names), device)
    result = evaluate_probe(head, loader, class_names, device)
    specification = head.specification()
    report = {
        **result["metrics"], "feature_type": "vision_optical_input_field", "feature_dim": 4096,
        "source_experiment_dir": str(settings.source_experiment_dir),
        "source_vision_checkpoint": str(settings.source_vision_checkpoint), "probe_head": specification,
    }
    write_json(settings.output_dir / "metrics" / "probe_inference.json", report)
    _save_predictions(result, class_names, settings.output_dir)
    save_confusion_matrix(report["confusion_matrix"], class_names,
                          settings.output_dir / "figures" / "probe_confusion_matrix.png")
    write_csv(settings.output_dir / "metrics" / "probe_confusion_matrix.csv",
              [{"true_name": name, **{predicted: report["confusion_matrix"][row][column]
                for column, predicted in enumerate(class_names)}} for row, name in enumerate(class_names)],
              ["true_name", *class_names])
    _write_source_comparison(settings, report)
    return report


def _save_predictions(result: dict[str, Any], class_names: Sequence[str], output_dir: Path) -> None:
    logits = result["logits"]; labels = result["labels"]; predictions = logits.argmax(1)
    rows = []
    for offset in range(len(labels)):
        truth = int(labels[offset]); prediction = int(predictions[offset])
        row = {"sample_index": int(result["sample_indices"][offset]), "true_label": truth,
               "true_name": class_names[truth], "pred_label": prediction,
               "pred_name": class_names[prediction], "correct": truth == prediction}
        row.update({f"logit_{name}": float(logits[offset, index]) for index, name in enumerate(class_names)})
        rows.append(row)
    write_csv(output_dir / "metrics" / "probe_predictions.csv", rows, list(rows[0]))


def _save_probe(head: VisionFieldProbeHead, path: Path, settings: Any,
                class_names: Sequence[str], epoch_metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "probe_head": head.specification(),
                "class_names": list(class_names), "epoch_metrics": dict(epoch_metrics)}, path)


def load_probe(path: Path, settings: Any, num_classes: int,
               device: torch.device) -> VisionFieldProbeHead:
    if not path.is_file():
        raise FileNotFoundError(f"Probe checkpoint not found: {path}; run --phase train_probe")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    head = build_probe(settings, num_classes)
    specification = payload.get("probe_head", {})
    current = head.specification()
    for key in ("input_dim", "head_type", "hidden_dim", "num_classes"):
        if specification.get(key) != current.get(key):
            raise RuntimeError(f"Probe checkpoint {key}={specification.get(key)!r} does not match config {current.get(key)!r}")
    head.load_state_dict(payload["state_dict"])
    return head.to(device)


def _write_source_comparison(settings: Any, probe: dict[str, Any]) -> None:
    metrics_dir = settings.source_experiment_dir / "metrics"
    teacher_path = metrics_dir / "teacher_inference.json"; student_path = metrics_dir / "student_inference.json"
    if not teacher_path.is_file() or not student_path.is_file():
        warnings.warn(
            f"Source teacher/student metrics not both available under {metrics_dir}; comparison was skipped"
        )
        return
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    student = json.loads(student_path.read_text(encoding="utf-8"))
    teacher_samples = teacher.get("samples")
    student_samples = student.get("samples")
    probe_samples = probe.get("samples")
    same_test_sample_count = teacher_samples == student_samples == probe_samples
    write_json(settings.output_dir / "metrics" / "probe_vs_student_comparison.json", {
        "teacher_top1_accuracy": teacher.get("top1_accuracy"),
        "student_optical_top1_accuracy": student.get("top1_accuracy"),
        "visionfield_probe_top1_accuracy": probe.get("top1_accuracy"),
        "teacher_macro_f1": teacher.get("macro_f1"),
        "student_optical_macro_f1": student.get("macro_f1"),
        "visionfield_probe_macro_f1": probe.get("macro_f1"),
        "teacher_samples": teacher_samples,
        "student_optical_samples": student_samples,
        "visionfield_probe_samples": probe_samples,
        "same_test_sample_count": same_test_sample_count,
        "comparison_note": (
            "Metrics use the same test sample count."
            if same_test_sample_count else
            "Sample counts differ; this is a smoke/subset comparison and is not a controlled accuracy comparison."
        ),
        "source_experiment_dir": str(settings.source_experiment_dir),
    })
