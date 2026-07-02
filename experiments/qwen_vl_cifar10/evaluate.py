from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class EvaluationResult:
    loss: float | None
    accuracy: float
    macro_f1: float
    per_class_accuracy: dict[str, float]
    labels: list[int]
    predictions: list[int]
    confusion_matrix: list[list[int]]


def evaluate_head(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    class_names: Sequence[str],
) -> EvaluationResult:
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    predictions: list[int] = []
    targets: list[int] = []
    loss_sum = 0.0
    head.eval()
    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = head(batch_features)
            loss_sum += float(criterion(logits, batch_labels).item())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(batch_labels.cpu().tolist())
    result = classification_metrics(targets, predictions, class_names)
    return EvaluationResult(
        loss=loss_sum / max(len(targets), 1),
        accuracy=result.accuracy,
        macro_f1=result.macro_f1,
        per_class_accuracy=result.per_class_accuracy,
        labels=result.labels,
        predictions=result.predictions,
        confusion_matrix=result.confusion_matrix,
    )


def classification_metrics(
    labels: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]
) -> EvaluationResult:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    num_classes = len(class_names)
    has_unparsed = any(prediction < 0 or prediction >= num_classes for prediction in predictions)
    num_columns = num_classes + int(has_unparsed)
    confusion = [[0 for _ in range(num_columns)] for _ in range(num_classes)]
    for label, prediction in zip(labels, predictions):
        if not 0 <= label < num_classes:
            raise ValueError(f"Label id out of range: {label}")
        column = prediction if 0 <= prediction < num_classes else num_classes
        confusion[label][column] += 1

    total = len(labels)
    correct = sum(confusion[index][index] for index in range(num_classes))
    per_class_accuracy: dict[str, float] = {}
    f1_values: list[float] = []
    for index, name in enumerate(class_names):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        per_class_accuracy[name] = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1_values.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

    return EvaluationResult(
        loss=None,
        accuracy=correct / total if total else 0.0,
        macro_f1=sum(f1_values) / num_classes if num_classes else 0.0,
        per_class_accuracy=per_class_accuracy,
        labels=list(labels),
        predictions=list(predictions),
        confusion_matrix=confusion,
    )


def write_predictions_csv(
    path: Path,
    labels: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    raw_outputs: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "index",
            "label",
            "prediction",
            "label_name",
            "predicted_name",
            "correct",
        ]
        if raw_outputs is not None:
            fieldnames.append("raw_output")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (label, prediction) in enumerate(zip(labels, predictions)):
            row: dict[str, object] = {
                "index": index,
                "label": label,
                "prediction": prediction,
                "label_name": class_names[label],
                "predicted_name": class_names[prediction]
                if 0 <= prediction < len(class_names)
                else "<unparsed>",
                "correct": int(label == prediction),
            }
            if raw_outputs is not None:
                row["raw_output"] = raw_outputs[index]
            writer.writerow(row)


def write_confusion_matrix_csv(
    path: Path, confusion_matrix: Sequence[Sequence[int]], class_names: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        has_unparsed = bool(confusion_matrix and len(confusion_matrix[0]) > len(class_names))
        writer.writerow(
            ["true\\predicted", *class_names, *(["<unparsed>"] if has_unparsed else [])]
        )
        for class_name, row in zip(class_names, confusion_matrix):
            writer.writerow([class_name, *row])
