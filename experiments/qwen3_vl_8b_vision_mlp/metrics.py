from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ClassificationResult:
    top1_accuracy: float
    top5_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_accuracy: dict[str, float]
    confusion_matrix: list[list[int]]


def classification_metrics(
    labels: Sequence[int], predictions: Sequence[int], top5_predictions: Sequence[Sequence[int]],
    class_names: Sequence[str],
) -> ClassificationResult:
    count = len(class_names)
    confusion = torch.zeros((count, count), dtype=torch.long)
    top5_correct = 0
    for label, prediction, candidates in zip(labels, predictions, top5_predictions):
        confusion[int(label), int(prediction)] += 1
        top5_correct += int(int(label) in candidates)
    total = len(labels)
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    per_class: dict[str, float] = {}
    for index, name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class[name] = recall
    return ClassificationResult(
        top1_accuracy=sum(int(a == b) for a, b in zip(labels, predictions)) / total if total else 0.0,
        top5_accuracy=top5_correct / total if total else 0.0,
        macro_precision=sum(precision_values) / count if count else 0.0,
        macro_recall=sum(recall_values) / count if count else 0.0,
        macro_f1=sum(f1_values) / count if count else 0.0,
        per_class_accuracy=per_class,
        confusion_matrix=confusion.tolist(),
    )


def metrics_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, class_names: Sequence[str]
) -> ClassificationResult:
    predictions = logits.argmax(dim=-1)
    k = min(5, logits.shape[-1])
    top5 = logits.topk(k, dim=-1).indices
    return classification_metrics(
        labels.tolist(), predictions.tolist(), top5.tolist(), class_names
    )

