from __future__ import annotations

from typing import Any, Sequence

import torch


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, class_names: Sequence[str]) -> dict[str, Any]:
    logits = logits.float().cpu()
    labels = labels.long().cpu()
    predictions = logits.argmax(1)
    top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
    matrix = torch.zeros(len(class_names), len(class_names), dtype=torch.long)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[truth, prediction] += 1
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision[name] = true_positive / predicted if predicted else 0.0
        recall[name] = true_positive / support if support else 0.0
        denom = precision[name] + recall[name]
        f1[name] = 2.0 * precision[name] * recall[name] / denom if denom else 0.0
    total = max(1, len(labels))
    return {
        "top1_accuracy": float((predictions == labels).sum()) / total,
        "top5_accuracy": float(sum(int(int(label) in row) for label, row in zip(labels.tolist(), top5.tolist()))) / total,
        "macro_f1": sum(f1.values()) / len(class_names),
        "balanced_accuracy": sum(recall.values()) / len(class_names),
        "per_class_accuracy": dict(recall),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "confusion_matrix": matrix.tolist(),
        "samples": len(labels),
    }

