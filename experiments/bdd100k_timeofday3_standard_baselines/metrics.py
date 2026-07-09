from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch


def metrics_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, class_names: Sequence[str]
) -> dict[str, Any]:
    predictions = logits.argmax(dim=1)
    top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
    matrix = torch.zeros(len(class_names), len(class_names), dtype=torch.long)
    for truth, prediction in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
        matrix[int(truth), int(prediction)] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_values.append(f1)
        per_class[name] = {
            "support": support,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total = max(1, int(matrix.sum()))
    return {
        "top1_accuracy": float(matrix.diag().sum()) / total,
        "top5_accuracy": float(
            sum(int(int(label) in row) for label, row in zip(labels.cpu().tolist(), top5.cpu().tolist()))
        )
        / max(1, len(labels)),
        "macro_f1": sum(f1_values) / len(class_names),
        "balanced_accuracy": sum(recalls) / len(class_names),
        "per_class_accuracy": {name: values["accuracy"] for name, values in per_class.items()},
        "per_class_precision": {name: values["precision"] for name, values in per_class.items()},
        "per_class_recall": {name: values["recall"] for name, values in per_class.items()},
        "per_class_f1": {name: values["f1"] for name, values in per_class.items()},
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "samples": int(matrix.sum()),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    names = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, matrix: list[list[int]], class_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row])


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

