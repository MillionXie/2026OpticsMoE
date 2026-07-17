from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from . import TASK_NAMES


def regression_metrics(true_scores: Sequence[float], predicted_scores: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(true_scores, dtype=np.float64)
    prediction = np.asarray(predicted_scores, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("true_scores and predicted_scores must be matching one-dimensional arrays")
    if len(truth) == 0:
        raise ValueError("Cannot compute regression metrics for an empty sequence")
    mae = float(np.mean(np.abs(prediction - truth)))
    return {
        "mae": mae,
        "srcc": _safe_correlation(_average_ranks(truth), _average_ranks(prediction)),
        "plcc": _safe_correlation(truth, prediction),
        "samples": int(len(truth)),
    }


def multitask_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, float]] = {}
    for task in TASK_NAMES:
        selected = [row for row in rows if row["task"] == task]
        if not selected:
            raise RuntimeError(f"No test predictions were produced for task {task}")
        by_task[task] = regression_metrics(
            [float(row["true_score"]) for row in selected],
            [float(row["predicted_score"]) for row in selected],
        )
    macro = {
        metric: float(np.mean([by_task[task][metric] for task in TASK_NAMES]))
        for metric in ("mae", "srcc", "plcc")
    }
    return {
        "score_scale": [0.0, 100.0],
        "tasks": by_task,
        "macro_average": macro,
        "test_task_samples": len(rows),
        "test_original_images": len({row["image_name"] for row in rows}),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0

