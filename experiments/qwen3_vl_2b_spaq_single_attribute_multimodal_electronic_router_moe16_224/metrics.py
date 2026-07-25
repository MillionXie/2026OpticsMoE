from __future__ import annotations

import math
from typing import Sequence

import numpy as np

def regression_metrics(true_scores: Sequence[float], predicted_scores: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(true_scores, dtype=np.float64)
    prediction = np.asarray(predicted_scores, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("true_scores and predicted_scores must be matching one-dimensional arrays")
    if len(truth) == 0:
        raise ValueError("Cannot compute regression metrics for an empty sequence")
    absolute = np.abs(prediction - truth)
    mae = float(np.mean(absolute))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "srcc": _safe_correlation(_average_ranks(truth), _average_ranks(prediction)),
        "plcc": _safe_correlation(truth, prediction),
        "within_5_accuracy": float(np.mean(absolute <= 5.0)),
        "within_10_accuracy": float(np.mean(absolute <= 10.0)),
        "samples": int(len(truth)),
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
