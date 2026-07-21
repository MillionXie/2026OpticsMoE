from __future__ import annotations

from typing import Sequence

import numpy as np


def binary_classification_metrics(labels: Sequence[float] | np.ndarray,
                                  logits: Sequence[float] | np.ndarray,
                                  threshold: float = 0.5) -> dict[str, object]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.size != z.size:
        raise ValueError("Binary labels/logits must be non-empty and have the same length")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(z).all():
        raise ValueError("Binary metrics require finite logits and labels in {0,1}")
    probabilities = _sigmoid(z)
    predicted = (probabilities >= threshold).astype(np.int64)
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
    precision = _safe_div(tp, tp + fp); recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    bce = float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))
    return {
        "bce_loss": bce,
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "auroc": _auroc(y, z),
        "average_precision": _average_precision(y, z),
        "auprc": _average_precision(y, z),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "positive_samples": positives,
        "negative_samples": negatives,
        "samples": int(y.size),
        "threshold": float(threshold),
        "raw_logit_min": float(z.min()), "raw_logit_max": float(z.max()),
        "probability_min": float(probabilities.min()), "probability_max": float(probabilities.max()),
    }


def probabilities_from_logits(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    return _sigmoid(np.asarray(logits, dtype=np.float64))


def _auroc(y: np.ndarray, scores: np.ndarray) -> float:
    positive = y == 1; negative = ~positive
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative samples")
    ranks = _average_ranks(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(y == 1))
    if positives == 0:
        raise ValueError("Average precision requires positive samples")
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    return float(np.sum(precision * sorted_y) / positives)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
