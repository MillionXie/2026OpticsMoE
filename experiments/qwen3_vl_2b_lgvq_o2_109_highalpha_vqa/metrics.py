from __future__ import annotations

import math
from typing import Any

import torch


def _rankdata(value: torch.Tensor) -> torch.Tensor:
    """Average ranks for ties, matching scipy.stats.rankdata(method='average')."""

    value = value.detach().double().flatten()
    order = torch.argsort(value, stable=True)
    sorted_value = value[order]
    ranks = torch.empty_like(value)
    start = 0
    while start < len(value):
        end = start + 1
        while end < len(value) and bool(sorted_value[end] == sorted_value[start]):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pcc(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().double().flatten()
    right = right.detach().double().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) <= 1.0e-12:
        return float("nan")
    return float((left * right).sum() / denominator)


def _kendall_tau_b(left: torch.Tensor, right: torch.Tensor) -> float:
    """Kendall's tau-b, including the tie correction used for KRCC.

    The implementation stays in PyTorch so evaluation does not acquire a
    SciPy runtime dependency.  Computing one row of pairs at a time keeps the
    temporary memory linear in the number of videos rather than materialising
    an ``N x N`` matrix.
    """

    left = left.detach().double().flatten().cpu()
    right = right.detach().double().flatten().cpu()
    if left.numel() != right.numel():
        raise ValueError("Kendall inputs must have the same number of values")
    if left.numel() < 2:
        return float("nan")

    numerator = left.new_zeros(())
    untied_left = 0
    untied_right = 0
    for index in range(left.numel() - 1):
        left_difference = left[index] - left[index + 1 :]
        right_difference = right[index] - right[index + 1 :]
        numerator += (
            left_difference.sign() * right_difference.sign()
        ).sum()
        untied_left += int((left_difference != 0).sum())
        untied_right += int((right_difference != 0).sum())

    denominator = math.sqrt(float(untied_left) * float(untied_right))
    if denominator <= 1.0e-12:
        return float("nan")
    return float(numerator) / denominator


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    if prediction.ndim != 2 or tuple(prediction.shape) != tuple(target.shape) or prediction.shape[1] != 2:
        raise ValueError("Prediction and target must both be [N,2]")
    report: dict[str, Any] = {}
    for index, name in enumerate(("spatial", "temporal")):
        predicted = prediction[:, index].float()
        truth = target[:, index].float()
        error = predicted - truth
        report[name] = {
            "srcc": _pcc(_rankdata(predicted), _rankdata(truth)),
            "krcc": _kendall_tau_b(predicted, truth),
            "plcc": _pcc(predicted, truth),
            "rmse": float(error.square().mean().sqrt()),
            "mae": float(error.abs().mean()),
            "count": len(predicted),
        }
    finite_srcc = [
        value["srcc"] for value in report.values() if math.isfinite(value["srcc"])
    ]
    report["selection_mean_srcc"] = (
        sum(finite_srcc) / len(finite_srcc) if finite_srcc else float("-inf")
    )
    report["alignment"] = {"enabled": False, "prediction_column": None}
    return report


__all__ = ["regression_metrics"]


