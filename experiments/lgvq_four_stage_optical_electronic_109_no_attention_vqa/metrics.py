from __future__ import annotations

import math
from typing import Any

import torch


def _rankdata(value: torch.Tensor) -> torch.Tensor:
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
    left = left.detach().double().flatten() - left.detach().double().mean()
    right = right.detach().double().flatten() - right.detach().double().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator) if float(denominator) > 1.0e-12 else float("nan")


def _kendall(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = left.detach().double().flatten().cpu(), right.detach().double().flatten().cpu()
    numerator = left.new_zeros(())
    untied_left = untied_right = 0
    for index in range(left.numel() - 1):
        dl, dr = left[index] - left[index + 1 :], right[index] - right[index + 1 :]
        numerator += (dl.sign() * dr.sign()).sum()
        untied_left += int((dl != 0).sum())
        untied_right += int((dr != 0).sum())
    denominator = math.sqrt(float(untied_left) * float(untied_right))
    return float(numerator) / denominator if denominator > 1.0e-12 else float("nan")


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    if prediction.ndim != 2 or prediction.shape != target.shape or prediction.shape[1] != 2:
        raise ValueError("Prediction and target must be [N,2]")
    report: dict[str, Any] = {}
    for column, name in enumerate(("spatial", "temporal")):
        predicted, truth = prediction[:, column].float(), target[:, column].float()
        error = predicted - truth
        report[name] = {
            "srcc": _pcc(_rankdata(predicted), _rankdata(truth)),
            "krcc": _kendall(predicted, truth),
            "plcc": _pcc(predicted, truth),
            "rmse": float(error.square().mean().sqrt()),
            "mae": float(error.abs().mean()),
            "count": int(predicted.numel()),
        }
    finite = [report[name]["srcc"] for name in ("spatial", "temporal") if math.isfinite(report[name]["srcc"])]
    report["selection_mean_srcc"] = sum(finite) / len(finite) if finite else float("-inf")
    report["alignment_target_enabled"] = False
    return report


__all__ = ["regression_metrics"]
