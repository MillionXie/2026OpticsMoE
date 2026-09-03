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
    while start < value.numel():
        stop = start + 1
        while stop < value.numel() and bool(sorted_value[stop] == sorted_value[start]):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _pcc(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().double().flatten()
    right = right.detach().double().flatten()
    left, right = left - left.mean(), right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator) if float(denominator) > 1.0e-12 else float("nan")


def _kendall(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().double().flatten().cpu()
    right = right.detach().double().flatten().cpu()
    numerator = left.new_zeros(())
    untied_left = 0
    untied_right = 0
    for index in range(left.numel() - 1):
        dl = left[index] - left[index + 1 :]
        dr = right[index] - right[index + 1 :]
        numerator += (dl.sign() * dr.sign()).sum()
        untied_left += int((dl != 0).sum())
        untied_right += int((dr != 0).sum())
    denominator = math.sqrt(float(untied_left) * float(untied_right))
    return float(numerator) / denominator if denominator > 1.0e-12 else float("nan")


def regression_metrics(
    prediction: torch.Tensor, target: torch.Tensor, target_name: str
) -> dict[str, Any]:
    prediction = prediction.detach().float().flatten()
    target = target.detach().float().flatten()
    if prediction.shape != target.shape or target_name not in {"spatial", "temporal"}:
        raise ValueError("Single-metric inputs or target name are invalid")
    error = prediction - target
    return {
        "target": target_name,
        "srcc": _pcc(_rankdata(prediction), _rankdata(target)),
        "krcc": _kendall(prediction, target),
        "plcc": _pcc(prediction, target),
        "rmse": float(error.square().mean().sqrt()),
        "mae": float(error.abs().mean()),
        "count": int(target.numel()),
    }


__all__ = ["regression_metrics"]
