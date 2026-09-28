"""Regression losses used by the temporal optical model."""

import torch
from torch.nn import functional as F


def pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    minimum_difference: float = 0.05,
) -> torch.Tensor:
    prediction, target = prediction.flatten(), target.flatten()
    difference = target[:, None] - target[None, :]
    predicted = prediction[:, None] - prediction[None, :]
    valid = torch.triu(
        torch.ones_like(difference, dtype=torch.bool), diagonal=1
    ) & (difference.abs() >= minimum_difference)
    if not bool(valid.any()):
        return prediction.new_zeros(())
    return F.softplus(-difference[valid].sign() * predicted[valid]).mean()


def batch_correlation_loss(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1.0e-6
) -> torch.Tensor:
    prediction, target = prediction.float().flatten(), target.float().flatten()
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    target_energy = target.square().sum()
    if float(target_energy.detach()) <= epsilon:
        return prediction.new_zeros(())
    prediction_energy = prediction.square().sum()
    denominator = (
        prediction_energy.clamp_min(epsilon) * target_energy.clamp_min(epsilon)
    ).sqrt()
    return 1.0 - ((prediction * target).sum() / denominator).clamp(-1.0, 1.0)
