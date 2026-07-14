import math

import torch


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else 0.0


def regression_metrics(targets, predictions):
    targets = torch.as_tensor(targets, dtype=torch.float64).flatten(); predictions = torch.as_tensor(predictions, dtype=torch.float64).flatten()
    if targets.numel() != predictions.numel() or targets.numel() == 0: raise ValueError("Regression targets/predictions must be nonempty and aligned")
    difference = predictions - targets; mae = difference.abs().mean(); rmse = difference.square().mean().sqrt()
    try:
        from scipy.stats import kendalltau, pearsonr, spearmanr
        plcc = pearsonr(targets.numpy(), predictions.numpy()).statistic
        srocc = spearmanr(targets.numpy(), predictions.numpy()).statistic
        krocc = kendalltau(targets.numpy(), predictions.numpy()).statistic
    except Exception:
        centered_target = targets - targets.mean(); centered_prediction = predictions - predictions.mean()
        plcc = (centered_target * centered_prediction).sum() / (centered_target.norm() * centered_prediction.norm() + 1e-12)
        target_rank = torch.argsort(torch.argsort(targets)).double(); prediction_rank = torch.argsort(torch.argsort(predictions)).double()
        target_rank -= target_rank.mean(); prediction_rank -= prediction_rank.mean()
        srocc = (target_rank * prediction_rank).sum() / (target_rank.norm() * prediction_rank.norm() + 1e-12); krocc = 0.0
    return {"samples": int(targets.numel()), "mae": float(mae), "rmse": float(rmse), "plcc": _safe_float(plcc), "srocc": _safe_float(srocc), "krocc": _safe_float(krocc), "prediction_mean": float(predictions.mean()), "target_mean": float(targets.mean())}


def denormalize_quality(normalized, score_min, score_max, higher_is_better):
    value = torch.as_tensor(normalized, dtype=torch.float64)
    if not bool(higher_is_better): value = 1.0 - value
    return value * (float(score_max) - float(score_min)) + float(score_min)

