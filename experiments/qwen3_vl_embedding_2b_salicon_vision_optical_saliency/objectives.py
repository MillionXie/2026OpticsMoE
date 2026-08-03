from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def density_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise RuntimeError(f"Saliency logits must be [B,1,H,W], got {logits.shape}")
    flat = torch.softmax(logits.float().flatten(1), dim=1)
    return flat.reshape_as(logits)


def normalize_density(target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    if target.ndim != 4 or target.shape[1] != 1:
        raise RuntimeError(f"Density target must be [B,1,H,W], got {target.shape}")
    target = target.float().clamp_min(0)
    denominator = target.flatten(1).sum(dim=1).clamp_min(eps)
    return target / denominator[:, None, None, None]


def kl_divergence(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    prediction = normalize_density(prediction, eps)
    target = normalize_density(target, eps)
    return (
        target
        * (
            target.clamp_min(eps).log()
            - prediction.clamp_min(eps).log()
        )
    ).flatten(1).sum(dim=1).mean()


def correlation_coefficient(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    x = prediction.float().flatten(1)
    y = target.float().flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = (x * y).sum(dim=1)
    denominator = x.square().sum(dim=1).sqrt() * y.square().sum(dim=1).sqrt()
    return (numerator / denominator.clamp_min(eps)).mean()


def similarity(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    prediction = normalize_density(prediction)
    target = normalize_density(target)
    return torch.minimum(prediction, target).flatten(1).sum(dim=1).mean()


def normalized_scanpath_saliency(
    prediction: torch.Tensor,
    fixation: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    x = prediction.float().flatten(1)
    z = (x - x.mean(dim=1, keepdim=True)) / x.std(
        dim=1, unbiased=False, keepdim=True
    ).clamp_min(eps)
    mask = fixation.float().flatten(1).gt(0)
    counts = mask.sum(dim=1)
    valid = counts.gt(0)
    if not torch.any(valid):
        return prediction.new_zeros(())
    per_sample = (z * mask).sum(dim=1) / counts.clamp_min(1)
    return per_sample[valid].mean()


def saliency_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    fixation: torch.Tensor,
    settings: object,
    *,
    teacher_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = density_from_logits(logits)
    target = normalize_density(target)
    kl = kl_divergence(prediction, target)
    cc = correlation_coefficient(prediction, target)
    sim = similarity(prediction, target)
    nss = normalized_scanpath_saliency(prediction, fixation)
    if teacher_logits is None:
        kd = logits.new_zeros(())
    else:
        temperature = float(settings.map_kd_temperature)
        teacher = density_from_logits(teacher_logits.float() / temperature)
        student = density_from_logits(logits.float() / temperature)
        kd = kl_divergence(student, teacher) * (temperature**2)
    total = (
        float(settings.kl_weight) * kl
        + float(settings.cc_weight) * (1.0 - cc)
        + float(settings.sim_weight) * (1.0 - sim)
        - float(settings.nss_weight) * nss
        + float(settings.map_kd_weight) * kd
    )
    return total, {
        "kl": kl,
        "cc": cc,
        "sim": sim,
        "nss": nss,
        "map_kd": kd,
    }


def auc_judd(
    prediction: torch.Tensor, fixation: torch.Tensor
) -> float:
    saliency = prediction.detach().float().flatten()
    fixations = fixation.detach().flatten().gt(0)
    positive_count = int(fixations.sum())
    negative_count = len(fixations) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    # Sorting the complete map once is mathematically equivalent to sweeping
    # every saliency threshold, but avoids O(num_fixations * num_pixels)
    # repeated full-image comparisons during validation.
    order = torch.argsort(saliency, descending=True)
    sorted_saliency = saliency[order]
    sorted_fixations = fixations[order].to(torch.float64)
    cumulative_true_positive = sorted_fixations.cumsum(0) / positive_count
    cumulative_false_positive = (
        (1.0 - sorted_fixations).cumsum(0) / negative_count
    )
    # Evaluate only at the end of each equal-score group. This makes the
    # result deterministic and gives the expected 0.5 AUC for a constant map.
    group_end = torch.ones_like(sorted_saliency, dtype=torch.bool)
    group_end[:-1] = sorted_saliency[:-1] != sorted_saliency[1:]
    true_positive = cumulative_true_positive[group_end]
    false_positive = cumulative_false_positive[group_end]
    zero = true_positive.new_zeros(1)
    true_positive = torch.cat((zero, true_positive))
    false_positive = torch.cat((zero, false_positive))
    return float(torch.trapz(true_positive, false_positive))


@dataclass
class SaliencyAccumulator:
    count: int = 0
    kl_sum: float = 0.0
    cc_sum: float = 0.0
    sim_sum: float = 0.0
    nss_sum: float = 0.0
    auc_sum: float = 0.0
    auc_count: int = 0
    mae_sum: float = 0.0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        fixation: torch.Tensor,
    ) -> None:
        prediction = density_from_logits(logits)
        target = normalize_density(target)
        batch = len(logits)
        self.count += batch
        self.kl_sum += float(kl_divergence(prediction, target)) * batch
        self.cc_sum += float(correlation_coefficient(prediction, target)) * batch
        self.sim_sum += float(similarity(prediction, target)) * batch
        self.nss_sum += float(
            normalized_scanpath_saliency(prediction, fixation)
        ) * batch
        pred_max = prediction.flatten(1).amax(dim=1).clamp_min(1.0e-8)
        target_max = target.flatten(1).amax(dim=1).clamp_min(1.0e-8)
        normalized_prediction = prediction / pred_max[:, None, None, None]
        normalized_target = target / target_max[:, None, None, None]
        self.mae_sum += float(
            (normalized_prediction - normalized_target).abs().flatten(1).mean()
        ) * batch
        for index in range(batch):
            value = auc_judd(prediction[index], fixation[index])
            if math.isfinite(value):
                self.auc_sum += value
                self.auc_count += 1

    def compute(self) -> dict[str, float | int]:
        if self.count <= 0:
            raise RuntimeError("No saliency samples accumulated")
        return {
            "samples": self.count,
            "kld": self.kl_sum / self.count,
            "cc": self.cc_sum / self.count,
            "sim": self.sim_sum / self.count,
            "nss": self.nss_sum / self.count,
            "auc_judd": (
                self.auc_sum / self.auc_count
                if self.auc_count
                else float("nan")
            ),
            "mae": self.mae_sum / self.count,
        }
