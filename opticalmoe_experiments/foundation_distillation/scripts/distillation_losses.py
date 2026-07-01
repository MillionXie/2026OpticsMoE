from typing import Dict

import torch
import torch.nn.functional as F


def feature_distillation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    semantic_feature_normalized: torch.Tensor,
    teacher_feature: torch.Tensor,
    outside_camera_energy_ratio: torch.Tensor = None,
    ce_weight: float = 1.0,
    feature_distill_weight: float = 0.5,
    leak_loss_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    teacher = F.normalize(teacher_feature.float(), dim=-1)
    student = F.normalize(semantic_feature_normalized.float(), dim=-1)
    cosine = F.cosine_similarity(student, teacher, dim=-1).mean()
    ce = F.cross_entropy(logits, labels)
    feature = 1.0 - cosine
    if outside_camera_energy_ratio is None:
        outside_camera_energy_ratio = logits.new_zeros(logits.shape[0])
    outside_ratio = torch.as_tensor(outside_camera_energy_ratio, device=logits.device).float().mean()
    leak = outside_ratio
    total = (
        float(ce_weight) * ce
        + float(feature_distill_weight) * feature
        + float(leak_loss_weight) * leak
    )
    return {
        "total_loss": total,
        "ce_loss": ce,
        "feature_loss": feature,
        "feature_cosine": cosine,
        "leak_loss": leak,
        "outside_camera_energy_ratio": outside_ratio,
    }
