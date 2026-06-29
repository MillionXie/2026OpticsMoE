from typing import Dict

import torch
import torch.nn.functional as F


def feature_distillation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    projected_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    ce_weight: float = 1.0,
    feature_distill_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    teacher = F.normalize(teacher_feature.float(), dim=-1)
    student = F.normalize(projected_feature.float(), dim=-1)
    cosine = F.cosine_similarity(student, teacher, dim=-1).mean()
    ce = F.cross_entropy(logits, labels)
    feature = 1.0 - cosine
    total = float(ce_weight) * ce + float(feature_distill_weight) * feature
    return {"total_loss": total, "ce_loss": ce, "feature_loss": feature, "feature_cosine": cosine}

