from __future__ import annotations

import torch
from torch.nn import functional as F


def cosine_distillation_loss(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise RuntimeError(
            f"Student/Teacher embeddings differ: {student.shape} vs {teacher.shape}"
        )
    return (
        1.0
        - F.cosine_similarity(student.float(), teacher.float(), dim=-1)
    ).mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("SupCon expects embeddings [B,D] and labels [B]")
    embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
    logits = embeddings @ embeddings.T / float(temperature)
    identity = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positives.any(dim=1)
    if not torch.all(valid):
        missing = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"Every PK-batch anchor needs a same-item positive; missing={missing}"
        )
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_denominator = torch.logsumexp(
        logits.masked_fill(identity, -torch.inf), dim=1
    )
    log_probability = logits - log_denominator[:, None]
    mean_positive = (
        (log_probability * positives).sum(dim=1) / positives.sum(dim=1)
    )
    return -mean_positive.mean()

