from __future__ import annotations

import torch
from torch import nn
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


class CrossBatchMemory(nn.Module):
    """FIFO memory of detached embeddings used only as additional negatives/positives."""

    def __init__(self, capacity: int, embedding_dim: int) -> None:
        super().__init__()
        self.capacity = int(capacity)
        self.embedding_dim = int(embedding_dim)
        self.register_buffer(
            "embeddings", torch.empty(max(0, self.capacity), self.embedding_dim)
        )
        self.register_buffer(
            "labels", torch.empty(max(0, self.capacity), dtype=torch.long)
        )
        self.register_buffer("size", torch.zeros((), dtype=torch.long))
        self.register_buffer("cursor", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        if self.capacity == 0:
            return
        values = F.normalize(embeddings.detach().float(), p=2, dim=-1)
        targets = labels.detach().long()
        if len(values) >= self.capacity:
            values = values[-self.capacity :]
            targets = targets[-self.capacity :]
        count = len(values)
        start = int(self.cursor)
        first = min(count, self.capacity - start)
        self.embeddings[start : start + first].copy_(values[:first])
        self.labels[start : start + first].copy_(targets[:first])
        remaining = count - first
        if remaining:
            self.embeddings[:remaining].copy_(values[first:])
            self.labels[:remaining].copy_(targets[first:])
        self.cursor.fill_((start + count) % self.capacity)
        self.size.fill_(min(self.capacity, int(self.size) + count))

    def values(self) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(self.size)
        return self.embeddings[:count], self.labels[:count]


def supervised_contrastive_loss_with_memory(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    memory: CrossBatchMemory | None,
) -> torch.Tensor:
    if memory is None or int(memory.size) == 0:
        return supervised_contrastive_loss(embeddings, labels, temperature)
    anchors = F.normalize(embeddings.float(), p=2, dim=-1)
    remembered, remembered_labels = memory.values()
    remembered = remembered.to(anchors.device, non_blocking=True)
    remembered_labels = remembered_labels.to(labels.device, non_blocking=True)
    candidates = torch.cat((anchors, remembered), dim=0)
    candidate_labels = torch.cat((labels.long(), remembered_labels), dim=0)
    logits = anchors @ candidates.T / float(temperature)
    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    self_mask[:, : len(anchors)] = torch.eye(
        len(anchors), dtype=torch.bool, device=anchors.device
    )
    positives = labels[:, None].eq(candidate_labels[None, :]) & ~self_mask
    if not torch.all(positives.any(dim=1)):
        raise RuntimeError("Every contrastive anchor needs a positive")
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_denominator = torch.logsumexp(
        logits.masked_fill(self_mask, -torch.inf), dim=1
    )
    log_probability = logits - log_denominator[:, None]
    return -(
        (log_probability * positives).sum(dim=1)
        / positives.sum(dim=1)
    ).mean()


def relational_similarity_loss(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise RuntimeError("Relational KD needs equal Student/Teacher shapes")
    student = F.normalize(student.float(), p=2, dim=-1)
    teacher = F.normalize(teacher.float(), p=2, dim=-1)
    return F.smooth_l1_loss(student @ student.T, teacher @ teacher.T)
