from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .datasets import ABOBundle, ABOSample
from .losses import CrossBatchMemory, supervised_contrastive_loss_with_memory
from .teacher_cache import TeacherEmbeddingStore


class NormalizedTeacherAdapter(nn.Module):
    """Offline product-specific metric head; never part of optical deployment."""

    def __init__(self, input_dim: int = 2048, output_dim: int = 224) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(input_dim))
        self.projection = nn.Linear(int(input_dim), int(output_dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.projection(self.norm(hidden.float()))
        if not torch.isfinite(raw).all():
            raise RuntimeError("Teacher adapter produced NaN/Inf")
        return F.normalize(raw, p=2, dim=-1)


class CosineIdentityHead(nn.Module):
    """Training-only normalized proxy classifier with an additive cosine margin."""

    def __init__(
        self,
        embedding_dim: int,
        class_count: int,
        *,
        scale: float = 30.0,
        margin: float = 0.10,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(class_count, embedding_dim))
        nn.init.normal_(self.weight, std=0.01)
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(
        self, embedding: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        cosine = F.normalize(embedding.float(), p=2, dim=-1) @ F.normalize(
            self.weight.float(), p=2, dim=-1
        ).T
        if labels is not None and self.margin:
            cosine = cosine - F.one_hot(
                labels.long(), num_classes=self.weight.shape[0]
            ).to(cosine.dtype) * self.margin
        return cosine * self.scale


class _PKIndexSampler:
    def __init__(
        self,
        samples: Sequence[ABOSample],
        p: int,
        k: int,
        seed: int,
    ) -> None:
        self.p = int(p)
        self.k = int(k)
        self.seed = int(seed)
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            grouped[int(sample.item_index)].append(index)
        self.grouped = dict(grouped)
        self.classes = sorted(self.grouped)
        self.batch_count = max(1, math.ceil(len(samples) / (self.p * self.k)))

    def batches(self, epoch: int) -> Iterator[list[int]]:
        rng = random.Random(self.seed + int(epoch) * 1_000_003)
        for _ in range(self.batch_count):
            classes = (
                rng.sample(self.classes, self.p)
                if len(self.classes) >= self.p
                else [rng.choice(self.classes) for _ in range(self.p)]
            )
            batch: list[int] = []
            for class_index in classes:
                pool = self.grouped[class_index]
                if len(pool) >= self.k:
                    batch.extend(rng.sample(pool, self.k))
                else:
                    batch.extend(rng.choice(pool) for _ in range(self.k))
            yield batch


def train_teacher_adapter(
    bundle: ABOBundle,
    settings: Any,
    device: torch.device,
) -> Path:
    store = TeacherEmbeddingStore(bundle, settings, apply_adapter=False)
    if store.raw_hidden is None:
        raise RuntimeError(
            "Raw 2048-D Teacher hidden is unavailable; rebuild cache with "
            "teacher_embedding_mode=adapted_head"
        )
    samples = bundle.stage2_train.samples
    sampler = _PKIndexSampler(
        samples,
        settings.teacher_adapter_pk_items,
        settings.teacher_adapter_pk_images,
        settings.random_seed + 701,
    )
    adapter = NormalizedTeacherAdapter(
        store.raw_hidden.shape[1], settings.embedding_dim
    ).to(device)
    classifier = CosineIdentityHead(
        settings.embedding_dim,
        len(bundle.stage2_item_ids),
        scale=settings.identity_scale,
        margin=settings.identity_margin,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": adapter.parameters(),
                "lr": settings.teacher_adapter_learning_rate,
            },
            {
                "params": classifier.parameters(),
                "lr": settings.head_learning_rate,
            },
        ],
        weight_decay=settings.teacher_adapter_weight_decay,
    )
    memory = CrossBatchMemory(
        settings.contrastive_memory_size, settings.embedding_dim
    ).to(device)
    best = math.inf
    history: list[dict[str, Any]] = []
    checkpoint = settings.teacher_adapter_checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, settings.teacher_adapter_epochs + 1):
        adapter.train()
        classifier.train()
        totals = defaultdict(float)
        for batch_indices in sampler.batches(epoch):
            batch_samples = [samples[index] for index in batch_indices]
            image_ids = [sample.image_id for sample in batch_samples]
            labels = torch.tensor(
                [sample.item_index for sample in batch_samples],
                dtype=torch.long,
                device=device,
            )
            hidden = store.get_raw(image_ids, device)
            optimizer.zero_grad(set_to_none=True)
            embedding = adapter(hidden)
            supcon = supervised_contrastive_loss_with_memory(
                embedding, labels, settings.temperature, memory
            )
            logits = classifier(embedding, labels)
            identity = F.cross_entropy(logits, labels)
            total = (
                settings.teacher_adapter_supcon_weight * supcon
                + settings.teacher_adapter_identity_weight * identity
            )
            total.backward()
            if settings.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    list(adapter.parameters()) + list(classifier.parameters()),
                    settings.gradient_clip_norm,
                )
            optimizer.step()
            memory.enqueue(embedding, labels)
            count = len(labels)
            totals["samples"] += count
            totals["total"] += float(total.detach()) * count
            totals["supcon"] += float(supcon.detach()) * count
            totals["identity"] += float(identity.detach()) * count
            totals["correct"] += float(logits.argmax(dim=-1).eq(labels).sum())
        denominator = totals["samples"]
        row = {
            "epoch": epoch,
            "total_loss": totals["total"] / denominator,
            "supcon_loss": totals["supcon"] / denominator,
            "identity_loss": totals["identity"] / denominator,
            "identity_accuracy": totals["correct"] / denominator,
            "samples": int(denominator),
            "memory_size": int(memory.size),
        }
        history.append(row)
        _write_history(
            settings.output_dir / "metrics" / "teacher_adapter_history.csv",
            history,
        )
        payload = {
            "format_version": 1,
            "epoch": epoch,
            "training_loss": row["total_loss"],
            "manifest_sha256": bundle.manifest_digest,
            "teacher_adapter": adapter.state_dict(),
            "training_only_identity_head": classifier.state_dict(),
            "input_dim": store.raw_hidden.shape[1],
            "embedding_dim": settings.embedding_dim,
        }
        torch.save(
            payload,
            settings.output_dir / "checkpoints" / "teacher_adapter_last.pt",
        )
        if row["total_loss"] < best:
            best = row["total_loss"]
            torch.save(payload, checkpoint)
        print(
            f"[teacher_adapter] epoch={epoch:03d} "
            f"loss={row['total_loss']:.5f} "
            f"supcon={row['supcon_loss']:.5f} "
            f"id_acc={row['identity_accuracy']:.4f} best={best:.5f}"
        )
    return checkpoint


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
