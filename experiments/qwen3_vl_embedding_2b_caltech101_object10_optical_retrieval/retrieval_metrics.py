from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .prepare_caltech101_retrieval_subset import Caltech101Sample


@dataclass(frozen=True)
class RetrievalEvaluation:
    metrics: dict[str, Any]
    rows: list[dict[str, Any]]
    confusion: torch.Tensor


def evaluate_embeddings(
    query_embeddings: torch.Tensor,
    query_samples: Sequence[Caltech101Sample],
    gallery_embeddings: torch.Tensor,
    gallery_samples: Sequence[Caltech101Sample],
    class_names: Sequence[str],
    aggregation: str,
    *,
    system_name: str,
) -> RetrievalEvaluation:
    query = _normalized(query_embeddings, "query")
    gallery = _normalized(gallery_embeddings, "gallery")
    if len(query) != len(query_samples) or len(gallery) != len(gallery_samples):
        raise RuntimeError("Embedding counts do not match retrieval sample metadata")
    class_count = len(class_names)
    gallery_labels = torch.tensor(
        [sample.class_index for sample in gallery_samples], dtype=torch.long
    )
    missing = [
        class_names[index]
        for index in range(class_count)
        if not torch.any(gallery_labels.eq(index))
    ]
    if missing:
        raise RuntimeError(f"Gallery is missing class prototypes: {missing}")
    individual = query @ gallery.T
    class_scores = []
    for class_index in range(class_count):
        selected = gallery_labels.eq(class_index)
        if aggregation == "mean_prototype":
            prototype = F.normalize(gallery[selected].mean(dim=0), dim=0)
            class_scores.append(query @ prototype)
        elif aggregation == "max_similarity":
            class_scores.append(individual[:, selected].amax(dim=1))
        else:
            raise ValueError(f"Unsupported gallery aggregation {aggregation!r}")
    scores = torch.stack(class_scores, dim=1)
    ranking = scores.argsort(dim=1, descending=True)
    truth = torch.tensor([sample.class_index for sample in query_samples], dtype=torch.long)
    top1 = ranking[:, 0]
    top3 = ranking[:, : min(3, class_count)]
    correct_top1 = top1.eq(truth)
    correct_top3 = top3.eq(truth[:, None]).any(dim=1)
    reciprocal_ranks = []
    confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(query_samples):
        order = ranking[index]
        rank = int(torch.nonzero(order.eq(truth[index]), as_tuple=False)[0, 0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        confusion[int(truth[index]), int(top1[index])] += 1
        true_similarity = float(scores[index, int(truth[index])])
        wrong = scores[index].clone()
        wrong[int(truth[index])] = -torch.inf
        best_wrong_similarity = float(wrong.max())
        row: dict[str, Any] = {
            "system": system_name,
            "sample_id": sample.sample_id,
            "query_image_path": str(sample.image_path),
            "true_class_index": sample.class_index,
            "true_class": sample.class_name,
            "predicted_class_index": int(top1[index]),
            "predicted_class": class_names[int(top1[index])],
            "top1_correct": bool(correct_top1[index]),
            "top3_correct": bool(correct_top3[index]),
            "true_class_rank": rank,
            "reciprocal_rank": 1.0 / rank,
            "correct_class_similarity": true_similarity,
            "max_wrong_class_similarity": best_wrong_similarity,
            "similarity_margin": true_similarity - best_wrong_similarity,
        }
        for position in range(min(3, class_count)):
            class_index = int(order[position])
            candidate_gallery = torch.nonzero(
                gallery_labels.eq(class_index), as_tuple=False
            ).flatten()
            local_best = candidate_gallery[
                individual[index, candidate_gallery].argmax()
            ]
            row[f"top{position + 1}_class"] = class_names[class_index]
            row[f"top{position + 1}_similarity"] = float(scores[index, class_index])
            row[f"top{position + 1}_gallery_path"] = str(
                gallery_samples[int(local_best)].image_path
            )
        rows.append(row)
    per_class: dict[str, Any] = {}
    for class_index, name in enumerate(class_names):
        selected = truth.eq(class_index)
        per_class[name] = {
            "query_count": int(selected.sum()),
            "top1_accuracy": float(correct_top1[selected].float().mean()),
            "top3_accuracy": float(correct_top3[selected].float().mean()),
        }
    metrics = {
        "system": system_name,
        "gallery_aggregation": aggregation,
        "query_count": len(query_samples),
        "gallery_image_count": len(gallery_samples),
        "class_count": class_count,
        "top1_retrieval_accuracy": float(correct_top1.float().mean()),
        "top3_retrieval_accuracy": float(correct_top3.float().mean()),
        "mrr": float(torch.tensor(reciprocal_ranks).mean()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    return RetrievalEvaluation(metrics, rows, confusion)


def _normalized(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim != 2:
        raise RuntimeError(f"{name} embeddings must be 2-D, got {tuple(values.shape)}")
    values = values.detach().cpu().float()
    norms = values.norm(dim=-1)
    if not torch.isfinite(values).all() or torch.any(norms <= 1e-12):
        raise RuntimeError(f"{name} embeddings contain NaN/Inf or zero norms")
    return F.normalize(values, dim=-1)
