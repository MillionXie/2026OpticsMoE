"""Direction-specific objectives for the SAME cross-modal model architecture.

SKU metadata supplies supervision/relevance only; it is never encoder input.
Actual dataset binding and a training runner are still pending server access.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _positive_mask(query_skus: list[str], candidate_skus: list[str], device) -> torch.Tensor:
    if not query_skus or not candidate_skus:
        raise ValueError("Query and candidate lists must be nonempty")
    if any(not isinstance(key, str) or not key for key in [*query_skus, *candidate_skus]):
        raise ValueError("Relevance requires nonempty exact SKU strings")
    return torch.tensor([[q == c for c in candidate_skus] for q in query_skus],
                        dtype=torch.bool, device=device)


def multi_positive_loss(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """Uniform target mass over all positives, instead of diagonal-only CE.

    Every positive receives attraction. Candidate images of the same SKU are
    not mislabeled as negatives. Each query needs at least one positive.
    """
    if logits.ndim != 2 or positives.shape != logits.shape or positives.dtype != torch.bool:
        raise ValueError("Expected [query,candidate] logits and Boolean relevance mask")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite retrieval logits")
    counts = positives.sum(1)
    if bool((counts == 0).any()):
        raise ValueError("Every query needs at least one positive")
    targets = positives.to(logits.dtype) / counts[:, None]
    return -(F.log_softmax(logits, dim=1) * targets).sum(1).mean()


def cross_modal_loss(
    image_features: torch.Tensor,
    title_features: torch.Tensor,
    image_skus: list[str],
    title_skus: list[str],
    *,
    direction: str,
    temperature: float = 0.07,
) -> torch.Tensor:
    """I2T uses rows of one similarity matrix; T2I uses its transpose.

    Titles without positive images in the current TRAIN minibatch cannot be
    T2I anchors; all titles remain negative candidates for I2T. No TEST labels
    or TEST vectors may be passed here by the eventual training adapter.
    """
    if direction not in {"image_to_text", "text_to_image", "symmetric"}:
        raise ValueError(f"Unknown retrieval direction: {direction}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if (image_features.ndim != 2 or title_features.ndim != 2
            or image_features.shape[1] != title_features.shape[1]
            or len(image_features) != len(image_skus) or len(title_features) != len(title_skus)):
        raise ValueError("Feature shapes must match query/candidate identities")
    logits = F.normalize(image_features, dim=1) @ F.normalize(title_features, dim=1).T / temperature
    positives = _positive_mask(image_skus, title_skus, logits.device)
    if direction == "image_to_text":
        return multi_positive_loss(logits, positives)
    selected = positives.any(0)
    if not bool(selected.any()):
        raise ValueError("No title has positive TRAIN images")
    reverse = multi_positive_loss(logits.T[selected], positives.T[selected])
    return reverse if direction == "text_to_image" else (
        multi_positive_loss(logits, positives) + reverse) / 2


@torch.inference_mode()
def exact_sku_ranking(scores: torch.Tensor, query_skus: list[str], candidate_skus: list[str]) -> dict:
    """Stable full ranking, with first relevant SKU for Hit@K/MRR.

    T2I's 24 same-SKU gallery images are 24 positives for ONE title query, not
    24 independent title queries. Returns full order for per-query evidence.
    """
    if scores.ndim != 2 or scores.shape != (len(query_skus), len(candidate_skus)):
        raise ValueError("Ranking scores/identity shapes differ")
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite ranking scores")
    positives = _positive_mask(query_skus, candidate_skus, scores.device)
    if bool((positives.sum(1) == 0).any()):
        raise ValueError("Every evaluation query needs an enrolled positive")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranked = positives.gather(1, order)
    ranks = torch.arange(1, scores.shape[1] + 1, device=scores.device)
    first = torch.where(ranked, ranks[None], scores.shape[1] + 1).min(1).values
    precision = ranked.float().cumsum(1) / ranks
    average_precision = (precision * ranked).sum(1) / positives.sum(1)
    return {
        "queries": len(query_skus), "candidates": len(candidate_skus),
        "hit_at_1": (first <= 1).float().mean().item(),
        "hit_at_5": (first <= 5).float().mean().item(),
        "hit_at_10": (first <= 10).float().mean().item(),
        "mrr": first.float().reciprocal().mean().item(),
        "map": average_precision.mean().item(),
        "first_positive_rank": first.cpu().tolist(), "order": order.cpu().tolist(),
    }
