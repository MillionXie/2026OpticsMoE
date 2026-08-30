from __future__ import annotations

import torch
from types import SimpleNamespace

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.training import (
    _retrieval_metrics,
    task_loss,
)


def test_disjoint_gallery_retrieval_is_exact() -> None:
    gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    gallery_labels = torch.tensor([0, 1, 2])
    queries = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    query_labels = torch.tensor([0, 1])
    metrics = _retrieval_metrics(
        queries, query_labels, gallery, gallery_labels
    )
    assert metrics["retrieval_recall_at_1"] == 1.0
    assert metrics["retrieval_map"] == 1.0


def test_leave_one_out_retrieval_excludes_self_from_relevance() -> None:
    embeddings = torch.eye(3)
    labels = torch.tensor([0, 1, 2])
    metrics = _retrieval_metrics(embeddings, labels)
    assert metrics["retrieval_recall_at_1"] == 0.0
    assert metrics["retrieval_map"] == 0.0


def test_three_task_losses_match_formal_output_contracts() -> None:
    classification = torch.randn(2, 101, requires_grad=True)
    loss, _ = task_loss(
        SimpleNamespace(task="caltech101"),
        {"logits": classification},
        {"label": torch.tensor([0, 100])},
    )
    loss.backward()
    assert torch.isfinite(loss)

    segmentation = torch.randn(2, 1, 224, 224, requires_grad=True)
    loss, _ = task_loss(
        SimpleNamespace(task="isic2016"),
        {"logits": segmentation},
        {"mask": torch.zeros_like(segmentation)},
    )
    loss.backward()
    assert torch.isfinite(loss)

    heatmaps = torch.randn(2, 14, 56, 56, requires_grad=True)
    loss, _ = task_loss(
        SimpleNamespace(task="lsp"),
        {"logits": heatmaps},
        {
            "heatmaps": torch.zeros_like(heatmaps),
            "keypoints": torch.full((2, 14, 2), 112.0),
            "visible": torch.ones(2, 14, dtype=torch.bool),
        },
    )
    loss.backward()
    assert torch.isfinite(loss)
