from __future__ import annotations

import numpy as np

from LightGenV2.tasks.t07_abo_image_retrieval.baseline_a100 import (
    EMBEDDING_DIM,
    EXPECTED_CATEGORIES,
    IMAGE_PIXELS,
    Sample,
    _balanced_timing_queries,
    _ranking_metrics,
)


def test_frozen_embedding_contract_is_full_qwen_and_fixed_224() -> None:
    assert EMBEDDING_DIM == 2048
    assert IMAGE_PIXELS == 224 * 224
    assert EXPECTED_CATEGORIES == 10


def test_ranking_metrics_use_twelve_relevant_gallery_products() -> None:
    relevant = np.zeros((2, 120), dtype=bool)
    relevant[0, :12] = True
    relevant[1, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] = True
    metrics = _ranking_metrics(relevant)
    assert metrics["precision_at_1"] == 1.0
    assert metrics["precision_at_5"] == 0.8
    assert metrics["positive_recall_at_10"] == 0.625
    assert metrics["r_at_10"] == 1.0
    assert metrics["relevant_candidates_per_query"] == 12.0
    assert metrics["query_count"] == 2
    assert metrics["candidate_count"] == 120


def test_timing_subset_round_robins_categories() -> None:
    samples = [
        Sample(str(i), f"p{i}", i % 10, f"c{i % 10}", "test", None)  # type: ignore[arg-type]
        for i in range(40)
    ]
    selected = _balanced_timing_queries(samples, 20)
    assert len(selected) == 20
    assert [sample.category_id for sample in selected[:10]] == list(range(10))
    assert [sample.category_id for sample in selected[10:]] == list(range(10))
