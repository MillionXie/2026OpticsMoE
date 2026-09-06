from __future__ import annotations

import numpy as np

from LightGenV2.tasks.t08_abo_image_text_retrieval.baseline_5090d import (
    DOCUMENT_INSTRUCTION,
    EMBEDDING_DIM,
    IMAGE_PIXELS,
    QUERY_INSTRUCTION,
    Query,
    _balanced_timing_queries,
    _metrics,
)


def test_frozen_baseline_contract_is_full_embedding_and_fixed_224() -> None:
    assert EMBEDDING_DIM == 2048
    assert IMAGE_PIXELS == 224 * 224
    assert "product title" in QUERY_INSTRUCTION
    assert DOCUMENT_INSTRUCTION == "Represent the user's input."


def test_single_relevant_candidate_metrics() -> None:
    result = _metrics(np.asarray([1, 2, 5, 6, 10, 11]))
    assert result["recall_at_1"] == 1 / 6
    assert result["recall_at_5"] == 3 / 6
    assert result["recall_at_10"] == 5 / 6
    assert result["median_rank"] == 5.5
    assert np.isclose(result["mrr"], np.mean(1 / np.asarray([1, 2, 5, 6, 10, 11])))


def test_timing_subset_round_robins_across_labels(tmp_path) -> None:
    queries = [
        Query(f"{label}-{item}", str(label), label, tmp_path / f"{label}-{item}.jpg")
        for label in range(3)
        for item in range(4)
    ]
    selected = _balanced_timing_queries(queries, 6)
    assert [query.label for query in selected] == [0, 1, 2, 0, 1, 2]
    assert len({query.sample_id for query in selected}) == 6
