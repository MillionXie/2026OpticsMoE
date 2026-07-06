from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


def summarize_timings(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize teacher feature-extraction wall time only, not optical latency."""

    materialized = list(rows)
    samples = sum(int(row["samples"]) for row in materialized)
    result: dict[str, Any] = {
        "batches": len(materialized),
        "samples": samples,
        "scope": "electronic_teacher_feature_extraction_only",
        "components": {},
    }
    for field in (
        "data_loading_sec",
        "multimodal_preprocess_sec",
        "host_to_device_sec",
        "multimodal_forward_sec",
        "hidden_pooling_sec",
        "end_to_end_sec",
    ):
        values = np.asarray(
            [float(row.get(field, 0.0)) for row in materialized], dtype=np.float64
        )
        if values.size:
            result["components"][field] = {
                "total_sec": float(values.sum()),
                "mean_batch_sec": float(values.mean()),
            }
    return result

