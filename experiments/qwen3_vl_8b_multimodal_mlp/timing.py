from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


TIMING_FIELDS = [
    "data_loading_sec",
    "multimodal_preprocess_sec",
    "host_to_device_sec",
    "multimodal_forward_sec",
    "hidden_pooling_sec",
    "mlp_forward_sec",
    "postprocess_sec",
    "pipeline_sec",
    "end_to_end_sec",
]


def summarize_timings(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    samples = sum(int(row["samples"]) for row in materialized)
    result: dict[str, Any] = {"batches": len(materialized), "samples": samples, "components": {}}
    for field in TIMING_FIELDS:
        values = np.asarray([float(row.get(field, 0.0)) for row in materialized], dtype=np.float64)
        if values.size == 0:
            continue
        per_sample_ms = values / np.asarray(
            [max(int(row["samples"]), 1) for row in materialized], dtype=np.float64
        ) * 1000.0
        result["components"][field] = {
            "total_sec": float(values.sum()),
            "mean_batch_sec": float(values.mean()),
            "std_batch_sec": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median_batch_sec": float(np.median(values)),
            "p90_batch_sec": float(np.percentile(values, 90)),
            "p95_batch_sec": float(np.percentile(values, 95)),
            "p99_batch_sec": float(np.percentile(values, 99)),
            "mean_per_sample_ms": float(per_sample_ms.mean()),
        }
    end_to_end = sum(float(row.get("end_to_end_sec", 0.0)) for row in materialized)
    result["throughput_images_per_sec"] = samples / end_to_end if end_to_end else 0.0
    result["measurement_method"] = (
        "CPU wall-clock time with CUDA synchronization at each GPU stage boundary; "
        "warm-up batches are excluded."
    )
    result["tokenizer_included"] = True
    result["forward_scope"] = "full_qwen3_vl_vision_language_forward"
    return result
