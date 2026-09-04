from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
COUNTS = (4, 9, 16, 25, 36, 49)
EXPECTED_TEST_VIDEOS = 558


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(result: Any) -> float:
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def recompute(rows: list[dict[str, str]]) -> dict[str, float]:
    target = np.asarray([float(row["temporal_target"]) for row in rows])
    prediction = np.asarray([float(row["temporal_prediction"]) for row in rows])
    difference = prediction - target
    return {
        "srcc": scalar(spearmanr(target, prediction)),
        "krcc": scalar(kendalltau(target, prediction)),
        "plcc": scalar(pearsonr(target, prediction)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
    }


def validate() -> dict[str, Any]:
    timing = load_json(ROOT / "results" / "timing_summary_all.json")
    performance = load_json(ROOT / "performance_results" / "comparison.json")
    identity = load_json(ROOT / "performance_results" / "formal_runtime_identity.json")
    device = identity["runtime"]["cuda_device_0"]
    if "5090" not in str(device):
        raise RuntimeError(f"Formal performance device is not RTX 5090 D: {device}")

    timing_sources = {
        4: "timing_summary.json",
        9: "timing_summary.json",
        16: "timing_summary.json",
        25: "timing_summary_25.json",
        36: "timing_summary_36.json",
        49: "timing_summary_49.json",
    }
    for source in sorted(set(timing_sources.values())):
        recorded = load_json(ROOT / "results" / source).get("gpu")
        if recorded != "NVIDIA GeForce RTX 5090 D":
            raise RuntimeError(f"Unexpected timing device in {source}: {recorded}")

    sample_ids: set[str] | None = None
    per_count: dict[str, Any] = {}
    for count in COUNTS:
        timed = timing["results"][str(count)]
        if int(timed["unique_videos"]) != EXPECTED_TEST_VIDEOS:
            raise RuntimeError(f"Timing {count}f does not contain 558 unique videos")
        training = performance["results"][str(count)]["training"]
        reported = training["metrics"]["temporal"]
        prediction_path = (
            ROOT
            / "performance_results"
            / f"frames{count}"
            / "linear_head"
            / "test_predictions.csv"
        )
        with prediction_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [row["sample_id"] for row in rows]
        if len(rows) != EXPECTED_TEST_VIDEOS or len(set(ids)) != EXPECTED_TEST_VIDEOS:
            raise RuntimeError(f"Performance {count}f predictions are not 558 unique videos")
        current_ids = set(ids)
        if sample_ids is None:
            sample_ids = current_ids
        elif current_ids != sample_ids:
            raise RuntimeError(f"Performance test IDs differ at {count}f")
        calculated = recompute(rows)
        errors = {name: abs(calculated[name] - float(reported[name])) for name in calculated}
        if not all(math.isfinite(value) and value <= 1.0e-6 for value in errors.values()):
            raise RuntimeError(f"Metric mismatch at {count}f: {errors}")
        if int(training["epochs_completed"]) != 50:
            raise RuntimeError(f"Performance {count}f did not complete 50 epochs")
        per_count[str(count)] = {
            "timing_unique_videos": int(timed["unique_videos"]),
            "performance_unique_videos": len(set(ids)),
            "best_epoch": int(training["best_epoch"]),
            "temporal_metrics_recomputed": calculated,
            "maximum_metric_absolute_error": max(errors.values()),
        }
    return {
        "schema_version": 1,
        "status": "passed",
        "formal_device": device,
        "frame_counts": list(COUNTS),
        "same_fixed_test_ids_across_all_counts": True,
        "timing_and_performance_test_videos_per_count": EXPECTED_TEST_VIDEOS,
        "performance_epochs_per_count": 50,
        "per_count": per_count,
    }


def main() -> int:
    report = validate()
    output = ROOT / "FORMAL_QA.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
