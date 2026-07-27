from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    write_csv,
    write_json,
)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-metrics", required=True)
    parser.add_argument(
        "--baseline-metrics",
        action="append",
        required=True,
        help="Repeat once per electronic baseline metrics/test_metrics.json",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    optical = _read(Path(args.optical_metrics).expanduser().resolve())
    rows = [
        {
            "model": "optical_moe16",
            "type": "optical_student",
            "top1": optical["top1_retrieval_accuracy"],
            "top3": optical["top3_retrieval_accuracy"],
            "mrr": optical["mrr"],
            "parameters": None,
            "trainable_parameters": None,
            "checkpoint": optical.get("checkpoint"),
        }
    ]
    manifests = {optical.get("manifest_sha256")}
    for value in args.baseline_metrics:
        metrics = _read(Path(value).expanduser().resolve())
        model = metrics["model"]
        rows.append(
            {
                "model": model["model_name"],
                "type": "electronic_cnn",
                "top1": metrics["top1_retrieval_accuracy"],
                "top3": metrics["top3_retrieval_accuracy"],
                "mrr": metrics["mrr"],
                "parameters": model["parameters"],
                "trainable_parameters": model["trainable_parameters"],
                "checkpoint": metrics.get("checkpoint"),
            }
        )
        manifests.add(metrics.get("manifest_sha256"))
    if len(manifests) != 1:
        raise RuntimeError(
            f"Runs do not share one fixed dataset manifest digest: {manifests}"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = {
        "manifest_sha256": next(iter(manifests)),
        "ranking_by_top1": sorted(rows, key=lambda row: row["top1"], reverse=True),
    }
    write_json(output_dir / "comparison.json", payload)
    write_csv(
        output_dir / "comparison.csv",
        payload["ranking_by_top1"],
        [
            "model",
            "type",
            "top1",
            "top3",
            "mrr",
            "parameters",
            "trainable_parameters",
            "checkpoint",
        ],
    )
    for row in payload["ranking_by_top1"]:
        print(
            f"{row['model']}: Top-1={row['top1']:.4f} "
            f"Top-3={row['top3']:.4f} MRR={row['mrr']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
