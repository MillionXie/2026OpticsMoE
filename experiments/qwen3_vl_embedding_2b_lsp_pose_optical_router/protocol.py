from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import (
    DatasetBundle,
    PoseRecord,
)


@dataclass(frozen=True)
class PoseProtocolBundle:
    train: list[PoseRecord]
    test: list[PoseRecord]
    metadata: dict[str, Any]


def build_periodic_test_protocol(bundle: DatasetBundle) -> PoseProtocolBundle:
    """Keep the canonical LSP train/test split without a development split.

    The existing loader defines the project protocol as 9,428 HR-LSPET plus
    the first 1,000 LSP samples for training, and the final 1,000 LSP samples
    for test. This function validates and records that split verbatim. Test is
    intentionally visible during training in the lab-requested experiment: it
    is evaluated periodically and is also the checkpoint-selection split.
    """

    train = [replace(record, split="train") for record in bundle.train]
    test = [replace(record, split="periodic_test") for record in bundle.test]
    train_ids = {record.sample_id for record in train}
    test_ids = {record.sample_id for record in test}
    if train_ids & test_ids:
        raise RuntimeError("Train/test sample identifiers overlap")
    if len(train) != 10428:
        raise RuntimeError(
            "Periodic-test protocol requires 9,428 HR-LSPET + 1,000 LSP "
            f"training samples, got {len(train)}"
        )
    if sum(record.source == "lspet" for record in train) != 9428:
        raise RuntimeError("Training must contain exactly 9,428 HR-LSPET samples")
    if sum(record.source == "lsp" for record in train) != 1000:
        raise RuntimeError("Training must contain exactly the first 1,000 LSP samples")
    if len(test) != 1000 or any(record.source != "lsp" for record in test):
        raise RuntimeError("Test must be exactly the final 1,000 LSP samples")
    metadata = {
        **bundle.metadata,
        "selection_protocol": (
            "HR-LSPET9428 + first-LSP1000 train; last-LSP1000 periodic test; "
            "no validation split"
        ),
        "train_samples": len(train),
        "train_lspet": sum(record.source == "lspet" for record in train),
        "train_lsp": sum(record.source == "lsp" for record in train),
        "validation_samples": 0,
        "test_samples": len(test),
        "test_lsp": sum(record.source == "lsp" for record in test),
        "test_visible_during_training": True,
        "test_used_for_checkpoint_selection": True,
    }
    return PoseProtocolBundle(train, test, metadata)


def persist_protocol(bundle: PoseProtocolBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pose_protocol.json").write_text(
        json.dumps(bundle.metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "pose_protocol_split.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "source", "split", "source_index", "image_path"),
        )
        writer.writeheader()
        for record in (*bundle.train, *bundle.test):
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "split": record.split,
                    "source_index": record.source_index,
                    "image_path": str(record.image_path),
                }
            )


__all__ = ["PoseProtocolBundle", "build_periodic_test_protocol", "persist_protocol"]
