from __future__ import annotations

import csv
import hashlib
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
    development: list[PoseRecord]
    test: list[PoseRecord]
    metadata: dict[str, Any]


def _rank(record: PoseRecord, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{int(seed)}:{record.sample_id}".encode("utf-8"))
    return digest.hexdigest(), record.sample_id


def split_train_development(
    bundle: DatasetBundle,
    *,
    development_count: int = 200,
    seed: int = 42,
) -> PoseProtocolBundle:
    """Seal LSP test and take development only from the first LSP 1000.

    ``prepare_lsp`` already implements the official protocol: all HR-LSPET and
    the first 1000 LSP images are training data; the last 1000 LSP images are
    test data.  We deterministically rank only the first-1000 LSP records and
    move exactly ``development_count`` of them to development.  HR-LSPET never
    enters development and the official last-1000 test is never touched.
    """

    development_count = int(development_count)
    candidates = [record for record in bundle.train if record.source == "lsp"]
    if len(candidates) != 1000:
        raise RuntimeError(
            "The sealed Router protocol requires exactly the canonical first "
            f"1000 LSP records in bundle.train, got {len(candidates)}"
        )
    if not 1 <= development_count < len(candidates):
        raise ValueError("development_count must be in [1,999]")
    selected_ids = {
        record.sample_id
        for record in sorted(candidates, key=lambda record: _rank(record, seed))[
            :development_count
        ]
    }
    development = [
        replace(record, split="development")
        for record in bundle.train
        if record.sample_id in selected_ids
    ]
    train = [
        replace(record, split="train")
        for record in bundle.train
        if record.sample_id not in selected_ids
    ]
    test = [replace(record, split="sealed_test") for record in bundle.test]
    identifiers = [
        {record.sample_id for record in records}
        for records in (train, development, test)
    ]
    if identifiers[0] & identifiers[1] or identifiers[0] & identifiers[2] or identifiers[1] & identifiers[2]:
        raise RuntimeError("Train/development/test sample identifiers overlap")
    if len(test) != 1000 or any(record.source != "lsp" for record in test):
        raise RuntimeError("Sealed test must be exactly the canonical last 1000 LSP images")
    metadata = {
        **bundle.metadata,
        "selection_protocol": (
            "HR-LSPET9428 + 800 LSP train; deterministic 200/first-LSP1000 "
            "development; last-LSP1000 sealed test"
        ),
        "development_selection": "ascending_sha256(f'{seed}:{sample_id}')",
        "development_seed": int(seed),
        "train_samples": len(train),
        "train_lspet": sum(record.source == "lspet" for record in train),
        "train_lsp": sum(record.source == "lsp" for record in train),
        "development_samples": len(development),
        "development_lsp": sum(record.source == "lsp" for record in development),
        "sealed_test_samples": len(test),
        "sealed_test_lsp": sum(record.source == "lsp" for record in test),
        "test_visible_during_training": False,
    }
    return PoseProtocolBundle(train, development, test, metadata)


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
        for record in (*bundle.train, *bundle.development, *bundle.test):
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "split": record.split,
                    "source_index": record.source_index,
                    "image_path": str(record.image_path),
                }
            )


__all__ = ["PoseProtocolBundle", "persist_protocol", "split_train_development"]
