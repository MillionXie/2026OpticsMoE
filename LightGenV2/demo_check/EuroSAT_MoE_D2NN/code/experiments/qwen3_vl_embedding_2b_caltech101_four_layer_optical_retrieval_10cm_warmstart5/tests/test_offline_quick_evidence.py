from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.tests.test_offline_quick_finetune import (
    _make_session,
)

from ..offline_quick_finetune import (
    finetune_offline_quick,
    load_offline_quick_data,
)


WARMSTART5_ARCHITECTURE = "vision2_language2_moe4_10cm_warmstart5_stage_b_v1"


def test_warmstart5_runner_rejects_10_percent_checkpoint_contract(
    tmp_path: Path,
) -> None:
    session, _ = _make_session(tmp_path)
    with pytest.raises(RuntimeError, match="Warmstart5 offline runner requires"):
        load_offline_quick_data(session)


def test_finetune_writes_paired_pre_post_query_evidence(tmp_path: Path) -> None:
    session, manifest_rows = _make_session(
        tmp_path,
        architecture=WARMSTART5_ARCHITECTURE,
    )
    output = tmp_path / "offline_results"
    report = finetune_offline_quick(
        session_dir=session,
        output_dir=output,
        device_name="cpu",
        epochs=1,
    )

    required = (
        "metrics.json",
        "pre_finetune_metrics.json",
        "post_finetune_metrics.json",
        "predictions.csv",
        "best_offline_tail_state.pt",
        "train_log.csv",
        "ccd_inventory.json",
    )
    assert all((output / name).is_file() for name in required)

    pre = json.loads((output / "pre_finetune_metrics.json").read_text("utf-8"))
    post = json.loads((output / "post_finetune_metrics.json").read_text("utf-8"))
    aggregate = json.loads((output / "metrics.json").read_text("utf-8"))
    assert pre["evaluation_point"] == "before_offline_finetune"
    assert post["evaluation_point"] == "after_train_loss_selected_offline_finetune"
    assert pre["gallery_or_test_used_for_epoch_selection"] is False
    assert post["gallery_or_test_used_for_epoch_selection"] is False
    assert aggregate["gallery_or_test_used_for_epoch_selection"] is False
    assert post["top1_retrieval_accuracy"] == pytest.approx(
        aggregate["top1_retrieval_accuracy"]
    )
    assert report["paired_test_queries"] == pre["query_count"] == 2
    assert report["query_prediction_rows"] == 4

    with (output / "predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * pre["query_count"]
    assert {
        "system",
        "sample_id",
        "key",
        "true_label",
        "predicted_label",
        "top1_correct",
        "similarity_margin",
        "rank",
    }.issubset(rows[0])

    expected_test_keys = {
        str(row["key"]) for row in manifest_rows if row["split"] == "test"
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["key"]].append(row)
        assert row["split"] == "test"
        assert row["top1_correct"] in {"True", "False"}
        assert int(row["rank"]) >= 1
        assert math.isfinite(float(row["similarity_margin"]))
    assert set(grouped) == expected_test_keys
    for key, pair in grouped.items():
        assert [row["system"] for row in pair] == [
            "quick210_0_pre_finetune",
            "quick210_1_post_finetune",
        ]
        assert all(row["key"] == key for row in pair)
