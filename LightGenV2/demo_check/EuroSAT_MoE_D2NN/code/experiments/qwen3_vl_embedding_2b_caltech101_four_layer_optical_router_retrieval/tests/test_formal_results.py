from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval import (
    formal_results as subject,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_run(root: Path, spec: subject.RunSpec, top1: float) -> None:
    run = root / spec.directory
    run.mkdir(parents=True)
    checkpoint_name = (
        "converted_warmstart5_initialization_checkpoint.pt"
        if spec.variant == "legacy"
        else "ema_best_train_loss_checkpoint.pt"
    )
    checkpoint = run / checkpoint_name
    checkpoint.write_bytes(b"fixture")
    query_count = 20
    correct_count = round(top1 * query_count)
    top1 = correct_count / query_count
    per_class = {
        "a": {"query_count": 10, "top1_accuracy": top1, "top3_accuracy": 1.0},
        "b": {"query_count": 10, "top1_accuracy": top1, "top3_accuracy": 1.0},
    }
    metrics = {
        "system": "student",
        "query_count": query_count,
        "gallery_image_count": 2,
        "sku_count": 2,
        "top1_retrieval_accuracy": top1,
        "top3_retrieval_accuracy": 1.0,
        "mrr": (1.0 + top1) / 2.0,
        "per_sku": per_class,
        "manifest_sha256": "a" * 64,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": 2,
        "selection_biased": False,
    }
    (run / "student_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if spec.variant != "legacy":
        _write_csv(
            run / "train_log.csv",
            [
                {"epoch": 1, "total_loss": 2.0, "train_top1": 0.5, "epoch_time_sec": 1.0, "test_top1": ""},
                {"epoch": 2, "total_loss": 1.0, "train_top1": 0.8, "epoch_time_sec": 1.0, "test_top1": ""},
            ],
        )
    predictions = []
    for index in range(query_count):
        correct = index < correct_count
        predictions.append(
            {
                "system": "student",
                "sample_id": f"q{index}",
                "true_sku": "a" if index < 10 else "b",
                "predicted_sku": ("a" if index < 10 else "b") if correct else "wrong",
                "top1_correct": correct,
            }
        )
    _write_csv(run / "retrieval_results.csv", predictions)


def _fake_checkpoint(path: Path) -> dict[str, object]:
    legacy = "converted_warmstart5" in path.name
    return {
        "checkpoint_version": 2,
        "epoch": 2,
        "train_loss": 1.0,
        "metadata": {
            "weight_variant": "ema",
            "selection_criterion": "minimum_training_total_loss",
            "test_metrics_used_for_selection": False,
            "optical_architecture": "fixture",
            "embedding_dim": 64,
            "detector_dim": 192,
        },
    }


def test_pending_report_is_friendly(tmp_path: Path) -> None:
    output = tmp_path / "out"
    markdown = tmp_path / "FORMAL_RESULTS.md"
    report = subject.generate_report(
        runs_dir=tmp_path / "missing",
        output_dir=output,
        markdown_path=markdown,
        bootstrap_samples=100,
    )
    assert report["status"] == "pending"
    assert all(run["status"] == "pending" for run in report["runs"])
    assert (output / "aggregate_results.json").is_file()
    assert "PENDING" in markdown.read_text(encoding="utf-8")


def test_complete_report_aggregates_and_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    values = {
        "legacy": [0.75],
        "E1": [0.70, 0.75, 0.80],
        "E2": [0.80, 0.85, 0.90],
        "E4": [0.75, 0.80, 0.85],
        "O2": [0.80, 0.80, 0.85],
    }
    offsets = {variant: 0 for variant in values}
    for spec in subject.RUN_SPECS:
        value = values[spec.variant][offsets[spec.variant]]
        offsets[spec.variant] += 1
        _make_run(runs, spec, value)
    monkeypatch.setattr(subject, "_checkpoint_payload", _fake_checkpoint)
    output = tmp_path / "formal"
    markdown = tmp_path / "FORMAL_RESULTS.md"
    report = subject.generate_report(
        runs_dir=runs,
        output_dir=output,
        markdown_path=markdown,
        bootstrap_samples=500,
        bootstrap_seed=7,
    )
    assert report["status"] == "complete"
    e2 = report["variants"]["E2"]["metrics"]["top1_retrieval_accuracy"]
    assert e2["n"] == 3
    assert e2["mean"] == pytest.approx(0.85)
    assert e2["sample_std"] == pytest.approx(0.05)
    assert e2["absolute_delta_vs_legacy"] == pytest.approx(0.10)
    assert len(report["paired_mcnemar"]) == 21
    assert all(row["query_count"] == 20 for row in report["paired_mcnemar"])
    assert "COMPLETE" in markdown.read_text(encoding="utf-8")
    assert (output / "aggregate_metrics.csv").is_file()
    assert (output / "per_class_metrics.csv").is_file()
    assert (output / "paired_mcnemar.csv").is_file()
    assert (output / "run_inventory.csv").is_file()


def test_training_log_with_test_observation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = next(spec for spec in subject.RUN_SPECS if spec.variant == "E1")
    _make_run(tmp_path, spec, 0.8)
    log = tmp_path / spec.directory / "train_log.csv"
    rows = subject._read_csv(log)
    rows[0]["test_top1"] = "0.8"
    _write_csv(log, rows)
    monkeypatch.setattr(subject, "_checkpoint_payload", _fake_checkpoint)
    record = subject._inspect_run(spec, tmp_path)
    assert record["status"] == "error"
    assert "Finite test metrics" in record["errors"][0]


def test_exact_mcnemar_known_cases() -> None:
    assert subject._exact_mcnemar_p(0, 0) == 1.0
    assert subject._exact_mcnemar_p(0, 5) == pytest.approx(0.0625)
    assert subject._exact_mcnemar_p(2, 2) == 1.0
