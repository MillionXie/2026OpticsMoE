from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.result_report import (
    generate_report,
)


CLASSES = [
    "airplanes",
    "Motorbikes",
    "Faces",
    "Leopards",
    "accordion",
    "grand_piano",
    "scorpion",
    "sunflower",
    "watch",
    "yin_yang",
]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _metrics(top1: float) -> dict:
    diagonal = int(round(top1 * 10))
    matrix = []
    for index in range(10):
        row = [0] * 10
        row[index] = diagonal
        row[(index + 1) % 10] = 10 - diagonal
        matrix.append(row)
    return {
        "system": "test_fixture",
        "query_count": 100,
        "gallery_image_count": 10,
        "sku_count": 10,
        "top1_retrieval_accuracy": top1,
        "top3_retrieval_accuracy": min(1.0, top1 + 0.1),
        "mrr": min(1.0, top1 + 0.05),
        "per_sku": {
            name: {
                "query_count": 10,
                "top1_accuracy": top1,
                "top3_accuracy": min(1.0, top1 + 0.1),
            }
            for name in CLASSES
        },
        "confusion_matrix": matrix,
    }


def _baseline(root: Path) -> Path:
    path = root / "reference/training_evidence/stage_b/metrics/evaluation_summary.json"
    _write_json(path, {"teacher": None, "student": _metrics(0.81)})
    return path


def test_baseline_only_report_marks_hardware_unavailable(tmp_path: Path) -> None:
    _baseline(tmp_path)
    output = tmp_path / "report"
    report = generate_report(root=tmp_path, output_dir=output, formats=("svg",))

    assert report["records"][0]["top1"] == 0.81
    assert report["availability"]["simulation_baseline"] is True
    assert report["availability"]["quick_language_global"] is False
    assert report["availability"]["ccd_qc"] is False
    by_name = {item["figure"]: item["status"] for item in report["figures"]}
    assert by_name["fig01_overall_metrics"] == "available"
    assert by_name["fig03_confusion_matrix"] == "available"
    assert by_name["fig04_stage_progression"] == "unavailable"
    assert by_name["fig05_ccd_quality_control"] == "unavailable"
    assert (output / "fig07_overview.svg").is_file()
    assert (output / "source_data/overall_metrics.csv").is_file()
    assert (output / "figure_manifest.json").is_file()
    assert "<text" in (output / "fig01_overall_metrics.svg").read_text(encoding="utf-8")


def test_real_schema_discovery_with_synthetic_test_fixture(tmp_path: Path) -> None:
    _baseline(tmp_path)
    session = tmp_path / "payload/quick210"
    stage = session / "04_language_global"
    metrics_path = stage / "offline_results/metrics.json"
    hardware = _metrics(0.77)
    hardware.update({"stage": "language_global", "measured_stages": ["language_global"]})
    _write_json(metrics_path, hardware)

    ccd_dir = stage / "ccd_captured"
    ccd_dir.mkdir(parents=True)
    for index, level in enumerate((32, 96, 160)):
        frame = np.full((478, 478), level, dtype=np.uint8)
        frame[0, 0] = 0
        frame[-1, -1] = 255
        Image.fromarray(frame, mode="L").save(ccd_dir / f"sample_{index}.png")

    predictions = stage / "query_predictions.csv"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("system", "sample_id", "true_sku", "predicted_sku", "top1_correct", "similarity_margin"),
        )
        writer.writeheader()
        writer.writerows(
            [
                {"system": "simulation", "sample_id": "a", "true_sku": "Faces", "predicted_sku": "Faces", "top1_correct": "true", "similarity_margin": "0.2"},
                {"system": "hardware", "sample_id": "a", "true_sku": "Faces", "predicted_sku": "Faces", "top1_correct": "true", "similarity_margin": "0.1"},
                {"system": "simulation", "sample_id": "b", "true_sku": "watch", "predicted_sku": "Faces", "top1_correct": "false", "similarity_margin": "-0.1"},
                {"system": "hardware", "sample_id": "b", "true_sku": "watch", "predicted_sku": "watch", "top1_correct": "true", "similarity_margin": "0.05"},
            ]
        )

    output = tmp_path / "report"
    report = generate_report(
        root=tmp_path,
        output_dir=output,
        formats=("svg",),
    )

    assert report["availability"]["quick_language_global"] is True
    assert report["availability"]["ccd_qc"] is True
    assert report["availability"]["paired_predictions"] is True
    assert report["availability"]["similarity_margin_pairs"] is True
    assert len(report["ccd_qc_summary"]) == 1
    by_name = {item["figure"]: item["status"] for item in report["figures"]}
    # A quick-language-global session is not a four-stage progression.  Every
    # evidence-backed panel is available, while the stage-progression panel
    # must remain explicitly unavailable rather than being imputed.
    assert by_name["fig04_stage_progression"] == "unavailable"
    assert all(
        status == "available"
        for name, status in by_name.items()
        if name != "fig04_stage_progression"
    )
    assert (output / "fig05_ccd_quality_control.svg").is_file()
    assert (output / "fig06_paired_query_changes.svg").is_file()
    inventory = report["input_source_inventory"]
    assert any(item["path"].endswith("offline_results/metrics.json") for item in inventory)
    assert sum(item["path"].endswith(".png") for item in inventory) == 3


def test_rerun_removes_stale_owned_hardware_figure(tmp_path: Path) -> None:
    _baseline(tmp_path)
    output = tmp_path / "report"
    generate_report(root=tmp_path, output_dir=output, formats=("svg",))
    stale = output / "fig05_ccd_quality_control.svg"
    stale.write_text("stale hardware output", encoding="utf-8")

    report = generate_report(root=tmp_path, output_dir=output, formats=("svg",))

    assert report["availability"]["ccd_qc"] is False
    assert not stale.exists()


def test_default_quick_discovery_keeps_pre_and_post_metrics_distinct(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path)
    result_dir = tmp_path / "payload/quick210/04_language_global/offline_results"
    pre = _metrics(0.62)
    pre.update(
        {
            "system": "quick210_0_pre_finetune",
            "evaluation_point": "before_offline_finetune",
        }
    )
    post = _metrics(0.76)
    post.update(
        {
            "system": "quick210_1_post_finetune",
            "evaluation_point": "after_train_loss_selected_offline_finetune",
        }
    )
    _write_json(result_dir / "pre_finetune_metrics.json", pre)
    _write_json(result_dir / "post_finetune_metrics.json", post)

    report = generate_report(
        root=tmp_path,
        output_dir=tmp_path / "report",
        formats=("svg",),
    )

    by_id = {row["record_id"]: row for row in report["records"]}
    assert by_id["quick_language_global_pre"]["top1"] == 0.62
    assert by_id["quick_language_global_post"]["top1"] == 0.76
    assert report["availability"]["quick_language_global"] is True
    paths = {item["path"] for item in report["input_source_inventory"]}
    assert any(path.endswith("pre_finetune_metrics.json") for path in paths)
    assert any(path.endswith("post_finetune_metrics.json") for path in paths)
