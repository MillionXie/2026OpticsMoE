from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from ..paper_evaluation import PredictionRunSpec, evaluate_prediction_runs


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_run(
    root: Path,
    *,
    profile: str,
    samples_per_class: int,
    error_period: int,
    suitable: bool = True,
) -> Path:
    rows: list[dict[str, object]] = []
    confusion = [[0 for _ in range(4)] for _ in range(4)]
    correct = 0
    for label in range(4):
        for item in range(samples_per_class):
            is_error = item % error_period == 0
            prediction = (label + 1) % 4 if is_error else label
            energies = [1.0, 1.0, 1.0, 1.0]
            energies[label] = 8.0 if is_error else 12.0
            energies[prediction] = 12.0
            is_correct = prediction == label
            correct += int(is_correct)
            confusion[label][prediction] += 1
            rows.append(
                {
                    "key": f"fixed_i{label * samples_per_class + item:04d}_y{label}",
                    "label": label,
                    "prediction": prediction,
                    "correct": is_correct,
                    "raw_energy_0": energies[0],
                    "raw_energy_1": energies[1],
                    "raw_energy_2": energies[2],
                    "raw_energy_3": energies[3],
                }
            )
    predictions = root / "hardware_predictions_raw.csv"
    _write_csv(predictions, rows)
    metrics = {
        "samples": len(rows),
        "profile": profile,
        "suitable_for_accuracy_reporting": suitable,
        "confusion_matrix": confusion,
        "capture_manifest": {"phase_sha256": "a" * 64},
    }
    if suitable:
        metrics["accuracy"] = correct / len(rows)
    (root / "hardware_metrics_raw.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return root


def test_paper_outputs_keep_quick40_out_of_formal_mask_comparison(
    tmp_path: Path,
) -> None:
    mask_a_formal = _write_run(
        tmp_path / "mask_a_formal",
        profile="formal_fixed_random_100_per_class",
        samples_per_class=100,
        error_period=5,
    )
    mask_b_formal = _write_run(
        tmp_path / "mask_b_formal",
        profile="formal_fixed_random_100_per_class",
        samples_per_class=100,
        error_period=10,
    )
    mask_a_quick = _write_run(
        tmp_path / "mask_a_quick",
        profile="quick_fixed_random_10_per_class",
        samples_per_class=10,
        error_period=5,
        suitable=False,
    )
    output = tmp_path / "paper"
    report = evaluate_prediction_runs(
        runs=[
            PredictionRunSpec("mask_a", mask_a_formal),
            PredictionRunSpec("mask_b", mask_b_formal),
            PredictionRunSpec("mask_a", mask_a_quick),
        ],
        output_dir=output,
    )
    assert report["run_count"] == 3
    assert report["formal400_run_count"] == 2
    assert len(report["paired_formal400_comparisons"]) == 1
    assert report["paired_formal400_comparisons"][0]["samples"] == 400

    run_rows = list(
        csv.DictReader((output / "run_summary.csv").open(encoding="utf-8-sig"))
    )
    assert len(run_rows) == 3
    quick = next(row for row in run_rows if row["profile"].startswith("quick"))
    assert quick["reporting_status"] == "quick40_diagnostic"
    assert quick["eligible_for_formal_comparison"] == "False"
    assert quick["formal_accuracy"] == ""
    formal_rows = list(
        csv.DictReader(
            (output / "formal400_mask_summary.csv").open(encoding="utf-8-sig")
        )
    )
    assert {row["mask_name"] for row in formal_rows} == {"mask_a", "mask_b"}
    assert all(row["reporting_status"] == "formal400" for row in formal_rows)
    assert float(formal_rows[0]["accuracy_wilson95_low"]) < float(
        formal_rows[0]["accuracy"]
    ) < float(formal_rows[0]["accuracy_wilson95_high"])

    per_class = list(
        csv.DictReader((output / "per_class_metrics.csv").open(encoding="utf-8-sig"))
    )
    assert len(per_class) == 12
    assert {"precision", "recall", "f1", "support"}.issubset(per_class[0])
    paired = list(
        csv.DictReader(
            (output / "paired_formal400_mask_comparison.csv").open(
                encoding="utf-8-sig"
            )
        )
    )
    assert len(paired) == 1
    assert float(paired[0]["accuracy_b_minus_a"]) == pytest.approx(0.1)

    figure_rows = list(
        csv.DictReader((output / "figure_manifest.csv").open(encoding="utf-8-sig"))
    )
    assert {row["format"] for row in figure_rows} == {"pdf", "svg", "png"}
    assert all(float(row["height_cm"]) == pytest.approx(5.0) for row in figure_rows)
    assert all(float(row["font_size_pt"]) == pytest.approx(7.0) for row in figure_rows)
    comparison = output / "figures" / "formal400_mask_comparison.png"
    with Image.open(comparison) as image:
        assert abs(image.height - round(5.0 / 2.54 * 600)) <= 2
    svg = (output / "figures" / "formal400_mask_comparison.svg").read_text(
        encoding="utf-8"
    )
    assert "Arial" in svg
    assert (output / "figures" / "formal400_mask_comparison.pdf").is_file()
    assert (output / "output_inventory.json").is_file()


def test_raw_energy_prediction_mismatch_is_rejected(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "bad",
        profile="quick_fixed_random_10_per_class",
        samples_per_class=10,
        error_period=5,
    )
    path = run / "hardware_predictions_raw.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    rows[0]["prediction"] = "3"
    _write_csv(path, rows)
    with pytest.raises(RuntimeError, match="raw-energy argmax"):
        evaluate_prediction_runs(
            runs=[PredictionRunSpec("bad_mask", run)],
            output_dir=tmp_path / "out",
            make_plots=False,
        )


def test_biased_demo_requires_explicit_diagnostic_permission(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "demo",
        profile="demo_topk",
        samples_per_class=1,
        error_period=5,
        suitable=False,
    )
    with pytest.raises(PermissionError, match="biased demo"):
        evaluate_prediction_runs(
            runs=[PredictionRunSpec("demo_mask", run)],
            output_dir=tmp_path / "denied",
            make_plots=False,
        )
    report = evaluate_prediction_runs(
        runs=[PredictionRunSpec("demo_mask", run)],
        output_dir=tmp_path / "allowed",
        allow_biased_diagnostic=True,
        make_plots=False,
    )
    only = next(iter(report["runs"].values()))
    assert only["reporting_status"] == "biased_demo_diagnostic"
    assert only["eligible_for_formal_comparison"] is False
