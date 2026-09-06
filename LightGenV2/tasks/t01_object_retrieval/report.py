"""Collect the three formal T01 runs into one auditable table and figure."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


METHODS = (
    ("main", "Optical Router + MoE"),
    ("d2nn", "Dense D2NN"),
    ("qwen", "Frozen Qwen embedding"),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _row(key: str, label: str, run_dir: Path) -> dict[str, Any]:
    metrics = _read_json(run_dir / "student_metrics.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    fairness_path = run_dir / "parameter_fairness_contract.json"
    fairness = _read_json(fairness_path) if fairness_path.is_file() else {}
    selection_path = run_dir / "metrics" / "ema_best_observed_test.json"
    selection = _read_json(selection_path) if selection_path.is_file() else {}
    architecture_path = run_dir / "student_architecture.json"
    architecture = (
        _read_json(architecture_path) if architecture_path.is_file() else {}
    )
    alpha: dict[str, Any] | None = None
    diagnostics_path = run_dir / "fusion_diagnostics_last_batch.json"
    if diagnostics_path.is_file():
        alpha = _read_json(diagnostics_path)
    return {
        "key": key,
        "method": label,
        "run_dir": str(run_dir.resolve()),
        "git_commit": _read_json(run_dir / "environment.json").get("git_commit"),
        "seed": manifest.get("optimization_seed"),
        "top1": metrics["top1_retrieval_accuracy"],
        "top3": metrics["top3_retrieval_accuracy"],
        "mrr": metrics["mrr"],
        "query_count": metrics["query_count"],
        "selected_epoch": selection.get("epoch"),
        "selection_biased": metrics.get("selection_biased", key != "qwen"),
        "active_expert_phase_parameters": fairness.get(
            "moe_active_expert_parameters_vision_language"
        ),
        "d2nn_phase_parameters": fairness.get(
            "d2nn_phase_parameters_vision_language"
        ),
        "physical_capture_count": architecture.get(
            "physical_capture_count_with_router",
            architecture.get("physical_capture_count", 0 if key == "qwen" else None),
        ),
        "fusion": alpha,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, label in METHODS:
        selected = [row for row in rows if row["key"] == key]
        if not selected:
            raise RuntimeError(f"No completed run was supplied for {key}")
        values = [float(row["top1"]) for row in selected]
        output.append(
            {
                "key": key,
                "method": label,
                "run_count": len(selected),
                "top1_mean": statistics.fmean(values),
                "top1_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "top1_min": min(values),
                "top1_max": max(values),
                "top3_mean": statistics.fmean(float(row["top3"]) for row in selected),
                "mrr_mean": statistics.fmean(float(row["mrr"]) for row in selected),
            }
        )
    return output


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(3.45, 1.97), constrained_layout=True)
    labels = [row["method"] for row in rows]
    values = [100.0 * float(row["top1_mean"]) for row in rows]
    errors = [100.0 * float(row["top1_std"]) for row in rows]
    colors = ["#2F7E79", "#7A7A7A", "#CF7B31"]
    bars = axis.bar(
        range(len(rows)),
        values,
        yerr=errors,
        capsize=2.0,
        color=colors,
        width=0.68,
    )
    axis.set_ylabel("Top-1 retrieval accuracy (%)")
    axis.set_xticks(range(len(rows)), labels, rotation=15, ha="right")
    axis.set_ylim(0.0, max(100.0, max(values) * 1.12))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the formal T01 comparison")
    parser.add_argument("--main", required=True, nargs="+")
    parser.add_argument("--d2nn", required=True, nargs="+")
    parser.add_argument("--qwen", required=True, nargs="+")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "reports" / "formal_comparison"),
    )
    args = parser.parse_args()
    paths = {
        "main": [Path(value).expanduser() for value in args.main],
        "d2nn": [Path(value).expanduser() for value in args.d2nn],
        "qwen": [Path(value).expanduser() for value in args.qwen],
    }
    rows = [
        _row(key, label, run_dir)
        for key, label in METHODS
        for run_dir in paths[key]
    ]
    aggregates = _aggregate(rows)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(
            {"schema_version": 1, "runs": rows, "aggregates": aggregates},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    flat_fields = [
        "key", "method", "run_dir", "git_commit", "seed", "top1", "top3",
        "mrr", "query_count", "selected_epoch", "selection_biased",
        "active_expert_phase_parameters", "d2nn_phase_parameters",
        "physical_capture_count",
    ]
    with (output / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in flat_fields} for row in rows])
    aggregate_fields = [
        "key", "method", "run_count", "top1_mean", "top1_std", "top1_min",
        "top1_max", "top3_mean", "mrr_mean",
    ]
    with (output / "comparison_aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)
    _plot(aggregates, output / "comparison_top1")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
