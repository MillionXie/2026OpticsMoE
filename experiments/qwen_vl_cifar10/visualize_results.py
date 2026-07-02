from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .config import EXPERIMENT_DIR


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _model_label(metrics: dict[str, Any]) -> str:
    return str(metrics.get("model_id", "unknown")).split("/")[-1]


def _stages(metrics: dict[str, Any]) -> dict[str, float]:
    timing = metrics.get("timing", {})
    stages = timing.get("stages", {}) if isinstance(timing, dict) else {}
    return {
        "dataset": float(
            stages.get("dataset_load_sec", metrics.get("dataset_load_time_sec", 0.0))
        ),
        "model load": float(
            stages.get("model_load_sec", metrics.get("model_load_time_sec", 0.0))
        ),
        "features": float(
            stages.get(
                "feature_extraction_total_sec",
                metrics.get("feature_extraction_total_sec", 0.0),
            )
        ),
        "training": float(
            stages.get(
                "head_or_adapter_train_sec", metrics.get("head_train_total_sec", 0.0)
            )
        ),
        "evaluation": float(
            stages.get("evaluation_total_sec", metrics.get("evaluation_total_sec", 0.0))
        ),
        "generation": float(
            stages.get("generation_total_sec", metrics.get("generation_total_sec", 0.0))
        ),
        "benchmark": float(
            stages.get(
                "benchmark_total_sec", metrics.get("benchmark_total_time_sec", 0.0)
            )
        ),
    }


def write_run_figure(metrics: dict[str, Any], output_path: Path) -> None:
    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    history = metrics.get("training_history", [])

    if history:
        epochs = [int(row["epoch"]) for row in history]
        axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="train")
        axes[0, 0].plot(epochs, [row["eval_loss"] for row in history], label="eval")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Cross-entropy loss")
        axes[0, 0].set_title("Training and evaluation loss")
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, [row["accuracy"] for row in history], label="accuracy")
        axes[0, 1].plot(epochs, [row["macro_f1"] for row in history], label="macro-F1")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_ylim(0.0, 1.0)
        axes[0, 1].set_title("Validation performance")
        axes[0, 1].legend()
    else:
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")
        axes[0, 0].text(
            0.5, 0.5, "No epoch history for this mode", ha="center", va="center"
        )

    per_class = metrics.get("per_class_accuracy", {})
    class_names = list(per_class)
    class_scores = [float(per_class[name]) for name in class_names]
    axes[1, 0].bar(class_names, class_scores, color="#4C78A8")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Per-class test accuracy")
    axes[1, 0].tick_params(axis="x", rotation=35)

    stages = _stages(metrics)
    stage_names = [name for name, seconds in stages.items() if seconds > 0]
    stage_minutes = [stages[name] / 60.0 for name in stage_names]
    axes[1, 1].barh(stage_names, stage_minutes, color="#F58518")
    axes[1, 1].set_xlabel("Minutes")
    axes[1, 1].set_title("Measured stage time (CUDA synchronized)")

    total_seconds = float(metrics.get("total_wall_time_sec", sum(stages.values())))
    total_minutes = total_seconds / 60.0
    figure.suptitle(
        f"{_model_label(metrics)} | accuracy={float(metrics.get('accuracy', 0.0)):.4f} | "
        f"macro-F1={float(metrics.get('macro_f1', 0.0)):.4f} | wall={total_minutes:.1f} min",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=200)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def write_comparison_figure(
    metrics_rows: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    if not metrics_rows:
        raise ValueError("No metrics files were found.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(metrics_rows, key=lambda row: _model_label(row))
    _write_summary_csv(rows, output_dir / "results_summary.csv")

    plt = _pyplot()
    labels = [_model_label(row) for row in rows]
    x = list(range(len(rows)))
    figure, axes = plt.subplots(
        2, 1, figsize=(max(10, 2.3 * len(rows)), 9), constrained_layout=True
    )
    width = 0.38
    axes[0].bar(
        [value - width / 2 for value in x],
        [float(row.get("accuracy", 0.0)) for row in rows],
        width,
        label="accuracy",
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        [float(row.get("macro_f1", 0.0)) for row in rows],
        width,
        label="macro-F1",
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Qwen3-VL CIFAR-10 performance")
    axes[0].legend()

    bottoms = [0.0] * len(rows)
    for stage in (
        "dataset",
        "model load",
        "features",
        "training",
        "evaluation",
        "generation",
        "benchmark",
    ):
        values = [_stages(row)[stage] / 60.0 for row in rows]
        if not any(values):
            continue
        axes[1].bar(x, values, bottom=bottoms, label=stage)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axes[1].set_ylabel("Minutes")
    axes[1].set_title("Measured stage-time breakdown")
    axes[1].legend(ncol=3)

    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
    output_path = output_dir / "qwen_vl_model_comparison.png"
    figure.savefig(output_path, dpi=200)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def _write_summary_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fields = (
        "model_id",
        "mode",
        "accuracy",
        "macro_f1",
        "total_wall_time_sec",
        "model_load_time_sec",
        "feature_extraction_total_sec",
        "head_train_total_sec",
        "evaluation_total_sec",
        "end_to_end_images_per_second",
        "cuda_peak_memory_mb",
        "feature_batch_size",
        "head_batch_size",
        "seed",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _load_metrics(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"Metrics root must be an object: {path}")
        rows.append(value)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot Qwen3-VL accuracy and research timings."
    )
    parser.add_argument("--runs-root", type=Path, default=EXPERIMENT_DIR / "runs")
    parser.add_argument("--metrics", type=Path, nargs="*")
    parser.add_argument(
        "--output-dir", type=Path, default=EXPERIMENT_DIR / "runs" / "summary"
    )
    args = parser.parse_args(argv)
    paths = args.metrics or sorted(args.runs_root.glob("**/metrics.json"))
    write_comparison_figure(_load_metrics(paths), args.output_dir)
    print(f"summary={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
