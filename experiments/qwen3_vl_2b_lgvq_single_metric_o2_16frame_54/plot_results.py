"""Create compact paper/handoff figures from completed LGVQ runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _run_record(run_dir: Path) -> dict[str, Any]:
    on = _read_json(run_dir / "test_metrics_optical_on.json")
    off = _read_json(run_dir / "test_metrics_optical_off.json")
    summary = _read_json(run_dir / "training_summary.json")
    history = _read_json(run_dir / "train_history.json")
    tested = [row for row in history if row.get("test_evaluated")]
    alphas = {
        stage: float(values["alpha"])
        for stage, values in on.get("fusion_diagnostics", {}).items()
    }
    return {
        "run": run_dir.name,
        "target": on["target"],
        "run_dir": str(run_dir.resolve()),
        "best_epoch": int(summary["best_epoch"]),
        "optical_on": on,
        "optical_off": off,
        "alphas": alphas,
        "test_history": tested,
    }


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
        }
    )


def _save(fig: Any, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=600, bbox_inches="tight")


def _plot_comparison(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [f"{r['target']}\n{r['run']}" for r in records]
    x = np.arange(len(records), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(17.8 / 2.54, 5.5 / 2.54))

    width = 0.34
    axes[0].bar(x - width / 2, [r["optical_on"]["srcc"] for r in records], width, label="Optical on", color="#0072B2")
    axes[0].bar(x + width / 2, [r["optical_off"]["srcc"] for r in records], width, label="Same checkpoint, optical off", color="#B5B5B5")
    axes[0].axhline(0.80, color="#D55E00", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("SRCC")
    axes[0].set_xticks(x, names, rotation=25, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)

    stages = ("vision_expert", "vision_global", "language_expert", "language_global")
    colors = ("#0072B2", "#56B4E9", "#E69F00", "#D55E00")
    for index, stage in enumerate(stages):
        axes[1].bar(
            x + (index - 1.5) * 0.18,
            [r["alphas"].get(stage, float("nan")) for r in records],
            0.18,
            label=stage.replace("_", " "),
            color=colors[index],
        )
    axes[1].set_ylabel("Optical fusion alpha")
    axes[1].set_xticks(x, names, rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False, ncol=2)

    for record, color in zip(records, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")):
        tested = record["test_history"]
        axes[2].plot(
            [row["epoch"] for row in tested],
            [row["test_optical_on"]["srcc"] for row in tested],
            marker="o",
            markersize=2.5,
            label=record["run"],
            color=color,
        )
    axes[2].axhline(0.80, color="#777777", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Test SRCC")
    axes[2].legend(frameon=False)

    for label, axis in zip("abc", axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.3)
    _save(fig, output, "lgvq_optical_contribution")
    plt.close(fig)


def _plot_phase(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(records), 1, squeeze=False, figsize=(17.8 / 2.54, max(4.0, 3.2 * len(records)) / 2.54)
    )
    axes = axes[:, 0]
    for axis, record in zip(axes, records):
        csv_path = Path(record["run_dir"]) / "phase_snapshots" / "phase_evolution_summary.csv"
        if not csv_path.is_file():
            axis.text(0.5, 0.5, "phase summary unavailable", ha="center", va="center")
            continue
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
        planes = sorted({row["plane"] for row in rows})
        for plane in planes:
            selected = [row for row in rows if row["plane"] == plane]
            axis.plot(
                [int(row["epoch"]) for row in selected],
                [float(row["wrapped_delta_from_first_rms_rad"]) for row in selected],
                marker="o",
                markersize=2.2,
                label=plane.replace("raw_", "").replace("_phase", ""),
            )
        axis.set_title(record["run"], loc="left")
        axis.set_ylabel("Wrapped phase Δ RMS (rad)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout(h_pad=1.0)
    _save(fig, output, "lgvq_phase_evolution")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = [_run_record(Path(value).expanduser().resolve()) for value in args.run_dir]

    for record in records:
        from .phase_snapshots import summarize_directory

        summarize_directory(Path(record["run_dir"]) / "phase_snapshots")

    table_rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "run": record["run"],
            "target": record["target"],
            "best_epoch": record["best_epoch"],
        }
        for mode in ("optical_on", "optical_off"):
            for metric in ("srcc", "krcc", "plcc", "rmse", "mae"):
                row[f"{mode}_{metric}"] = record[mode][metric]
        for stage, alpha in record["alphas"].items():
            row[f"alpha_{stage}"] = alpha
        table_rows.append(row)
    fields = sorted({key for row in table_rows for key in row})
    with (output / "lgvq_result_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    (output / "lgvq_result_table.json").write_text(
        json.dumps(table_rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _configure_matplotlib()
    _plot_comparison(records, output)
    _plot_phase(records, output)
    print(json.dumps({"status": "complete", "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
