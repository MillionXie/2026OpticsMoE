"""Create the compact ABO optical-MoE evidence figure from completed runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MATCHED_QWEN = {"recall_at_1": 0.5979166667, "recall_at_5": 0.8516666667,
                "recall_at_10": 0.90625, "mrr": 0.7150933891}
FULL_QWEN = {"recall_at_1": 0.7370833333, "recall_at_5": 0.93375,
             "recall_at_10": 0.9604166667, "mrr": 0.8230330540}


def _rows(run: Path) -> list[dict[str, str]]:
    with (run / "training_history.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _best(rows: list[dict[str, str]]) -> dict[str, str]:
    measured = [row for row in rows if row.get("test_recall_at_1")]
    if not measured:
        raise RuntimeError("No periodic test measurement exists")
    return max(measured, key=lambda row: float(row["test_recall_at_1"]))


def _shares(row: dict[str, str], key: str) -> np.ndarray:
    counts = np.asarray(json.loads(row[key]), dtype=np.float64)
    return counts / counts.sum()


def summarize(runs: list[Path], output: Path) -> dict[str, Any]:
    histories = {run.name: _rows(run) for run in runs}
    best = {name: _best(rows) for name, rows in histories.items()}
    selected_name = max(best, key=lambda name: float(best[name]["test_recall_at_1"]))
    selected = best[selected_name]
    output.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    for name, rows in histories.items():
        measured = [row for row in rows if row.get("test_recall_at_1")]
        axes[0].plot([int(row["epoch"]) for row in measured],
                     [float(row["test_recall_at_1"]) for row in measured],
                     marker="o", label=name.replace("optical_router_moe_", ""))
    axes[0].axhline(MATCHED_QWEN["recall_at_1"], color="#777777", linestyle="--",
                    label="Frozen Qwen 64-D")
    axes[0].axhline(FULL_QWEN["recall_at_1"], color="#222222", linestyle=":",
                    label="Frozen Qwen 2048-D")
    axes[0].set(xlabel="Epoch", ylabel="Test R@1", title="a  Periodic test retrieval")
    axes[0].legend(fontsize=8)

    keys = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")
    labels = ("R@1", "R@5", "R@10", "MRR")
    x = np.arange(len(keys))
    axes[1].bar(x - 0.25, [MATCHED_QWEN[key] for key in keys], 0.25,
                label="Qwen 64-D", color="#9e9e9e")
    axes[1].bar(x, [FULL_QWEN[key] for key in keys], 0.25,
                label="Qwen 2048-D", color="#4c78a8")
    axes[1].bar(x + 0.25, [float(selected[f"test_{key}"]) for key in keys], 0.25,
                label="Optical MoE", color="#f58518")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set(title="b  Best checkpoint", ylabel="Score")
    axes[1].legend(fontsize=8)

    route_keys = ("vision_image_router_counts", "language_image_router_counts",
                  "language_title_router_counts")
    route_names = ("Vision image", "Language image", "Language title")
    bottom = np.zeros(3)
    for expert in range(4):
        values = np.asarray([_shares(selected, key)[expert] for key in route_keys])
        axes[2].bar(route_names, values, bottom=bottom, label=f"Expert {expert}")
        bottom += values
    axes[2].axhline(0.5, color="white", linewidth=0.8, alpha=0.8)
    axes[2].set_ylim(0, 1)
    axes[2].set(title="c  Top-2 usage at selected epoch", ylabel="Selection share")
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].legend(fontsize=8, ncol=2)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"optical_moe_summary.{suffix}", dpi=200)
    plt.close(figure)

    records = []
    for name, metrics in (("frozen_qwen_64d", MATCHED_QWEN),
                          ("frozen_qwen_2048d", FULL_QWEN)):
        records.append({"method": name, "selected_epoch": "", **metrics})
    records.append({
        "method": selected_name, "selected_epoch": int(selected["epoch"]),
        **{key: float(selected[f"test_{key}"]) for key in keys},
    })
    with (output / "optical_moe_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "selected_run": selected_name,
        "selected_epoch": int(selected["epoch"]),
        "metrics": records[-1],
        "router_selection_shares": {
            name: _shares(selected, key).tolist()
            for name, key in zip(route_names, route_keys)
        },
        "candidate_best_r1": {
            name: {"epoch": int(row["epoch"]), "recall_at_1": float(row["test_recall_at_1"])}
            for name, row in best.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(
        [Path(value).expanduser().resolve() for value in args.run_dir],
        Path(args.output).expanduser().resolve(),
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
