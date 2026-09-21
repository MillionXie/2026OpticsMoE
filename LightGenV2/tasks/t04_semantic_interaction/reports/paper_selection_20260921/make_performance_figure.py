"""Render the fixed paper comparison; values come from auditable run files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"


def main() -> None:
    rows = list(csv.DictReader((ROOT / "paper_metrics.csv").open(encoding="utf-8")))
    selected = [rows[index] for index in (1, 4, 5, 6)]
    labels = [
        "Changed-cell\naccuracy", "Edit-grid\nIoU", "Object\nF1", "Scene exact\nmatch"
    ]
    ours = np.array([float(row["ours_expansion_0p5_epoch30"]) for row in selected])
    baseline = np.array([float(row["qwen_frozen_epoch30"]) for row in selected])
    x = np.arange(len(labels)); width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.3), facecolor="white")
    baseline_bars = ax.bar(x - width / 2, baseline, width, color="#B7BCC4", label="Frozen Qwen3-VL")
    ours_bars = ax.bar(x + width / 2, ours, width, color="#2878B5", label="Ours (compressed optical MoE)")
    ax.set_ylim(0.58, 1.01)
    ax.set_ylabel("Test score")
    ax.set_xticks(x, labels)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    for bars in (baseline_bars, ours_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(0.01, -0.25,
            "Same 5,000/1,000 split and shared readout; Ours uses the selected epoch-30 checkpoint.",
            transform=ax.transAxes, fontsize=8.5, color="#666666")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.25)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "paper_performance_comparison.png", dpi=320, facecolor="white")
    fig.savefig(OUT / "paper_performance_comparison.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
