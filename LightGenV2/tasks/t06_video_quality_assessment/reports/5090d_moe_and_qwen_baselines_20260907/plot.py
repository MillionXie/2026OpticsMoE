"""Render the unified RTX 5090 D optical-MoE and Qwen baseline figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def _read_csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _finish(figure: plt.Figure, stem: str) -> None:
    figure.savefig(HERE / f"{stem}.png", dpi=300)
    figure.savefig(HERE / f"{stem}.pdf")
    plt.close(figure)


def optical_stage_timing(rows: list[dict[str, str]]) -> None:
    labels = [row["task_id"] for row in rows]
    positions = np.arange(len(rows))
    post = np.asarray([float(row["post_ccd_feature_avg_median_ms"]) for row in rows])
    residual = np.asarray([float(row["parallel_residual_avg_median_ms"]) for row in rows])
    head = np.asarray([float(row["task_head_median_ms"]) for row in rows])
    passes = np.asarray([float(row["total_optical_passes"]) for row in rows])
    critical = np.asarray([float(row["critical_path_estimate_ms_per_call"]) for row in rows])
    physical = passes * 1.314
    serialized = critical - physical

    figure, axes = plt.subplots(1, 3, figsize=(18.0 / 2.54, 5.4 / 2.54))
    figure.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.24, wspace=0.48)
    colors = {"post": "#2E75B6", "residual": "#70AD47", "head": "#ED7D31"}

    axes[0].bar(positions, post, width=0.66, color=colors["post"])
    axes[0].set_ylabel("Median time (ms)")
    axes[0].set_title("a  Post-CCD feature work")
    for index, value in enumerate(post):
        axes[0].text(index, value + max(post) * 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=6)

    axes[1].bar(positions, residual, width=0.66, color=colors["residual"], label="Parallel residual")
    axes[1].axhline(1.314, color="#C84C4C", linestyle="--", linewidth=0.9, label="Physical pass 1.314 ms")
    axes[1].set_ylim(0, 1.48)
    axes[1].set_ylabel("Median time (ms)")
    axes[1].set_title("b  Residual coverage")
    axes[1].legend(frameon=False, loc="upper left")

    axes[2].bar(positions, physical, width=0.66, color="#4472C4", label="Physical passes")
    axes[2].bar(positions, serialized, bottom=physical, width=0.66, color="#A5A5A5", label="Serialized electronics + tail")
    axes[2].scatter(positions, head, marker="_", s=80, linewidth=1.2, color=colors["head"], label="Task head only")
    axes[2].set_ylabel("Time per physical call (ms)")
    axes[2].set_title("c  Graph critical path")
    axes[2].legend(frameon=False, loc="upper left")
    for index, value in enumerate(critical):
        axes[2].text(index, value + max(critical) * 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=6)

    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(0.07, 0.055, "T06 values are per 16-video physical field; other tasks are per sample.", fontsize=6)
    _finish(figure, "optical_moe_stage_timing")


def moe_qwen_comparison(rows: list[dict[str, str]]) -> None:
    labels = [row["task_id"] for row in rows]
    positions = np.arange(len(rows))
    ours_metric = np.asarray([float(row["ours_value"]) for row in rows])
    qwen_metric = np.asarray([float(row["qwen_value"]) for row in rows])
    ours_time = np.asarray([float(row["ours_critical_ms_per_call"]) for row in rows])
    qwen_time = np.asarray([float(row["qwen_mean_ms_per_call"]) for row in rows])
    ours_energy = np.asarray([float(row["ours_rig_wall_proxy_j_per_call"]) for row in rows])
    qwen_energy = np.asarray([float(row["qwen_measured_energy_j_per_call"]) for row in rows])
    width = 0.34

    figure, axes = plt.subplots(1, 3, figsize=(18.0 / 2.54, 5.5 / 2.54))
    figure.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.28, wspace=0.62)
    ours_color = "#2E75B6"
    qwen_color = "#C84C4C"

    axes[0].bar(positions - width / 2, ours_metric, width, color=ours_color, label="Optical MoE")
    axes[0].bar(positions + width / 2, qwen_metric, width, color=qwen_color, label="Frozen Qwen")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Task-specific primary metric")
    axes[0].set_title("a  Performance")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].text(3, 0.08, "not same\nmetric", ha="center", va="bottom", fontsize=6, color="#555555")

    axes[1].bar(positions - width / 2, ours_time, width, color=ours_color, label="Optical MoE graph")
    axes[1].bar(positions + width / 2, qwen_time, width, color=qwen_color, label="Qwen core")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Time per matched call (ms, log)")
    axes[1].set_title("b  Model/core latency")
    axes[1].legend(frameon=False, loc="upper left")

    axes[2].bar(positions - width / 2, ours_energy, width, color=ours_color, hatch="//", label="Optical-rig wall proxy")
    axes[2].bar(positions + width / 2, qwen_energy, width, color=qwen_color, label="Qwen measured GPU")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Energy per call (J, log)")
    axes[2].set_title("c  Energy boundary")
    axes[2].legend(frameon=False, loc="upper left")

    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(0.07, 0.105, "Metrics are task-specific; T04 endpoints differ and are not comparable.", fontsize=6)
    figure.text(0.07, 0.045, "Hatched MoE energy excludes GPU electronics; it is not full-system energy.", fontsize=6)
    _finish(figure, "moe_qwen_comparison")


def main() -> int:
    _style()
    optical_stage_timing(_read_csv("optical_task_summary.csv"))
    moe_qwen_comparison(_read_csv("baseline_and_moe_summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
