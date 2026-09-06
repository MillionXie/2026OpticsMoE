"""Render the cross-task RTX 5090 D baseline handoff figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def _load() -> list[dict[str, str]]:
    with (HERE / "summary.csv").open("r", encoding="utf-8-sig", newline="") as stream:
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
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def baseline_overview(rows: list[dict[str, str]]) -> None:
    labels = [row["short_name"] for row in rows]
    positions = np.arange(len(rows))
    metrics = np.asarray([float(row["primary_metric_value"]) for row in rows])
    latency = np.asarray([float(row["latency_mean_ms"]) for row in rows])
    mean_power = np.asarray([float(row["active_mean_w"]) for row in rows])
    peak_power = np.asarray([float(row["active_peak_w"]) for row in rows])

    figure, axes = plt.subplots(
        1, 3, figsize=(18.0 / 2.54, 5.2 / 2.54), constrained_layout=True
    )
    colors = ["#3B6FB6", "#4C9F70", "#D98C3F", "#8064A2", "#C84C4C"]

    axes[0].bar(positions, metrics, color=colors, width=0.68)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Primary metric")
    axes[0].set_title("a  Performance")
    for index, (value, row) in enumerate(zip(metrics, rows)):
        axes[0].text(
            index,
            value + 0.025,
            f"{value:.3f}\n{row['primary_metric_name']}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    axes[1].bar(positions, latency, color=colors, width=0.68)
    axes[1].set_yscale("log")
    axes[1].set_ylim(latency.min() * 0.72, latency.max() * 1.75)
    axes[1].set_ylabel("Latency (ms, log scale)")
    axes[1].set_title("b  First block to output")
    for index, value in enumerate(latency):
        axes[1].text(index, value * 1.12, f"{value:.1f}", ha="center", va="bottom")

    width = 0.34
    axes[2].bar(
        positions - width / 2,
        mean_power,
        width,
        color="#2E75B6",
        label="Active mean",
    )
    axes[2].bar(
        positions + width / 2,
        peak_power,
        width,
        color="#ED7D31",
        label="Active peak",
    )
    axes[2].axhline(575.0, color="#666666", linestyle="--", linewidth=0.8, label="Rated 575 W")
    axes[2].set_ylim(0, 650)
    axes[2].set_ylabel("GPU board power (W)")
    axes[2].set_title("c  Power")
    axes[2].legend(frameon=False, loc="upper left")

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=24, ha="right")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].text(
        0.0,
        -0.43,
        "Metrics are task-specific and must not be compared across tasks.",
        transform=axes[0].transAxes,
        fontsize=6,
    )
    base = HERE / "qwen5090d_baseline_overview"
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def temporal_power_comparison() -> None:
    labels = ["Optical\n16 parallel", "Qwen\n16 sequential"]
    time_ms = [9.084, 1046.928]
    average_w = [80.388, 120.680 / 1.046928]
    energy_j = [0.730244592, 120.680]
    upper_j = [np.nan, 575.0 * 1.046928]
    figure, axes = plt.subplots(
        1, 3, figsize=(18.0 / 2.54, 5.0 / 2.54), constrained_layout=True
    )
    colors = ["#2E75B6", "#C84C4C"]
    values = [time_ms, average_w, energy_j]
    ylabels = ["Time for 16 videos (ms)", "Average power (W)", "Energy for 16 videos (J)"]
    titles = ["a  Latency", "b  Power", "c  Energy"]
    for axis, data, ylabel, title in zip(axes, values, ylabels, titles):
        bars = axis.bar([0, 1], data, color=colors, width=0.62)
        axis.set_yscale("log")
        axis.set_xticks([0, 1], labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, data):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.12,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )
    axes[0].set_ylim(min(time_ms) * 0.70, max(time_ms) * 1.45)
    axes[1].set_ylim(min(average_w) * 0.92, max(average_w) * 1.25)
    axes[2].set_ylim(min(energy_j) * 0.70, upper_j[1] * 1.65)
    axes[2].scatter([1], [upper_j[1]], marker="_", s=170, color="#222222", zorder=3)
    axes[2].text(1, upper_j[1] * 1.12, "601.98 rated upper", ha="center", va="bottom", fontsize=6)
    base = HERE / "temporal16_power_comparison"
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    _style()
    baseline_overview(_load())
    temporal_power_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
