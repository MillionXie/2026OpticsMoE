"""Plot the formal 9x4 versus 16x4 throughput/quality/router trade-off."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "paper_results"


def _load(name: str) -> dict:
    return json.loads((ROOT / name / "result.json").read_text(encoding="utf-8"))


def main() -> int:
    nine = _load("temporal_multivideo9x4_contentroute")
    sixteen = _load("temporal_multivideo16x4")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(18 / 2.54, 5.2 / 2.54), constrained_layout=True)
    colors = ("#4477AA", "#EE6677")
    labels = ("9 videos × 4 frames", "16 videos × 4 frames")

    x = np.arange(2)
    srcc = [nine["test_metrics_optical_on"]["srcc"], sixteen["test_metrics_optical_on"]["srcc"]]
    plcc = [nine["test_metrics_optical_on"]["plcc"], sixteen["test_metrics_optical_on"]["plcc"]]
    width = 0.34
    axes[0].bar(x - width / 2, srcc, width, label="SRCC", color="#4477AA")
    axes[0].bar(x + width / 2, plcc, width, label="PLCC", color="#CC6677")
    axes[0].axhline(0.81, color="0.35", linestyle="--", linewidth=0.8, label="SRCC target")
    axes[0].set_ylim(0.78, 0.83)
    axes[0].set_xticks(x, ("9×4", "16×4"))
    axes[0].set_ylabel("correlation")
    axes[0].set_title("a  Quality")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].bar(x, (9, 16), color=colors)
    axes[1].set_xticks(x, ("9×4", "16×4"))
    axes[1].set_ylabel("independent MOS / optical field")
    axes[1].set_title("b  Parallel outputs")
    for index, value in enumerate((9, 16)):
        axes[1].text(index, value + 0.3, str(value), ha="center", va="bottom")

    frame = np.asarray(sixteen["router"]["frame_selected_share"])
    video = np.asarray(sixteen["router"]["video_selected_share"])
    expert = np.arange(4)
    axes[2].bar(expert - width / 2, frame, width, label="frame router", color="#228833")
    axes[2].bar(expert + width / 2, video, width, label="video router", color="#AA3377")
    axes[2].axhline(0.25, color="0.35", linestyle="--", linewidth=0.8)
    axes[2].set_xticks(expert, ("E1", "E2", "E3", "E4"))
    axes[2].set_ylabel("Top-2 selection share")
    axes[2].set_ylim(0, 0.42)
    axes[2].set_title("c  16×4 expert use")
    axes[2].legend(frameon=False)

    output = ROOT / "temporal_multivideo16x4" / "multivideo_tradeoff"
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
