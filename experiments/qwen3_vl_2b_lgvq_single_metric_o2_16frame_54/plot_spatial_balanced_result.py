"""Draw the compact Spatial result/collapse audit figure (Arial, 7 pt)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.75), constrained_layout=True)
    metrics = ("srcc", "krcc", "plcc")
    on = [data["normal_optical_electronic"][key] for key in metrics]
    off = [data["same_checkpoint_optics_bypassed"][key] for key in metrics]
    x = np.arange(3)
    axes[0].bar(x - 0.18, on, 0.36, label="Optical + electronic", color="#0072B2")
    axes[0].bar(x + 0.18, off, 0.36, label="Same weights, optics off", color="#B7B7B7")
    axes[0].set_xticks(x, [key.upper() for key in metrics])
    axes[0].set_ylim(0, 0.75)
    axes[0].set_ylabel("Correlation")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].set_title("a  Optical contribution", loc="left", fontweight="bold")
    for axis, key, title in (
        (axes[1], "vision_router_selected_share", "b  Vision router"),
        (axes[2], "language_router_selected_share", "c  Sequence router"),
    ):
        values = data[key]
        axis.bar(np.arange(4), values, color="#009E73")
        axis.axhline(0.25, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_xticks(np.arange(4), ["E0", "E1", "E2", "E3"])
        axis.set_ylim(0, 0.55)
        axis.set_ylabel("Selected share")
        axis.set_title(title, loc="left", fontweight="bold")
        for i, value in enumerate(values):
            axis.text(i, value + 0.018, f"{value:.2f}", ha="center", va="bottom")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"spatial_balanced_metrics_router.{suffix}", dpi=300)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
