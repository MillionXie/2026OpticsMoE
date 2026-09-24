"""Draw the candidate CCD/slot numbering without using sample data."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .model import DirectCCDOptics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = DirectCCDOptics("moe")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    specs = (
        ("Router CCD: 16 ports, 16 px gaps", model.router_centers,
         model.router_side, True),
        ("Expert input: 16 fixed slots", [
            (y + model.expert_size // 2, x + model.expert_size // 2)
            for y, x in model.slots], model.expert_size, True),
        ("Final CCD: compact 3-4-3 candidate", model.output_centers,
         model.output_side, False),
    )
    for ax, (title, centers, side, stage_colors) in zip(axes, specs):
        ax.set_xlim(0, model.width)
        ax.set_ylim(model.height, 0)
        ax.set_aspect("equal")
        ax.set_facecolor("#101620")
        for index, (y, x) in enumerate(centers):
            color = ("#40c9a2", "#f2bb4c", "#e87473", "#9b8cea")[index // 4] \
                if stage_colors else "#53b8ed"
            ax.add_patch(Rectangle((x-side/2, y-side/2), side, side,
                                   facecolor=color, edgecolor="white", linewidth=1.4,
                                   alpha=0.85))
            ax.text(x, y, str(index + 1 if stage_colors else index),
                    ha="center", va="center", fontsize=11, weight="bold")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
