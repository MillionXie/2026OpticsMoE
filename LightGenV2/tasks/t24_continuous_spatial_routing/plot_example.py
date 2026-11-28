"""Render test index 0 using the verified Kather seed-17 routing predictions."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

TASK = Path(__file__).resolve().parent
FACE, SIDE = 478, 224


def load(mode):
    root = TASK / "runs/simulation" / f"kather_s17_{mode}"
    path = root / "predictions_test.npz"
    manifest = json.loads((root / "manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[path.name]
    with np.load(path, allow_pickle=False) as data:
        return str(data["ids"][0]), int(data["labels"][0]), data["positions"][0], data["power"][0]


def main():
    rows = [load(mode) for mode in ("fixed", "continuous")]
    assert rows[0][:2] == rows[1][:2]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.3))
    fig.subplots_adjust(left=0.085, right=0.97, top=0.82, bottom=0.24, wspace=0.22)
    fig.suptitle(f"Kather2016   ·   {rows[0][0]}   ·   test image 0   ·   seed 17", fontsize=14, y=0.97)
    for ax, name, (_, _, positions, power) in zip(axes, ("Four fixed locations", "Continuous placement"), rows):
        for i, ((y, x), color) in enumerate(zip(positions, ("#007992", "#d36927"))):
            ax.add_patch(Rectangle((x, y), SIDE, SIDE, facecolor=color, alpha=0.25, edgecolor="none"))
            ax.add_patch(Rectangle((x, y), SIDE, SIDE, fill=False, edgecolor=color, linewidth=2))
            ax.text(x + 6, y + 20, str(i + 1), color=color, weight="bold", fontsize=12)
        intersection = np.maximum(0, SIDE - np.abs(positions[0] - positions[1])).prod()
        ax.set_title(name, fontsize=13)
        ax.set(xlim=(0, FACE), ylim=(FACE, 0), aspect="equal",
               xticks=[0, 112, 224, 336, FACE], yticks=[0, 112, 224, 336, FACE],
               xlabel="x (pixels)")
        ax.tick_params(labelsize=8)
        ax.text(0.5, -0.22, f"Route power:  1  {power[0]:.1%}    2  {power[1]:.1%}\n"
                f"Footprint overlap:  {intersection / SIDE**2:.1%}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10, linespacing=1.5)
    axes[0].set_ylabel("y (pixels)")
    fig.text(0.5, 0.025, "Each square is a 224 × 224 input copy. Overlap is a fraction of one copy's area.",
             ha="center", fontsize=9, color="#414a50")
    out = TASK / "reports/kather_pilot/placement_example.png"
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
