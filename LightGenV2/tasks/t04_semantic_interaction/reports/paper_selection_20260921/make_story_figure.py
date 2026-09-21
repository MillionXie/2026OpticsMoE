"""Create the fixed, publication-facing layered-scene task illustration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUT = ROOT / "figures"

SELECTION = [
    ("A", "add", "test_000588", "A new object must be placed relative to the house."),
    ("B", "replace", "test_000097", "Only the requested object identity changes."),
    ("C", "move", "test_000766", "Identity is preserved while the spatial relation changes."),
    ("D", "remove", "test_000839", "The requested object disappears; its neighbours remain."),
]


def main() -> None:
    records = {
        row["sample_id"]: row
        for row in map(json.loads, (ASSETS / "selected_story_records.jsonl").open(encoding="utf-8-sig"))
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.6, 10.0), constrained_layout=False, facecolor="white")
    grid = fig.add_gridspec(
        len(SELECTION), 3, width_ratios=(1, 0.78, 1),
        left=0.045, right=0.985, top=0.955, bottom=0.035,
        hspace=0.34, wspace=0.08,
    )
    task_colors = {"add": "#2878B5", "replace": "#7E57C2", "move": "#D97706", "remove": "#338A5A"}
    for row_index, (letter, task, sample_id, interpretation) in enumerate(SELECTION):
        record = records[sample_id]
        source = Image.open(ASSETS / f"{sample_id}_source.png").convert("RGB")
        target = Image.open(ASSETS / f"{sample_id}_target.png").convert("RGB")
        for col_index, (image, label) in enumerate(((source, "Input scene"), (target, "Target scene"))):
            column = 0 if col_index == 0 else 2
            ax = fig.add_subplot(grid[row_index, column])
            ax.imshow(image)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#D4D4D4"); spine.set_linewidth(0.8)
            ax.set_title(label, fontsize=10, color="#454545", pad=5)
            if column == 0:
                ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=14,
                        fontweight="bold", va="top", ha="left", color="#111111")
        text_ax = fig.add_subplot(grid[row_index, 1]); text_ax.axis("off")
        text_ax.text(0.5, 0.78, task.upper(), ha="center", va="center", fontsize=10,
                     fontweight="bold", color=task_colors[task])
        text_ax.annotate("", xy=(0.94, 0.52), xytext=(0.06, 0.52),
                         arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))
        text_ax.text(0.5, 0.43, record["instruction"], ha="center", va="top",
                     fontsize=10.5, color="#111111", wrap=True)
        text_ax.text(0.5, 0.12, interpretation, ha="center", va="bottom",
                     fontsize=8.7, color="#666666", wrap=True)
    fig.savefig(OUT / "layered_scene_story_examples.png", dpi=320, facecolor="white")
    fig.savefig(OUT / "layered_scene_story_examples.pdf", facecolor="white")
    plt.close(fig)

    # Separate rows are convenient for slides and supplementary figures.
    for letter, task, sample_id, interpretation in SELECTION:
        record = records[sample_id]
        source = Image.open(ASSETS / f"{sample_id}_source.png").convert("RGB")
        target = Image.open(ASSETS / f"{sample_id}_target.png").convert("RGB")
        row_fig = plt.figure(figsize=(9.0, 3.0), facecolor="white")
        row_grid = row_fig.add_gridspec(1, 3, width_ratios=(1, 1.25, 1),
                                        left=0.04, right=0.98, top=0.88, bottom=0.10, wspace=0.08)
        for column, (image, label) in enumerate(((source, "Input scene"), (target, "Target scene"))):
            ax = row_fig.add_subplot(row_grid[0, 0 if column == 0 else 2])
            ax.imshow(image); ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#D4D4D4"); spine.set_linewidth(0.8)
            ax.set_title(label, fontsize=10, color="#454545", pad=4)
        text_ax = row_fig.add_subplot(row_grid[0, 1]); text_ax.axis("off")
        text_ax.text(0.5, 0.82, f"{letter}  {task.upper()}", ha="center", fontsize=11,
                     fontweight="bold", color=task_colors[task])
        text_ax.annotate("", xy=(0.95, 0.57), xytext=(0.05, 0.57),
                         arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))
        text_ax.text(0.5, 0.46, record["instruction"], ha="center", va="top", fontsize=10.5, wrap=True)
        text_ax.text(0.5, 0.12, interpretation, ha="center", va="bottom",
                     fontsize=8.7, color="#666666", wrap=True)
        row_fig.savefig(OUT / f"story_{letter.lower()}_{task}.png", dpi=320, facecolor="white")
        row_fig.savefig(OUT / f"story_{letter.lower()}_{task}.pdf", facecolor="white")
        plt.close(row_fig)


if __name__ == "__main__":
    main()
