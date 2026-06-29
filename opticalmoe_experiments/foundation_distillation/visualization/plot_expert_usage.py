import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_expert_usage(rows, path):
    if not rows:
        return
    latest = max(int(row["epoch"]) for row in rows)
    selected = [row for row in rows if int(row["epoch"]) == latest]
    selected.sort(key=lambda row: int(row["expert_index"]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar([row["expert_id"] for row in selected], [float(row["normalized_prompt_power"]) for row in selected])
    ax.set(xlabel="Expert", ylabel="Normalized prompt power", title=f"Expert usage at epoch {latest}", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    save_expert_usage(rows, args.out)


if __name__ == "__main__":
    main()

