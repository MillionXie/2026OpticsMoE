import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_feature_similarity(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    epochs = [int(row["epoch"]) for row in rows]
    ax.plot(epochs, [float(row["train_feature_cosine"]) for row in rows], label="Train", linewidth=2)
    ax.plot(epochs, [float(row["val_feature_cosine"]) for row in rows], label="Validation", linewidth=2)
    ax.set(xlabel="Epoch", ylabel="Cosine similarity", title="Student-teacher feature similarity", ylim=(-1, 1))
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.metrics, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    save_feature_similarity(rows, args.out)


if __name__ == "__main__":
    main()

