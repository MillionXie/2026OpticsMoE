import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [{key: float(value) if key != "epoch" else int(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def save_distillation_curves(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    axes[0].plot(epochs, [row["train_total_loss"] for row in rows], label="Train", linewidth=2)
    axes[0].plot(epochs, [row["val_total_loss"] for row in rows], label="Validation", linewidth=2)
    axes[0].set(xlabel="Epoch", ylabel="Weighted loss", title="Total loss")
    axes[1].plot(epochs, [row["train_acc"] for row in rows], label="Train", linewidth=2)
    axes[1].plot(epochs, [row["val_acc"] for row in rows], label="Validation", linewidth=2)
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Classification accuracy", ylim=(0, 1))
    axes[2].plot(epochs, [row["train_feature_cosine"] for row in rows], label="Train", linewidth=2)
    axes[2].plot(epochs, [row["val_feature_cosine"] for row in rows], label="Validation", linewidth=2)
    axes[2].set(xlabel="Epoch", ylabel="Cosine similarity", title="Teacher feature alignment", ylim=(-1, 1))
    for ax in axes:
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
    save_distillation_curves(load_rows(args.metrics), args.out)


if __name__ == "__main__":
    main()

