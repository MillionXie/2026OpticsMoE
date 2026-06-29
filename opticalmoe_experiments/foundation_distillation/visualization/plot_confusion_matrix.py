import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["true_class"] for row in rows]
    matrix = np.asarray([[int(row[label]) for label in labels] for row in rows])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels=labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

