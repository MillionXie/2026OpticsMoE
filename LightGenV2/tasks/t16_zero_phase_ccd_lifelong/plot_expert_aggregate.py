"""Plot existing validation-only routing summaries; requires no model inference."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = ("eurosat", "clevr", "speech_binary", "physical_binary_raw")


def plot(validation, destination):
    mean = np.asarray([validation[t]["router"]["mean_power_by_slot"] for t in TASKS])
    largest = np.asarray([validation[t]["router"]["argmax_fraction_by_slot"]
                          for t in TASKS])
    figure, axes = plt.subplots(2, 1, figsize=(13, 5), constrained_layout=True)
    for axis, matrix, label in zip(axes, (mean, largest),
                                   ("Mean optical-power fraction",
                                    "Fraction with largest route weight")):
        image = axis.imshow(matrix, vmin=0, vmax=max(0.2, matrix.max()),
                            aspect="auto", cmap="viridis")
        axis.set_yticks(range(4), ("EuroSAT", "CLEVR", "Speech", "Physical"))
        axis.set_xticks(range(16), range(1, 17))
        axis.set_ylabel(label)
        figure.colorbar(image, ax=axis, fraction=0.02)
    axes[-1].set_xlabel("Physical expert slot")
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(result["validation"], args.out)


if __name__ == "__main__":
    main()
