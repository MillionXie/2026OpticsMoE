import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--master_dir", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", default="same_input_predictions")
    args = parser.parse_args()
    if args.run_dir:
        df = pd.read_csv(Path(args.run_dir) / "metrics" / "same_input_task_switching.csv")
    else:
        df = pd.read_csv(Path(args.master_dir) / "master_same_input_switching.csv")
        if args.run_id:
            df = df[df["run_id"] == args.run_id]
    summary = df.groupby("task_name")["correct"].mean().reset_index(name="accuracy")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(summary["task_name"], summary["accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Same-input task switching")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=300)
    summary.to_csv(out / f"{args.name}_plot_data.csv", index=False)


if __name__ == "__main__":
    main()
