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
    parser.add_argument("--name", default="task_accuracy")
    args = parser.parse_args()
    if args.run_dir:
        df = pd.read_csv(Path(args.run_dir) / "metrics" / "task_metrics.csv")
    else:
        df = pd.read_csv(Path(args.master_dir) / "master_task_metrics.csv")
        if args.run_id:
            df = df[df["run_id"] == args.run_id]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for task, sub in df.groupby("task_name"):
        ax.plot(sub["epoch"], sub["val_acc"], label=f"{task} val", linewidth=2)
        if "train_acc" in sub:
            ax.plot(sub["epoch"], sub["train_acc"], linestyle="--", label=f"{task} train", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=300)
    df.to_csv(out / f"{args.name}_plot_data.csv", index=False)


if __name__ == "__main__":
    main()
