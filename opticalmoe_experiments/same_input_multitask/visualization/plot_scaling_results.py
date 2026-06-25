import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_dir", required=True)
    parser.add_argument("--model_type", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", default="scaling_results")
    args = parser.parse_args()
    df = pd.read_csv(Path(args.master_dir) / "master_scaling_results.csv")
    if args.model_type:
        df = df[df["model_type"] == args.model_type]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for model, sub in df.groupby("model_type"):
        ax.plot(sub["num_tasks"], sub["macro_final_test_acc"], marker="o", label=model)
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Macro final test accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=300)
    df.to_csv(out / f"{args.name}_plot_data.csv", index=False)


if __name__ == "__main__":
    main()
