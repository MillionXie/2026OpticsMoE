import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_dir", default="dataset_switching/results")
    parser.add_argument("--out_dir", default="dataset_switching/figures/task_accuracy")
    parser.add_argument("--name", default="task_accuracy")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    master = Path(args.master_dir)
    df = pd.read_csv(master / "master_final_metrics.csv")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = df["run_id"].astype(str) + ":" + df["task_name"].astype(str)
    ax.bar(range(len(df)), df["final_test_acc"].astype(float))
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Dataset-switching task accuracy")
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=200)
    df.to_csv(out / f"{args.name}_plot_data.csv", index=False)


if __name__ == "__main__":
    main()
