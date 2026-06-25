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
    parser.add_argument("--value", default="normalized_power_cosine")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", default="prompt_similarity")
    args = parser.parse_args()
    if args.run_dir:
        df = pd.read_csv(Path(args.run_dir) / "diagnostics" / "prompt_similarity.csv")
    else:
        df = pd.read_csv(Path(args.master_dir) / "master_prompt_similarity.csv")
        if args.run_id:
            df = df[df["run_id"] == args.run_id]
    if "epoch" in df and len(df):
        df = df[df["epoch"] == df["epoch"].max()]
    tasks = sorted(set(df["task_a"]).union(set(df["task_b"])))
    table = pd.DataFrame(1.0, index=tasks, columns=tasks)
    for _, row in df.iterrows():
        table.loc[row["task_a"], row["task_b"]] = row[args.value]
        table.loc[row["task_b"], row["task_a"]] = row[args.value]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(table.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=300)
    table.to_csv(out / f"{args.name}_plot_data.csv")


if __name__ == "__main__":
    main()
