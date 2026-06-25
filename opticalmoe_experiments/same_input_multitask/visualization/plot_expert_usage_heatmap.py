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
    parser.add_argument("--value", default="normalized_prompt_power")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", default="expert_usage_heatmap")
    args = parser.parse_args()
    if args.run_dir:
        df = pd.read_csv(Path(args.run_dir) / "diagnostics" / "task_expert_energy_history.csv")
    else:
        df = pd.read_csv(Path(args.master_dir) / "master_expert_usage.csv")
        if args.run_id:
            df = df[df["run_id"] == args.run_id]
    df = df[df[args.value].notna()]
    if "epoch" in df and len(df):
        df = df[df["epoch"] == df["epoch"].max()]
    table = df.pivot_table(index="task_name", columns="expert_id", values=args.value, aggfunc="mean")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(table.values, cmap="magma")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_title(args.value)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=300)
    table.to_csv(out / f"{args.name}_plot_data.csv")


if __name__ == "__main__":
    main()
