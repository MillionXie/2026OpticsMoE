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
    parser.add_argument("--name", default="prompt_swap_matrix")
    args = parser.parse_args()
    if args.run_dir:
        df = pd.read_csv(Path(args.run_dir) / "metrics" / "prompt_swap_matrix.csv")
    else:
        df = pd.read_csv(Path(args.master_dir) / "master_prompt_swap.csv")
        if args.run_id:
            df = df[df["run_id"] == args.run_id]
    table = df.pivot_table(index="readout_task", columns="prompt_task", values="accuracy", aggfunc="mean")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(table.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Prompt task")
    ax.set_ylabel("Readout task")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=300)
    table.to_csv(out / f"{args.name}_plot_data.csv")


if __name__ == "__main__":
    main()
