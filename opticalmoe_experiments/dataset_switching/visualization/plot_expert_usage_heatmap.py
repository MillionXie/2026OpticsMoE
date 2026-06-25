import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_dir", default="dataset_switching/results")
    parser.add_argument("--value", default="normalized_prompt_power")
    parser.add_argument("--epoch", default="final")
    parser.add_argument("--out_dir", default="dataset_switching/figures/expert_usage")
    parser.add_argument("--name", default="expert_usage")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(Path(args.master_dir) / "master_expert_usage.csv")
    if args.epoch != "final":
        df = df[df["epoch"] == int(args.epoch)]
    else:
        df = df[df["epoch"] == df["epoch"].max()]
    pivot = df.pivot_table(index=["run_id", "task_name"], columns="expert_id", values=args.value, aggfunc="mean")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(pivot))))
    im = ax.imshow(pivot.astype(float).values, cmap="magma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([":".join(map(str, idx)) for idx in pivot.index])
    ax.set_title(args.value)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=200)
    pivot.to_csv(out / f"{args.name}_plot_data.csv")


if __name__ == "__main__":
    main()
