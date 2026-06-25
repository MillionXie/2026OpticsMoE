import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--master_dir", default="dataset_switching/results")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--out_dir", default="dataset_switching/figures/prompt_swap")
    parser.add_argument("--name", default="prompt_swap_matrix")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.run_dir:
        df = pd.read_csv(Path(args.run_dir) / "metrics" / "prompt_swap_matrix.csv")
    else:
        df = pd.read_csv(Path(args.master_dir) / "master_prompt_swap.csv")
        if args.run_id:
            df = df[df["run_id"] == args.run_id]
    matched = df[df["label_space_matched"].astype(str).str.lower().isin(["true", "1"])]
    tasks = sorted(matched["eval_dataset"].unique())
    prompts = sorted(matched["prompt_task"].unique())
    matrix = matched[matched["eval_dataset"] == matched["readout_task"]].pivot_table(index="eval_dataset", columns="prompt_task", values="accuracy", aggfunc="mean").reindex(index=tasks, columns=prompts)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix.astype(float).values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompts, rotation=45, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title("Same-readout prompt swap accuracy")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=200)
    matrix.to_csv(out / f"{args.name}_plot_data.csv")


if __name__ == "__main__":
    main()
