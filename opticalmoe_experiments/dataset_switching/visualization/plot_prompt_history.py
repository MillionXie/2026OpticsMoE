import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_dir", default="dataset_switching/results")
    parser.add_argument("--value", default="normalized_prompt_power")
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--out_dir", default="dataset_switching/figures/prompt_history")
    parser.add_argument("--name", default="prompt_history")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(Path(args.master_dir) / "master_expert_usage.csv")
    if args.task_name:
        df = df[df["task_name"] == args.task_name]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for (run_id, task_name, expert_id), group in df.groupby(["run_id", "task_name", "expert_id"]):
        ax.plot(group["epoch"], group[args.value], label=f"{run_id}:{task_name}:{expert_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(args.value)
    ax.set_title("Prompt history")
    if len(df["expert_id"].unique()) <= 9:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=200)
    df.to_csv(out / f"{args.name}_plot_data.csv", index=False)


if __name__ == "__main__":
    main()
