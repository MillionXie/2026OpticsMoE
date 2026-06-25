import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_dir", default="dataset_switching/results")
    parser.add_argument("--out_dir", default="dataset_switching/figures/independent_vs_moe")
    parser.add_argument("--name", default="independent_vs_moe")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    master = Path(args.master_dir)
    final = pd.read_csv(master / "master_final_metrics.csv") if (master / "master_final_metrics.csv").exists() else pd.DataFrame()
    indep = pd.read_csv(master / "master_independent_baseline.csv") if (master / "master_independent_baseline.csv").exists() else pd.DataFrame()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if not final.empty:
        rows.extend({"label": f"{r.run_id}:{r.task_name}", "acc": r.final_test_acc} for r in final.itertuples())
    if not indep.empty:
        rows.extend({"label": f"independent:{r.task_name}", "acc": r.test_acc} for r in indep.itertuples())
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No MoE or independent baseline rows found.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(df)), df["acc"].astype(float))
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=75, ha="right")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(out / f"{args.name}.png", dpi=200)
    df.to_csv(out / f"{args.name}_plot_data.csv", index=False)


if __name__ == "__main__":
    main()
