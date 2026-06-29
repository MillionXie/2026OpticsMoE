import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "end_to_end_ce_baseline": "End-to-end CE",
    "feature_distillation": "CLIP feature distillation",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.master_csv, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("dataset_name") == args.dataset]
    if not rows:
        raise ValueError(f"No rows found for dataset={args.dataset!r}.")
    for row in rows:
        if not row.get("experiment_variant"):
            row["experiment_variant"] = "feature_distillation" if row.get("teacher_type") != "none" else "end_to_end_ce_baseline"
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_rows = [
        {
            "run_id": row["run_id"],
            "experiment_variant": row["experiment_variant"],
            "final_test_acc": float(row["final_test_acc"]),
        }
        for row in rows
    ]
    with open(path.with_name(path.stem + "_plot_data.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=plot_rows[0].keys())
        writer.writeheader()
        writer.writerows(plot_rows)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.2 * len(plot_rows)), 4.2))
    labels = [LABELS.get(row["experiment_variant"], row["experiment_variant"]) for row in plot_rows]
    values = [row["final_test_acc"] for row in plot_rows]
    ax.bar(range(len(values)), values, color=["#0072B2" if "distill" in row["experiment_variant"] else "#D55E00" for row in plot_rows])
    ax.set_xticks(range(len(labels)), labels=labels, rotation=15, ha="right")
    ax.set_ylabel("Final test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{args.dataset}: distillation vs end-to-end baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

