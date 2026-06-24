import csv
import importlib.util
import sys
from pathlib import Path

VIS_DIR = Path(__file__).resolve().parent
if str(VIS_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_DIR))
_IO_SPEC = importlib.util.spec_from_file_location("visualization_local_io", VIS_DIR / "io.py")
_IO = importlib.util.module_from_spec(_IO_SPEC)
_IO_SPEC.loader.exec_module(_IO)
_COMMON_SPEC = importlib.util.spec_from_file_location("visualization_local_common", VIS_DIR / "common.py")
_COMMON = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON)

import matplotlib.pyplot as plt
import numpy as np

from style import save_figure, set_paper_style

add_common_args = _COMMON.add_common_args
as_float = _COMMON.as_float
ensure_out_dir = _COMMON.ensure_out_dir
filter_rows = _COMMON.filter_rows
load_master_table = _IO.load_master_table
resolve_runs_from_args = _IO.resolve_runs_from_args
save_plot_data = _IO.save_plot_data


def _read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_rows(args):
    rows = []
    for run_dir in resolve_runs_from_args(args):
        rows.extend(_read_csv(Path(run_dir) / "diagnostics" / "expert_ablation.csv"))
    if not rows and args.master_dir:
        rows = load_master_table(args.master_dir, "expert_ablation")
    return filter_rows(rows, args)


def make_plot(args):
    rows = _load_rows(args)
    if not rows:
        print("No expert ablation table found. Run single_task/scripts/run_expert_ablation.py first.")
        return None
    out_dir = ensure_out_dir(args.out_dir)
    set_paper_style()
    run_ids = sorted({row.get("run_id", "run") for row in rows})
    expert_ids = sorted({row.get("expert_id", "") for row in rows})
    value_key = args.value
    matrix = np.zeros((len(run_ids), len(expert_ids)), dtype=float)
    for i, run_id in enumerate(run_ids):
        for j, expert_id in enumerate(expert_ids):
            matches = [row for row in rows if row.get("run_id", "run") == run_id and row.get("expert_id", "") == expert_id]
            matrix[i, j] = as_float(matches[0].get(value_key), 0.0) if matches else 0.0
    fig, ax = plt.subplots(figsize=(args.width, max(args.height, 0.45 * len(run_ids) + 1.8)))
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(expert_ids)))
    ax.set_xticklabels(expert_ids, rotation=45, ha="right")
    ax.set_yticks(range(len(run_ids)))
    ax.set_yticklabels(run_ids)
    ax.set_title(value_key)
    fig.tight_layout()
    out_base = out_dir / (args.name or "expert_ablation")
    save_figure(fig, out_base)
    plt.close(fig)
    save_plot_data(rows, out_base.with_name(out_base.name + "_plot_data.csv"))
    return str(out_base)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot expert ablation heatmaps.")
    add_common_args(parser)
    parser.add_argument("--value", default="acc_drop")
    args = parser.parse_args()
    make_plot(args)


if __name__ == "__main__":
    main()
