import importlib.util
import sys
from collections import defaultdict
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
as_int = _COMMON.as_int
ensure_out_dir = _COMMON.ensure_out_dir
filter_rows = _COMMON.filter_rows
require_rows = _COMMON.require_rows
load_expert_usage_rows_from_args = _IO.load_expert_usage_rows_from_args
save_plot_data = _IO.save_plot_data


def _select_epoch(rows, epoch_spec):
    if epoch_spec in ("final", "best"):
        max_epoch = max(as_int(row.get("epoch")) for row in rows)
        return [row for row in rows if as_int(row.get("epoch")) == max_epoch]
    epoch = int(epoch_spec)
    return [row for row in rows if as_int(row.get("epoch")) == epoch]


def make_plot(args):
    rows = filter_rows(load_expert_usage_rows_from_args(args), args)
    require_rows(rows, "expert usage")
    rows = _select_epoch(rows, args.epoch)
    require_rows(rows, f"expert usage epoch={args.epoch}")
    out_dir = ensure_out_dir(args.out_dir)
    set_paper_style()
    run_ids = sorted({row.get("run_id", "run") for row in rows})
    expert_ids = sorted({row.get("expert_id", "") for row in rows})
    matrix = np.zeros((len(run_ids), len(expert_ids)), dtype=float)
    index = {(row.get("run_id", "run"), row.get("expert_id", "")): row for row in rows}
    plot_rows = []
    for i, run_id in enumerate(run_ids):
        for j, expert_id in enumerate(expert_ids):
            value = as_float(index.get((run_id, expert_id), {}).get(args.value), 0.0)
            matrix[i, j] = value
            plot_rows.append({"run_id": run_id, "expert_id": expert_id, "value_name": args.value, "value": value})
    fig, ax = plt.subplots(figsize=(args.width, max(args.height, 0.45 * len(run_ids) + 1.8)))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(expert_ids)))
    ax.set_xticklabels(expert_ids, rotation=45, ha="right")
    ax.set_yticks(range(len(run_ids)))
    ax.set_yticklabels(run_ids)
    ax.set_title(args.value)
    fig.tight_layout()
    out_base = out_dir / (args.name or f"expert_usage_{args.value}")
    save_figure(fig, out_base)
    plt.close(fig)
    save_plot_data(plot_rows, out_base.with_name(out_base.name + "_plot_data.csv"))
    return str(out_base)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot expert usage heatmaps.")
    add_common_args(parser)
    parser.add_argument("--value", default="normalized_prompt_power")
    parser.add_argument("--epoch", default="final")
    args = parser.parse_args()
    make_plot(args)


if __name__ == "__main__":
    main()
