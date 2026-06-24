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

from style import save_figure, set_paper_style, style_axis

add_common_args = _COMMON.add_common_args
as_float = _COMMON.as_float
as_int = _COMMON.as_int
ensure_out_dir = _COMMON.ensure_out_dir
filter_rows = _COMMON.filter_rows
require_rows = _COMMON.require_rows
load_expert_usage_rows_from_args = _IO.load_expert_usage_rows_from_args
save_plot_data = _IO.save_plot_data


def make_plot(args):
    rows = filter_rows(load_expert_usage_rows_from_args(args), args)
    require_rows(rows, "expert usage")
    out_dir = ensure_out_dir(args.out_dir)
    set_paper_style()
    run_ids = sorted({row.get("run_id", "run") for row in rows})
    plot_rows = []
    outputs = []
    for run_id in run_ids:
        run_rows = [row for row in rows if row.get("run_id", "run") == run_id]
        expert_ids = sorted({row.get("expert_id", "") for row in run_rows})
        if args.top_k > 0:
            latest_epoch = max(as_int(row.get("epoch")) for row in run_rows)
            latest = [row for row in run_rows if as_int(row.get("epoch")) == latest_epoch]
            expert_ids = [
                row.get("expert_id", "")
                for row in sorted(latest, key=lambda row: as_float(row.get(args.value)), reverse=True)[: args.top_k]
            ]
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        for expert_id in expert_ids:
            series = sorted([row for row in run_rows if row.get("expert_id") == expert_id], key=lambda row: as_int(row.get("epoch")))
            epochs = [as_int(row.get("epoch")) for row in series]
            values = [as_float(row.get(args.value)) for row in series]
            ax.plot(epochs, values, label=expert_id)
            for epoch, value in zip(epochs, values):
                plot_rows.append({"run_id": run_id, "expert_id": expert_id, "epoch": epoch, "value_name": args.value, "value": value})
        ax.set_xlabel("Epoch")
        ax.set_ylabel(args.value)
        ax.set_title(run_id)
        ax.legend(frameon=False, ncol=3)
        style_axis(ax)
        fig.tight_layout()
        out_base = out_dir / ((args.name + f"_{run_id}") if args.name and len(run_ids) > 1 else (args.name or f"{run_id}_prompt_history"))
        save_figure(fig, out_base)
        plt.close(fig)
        outputs.append(str(out_base))
    save_plot_data(plot_rows, out_dir / ((args.name or "prompt_history") + "_plot_data.csv"))
    return outputs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot prompt amplitude or normalized power history.")
    add_common_args(parser)
    parser.add_argument("--value", default="normalized_prompt_power")
    parser.add_argument("--top_k", type=int, default=0)
    args = parser.parse_args()
    make_plot(args)


if __name__ == "__main__":
    main()
