import sys
import importlib.util
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

from style import get_model_palette, save_figure, set_paper_style, style_axis

add_common_args = _COMMON.add_common_args
as_float = _COMMON.as_float
ensure_out_dir = _COMMON.ensure_out_dir
filter_rows = _COMMON.filter_rows
pretty_label = _COMMON.pretty_label
require_rows = _COMMON.require_rows
sort_rows = _COMMON.sort_rows
load_final_rows_from_args = _IO.load_final_rows_from_args
save_plot_data = _IO.save_plot_data


UNIT_SCALE = {"sec": 1.0, "min": 60.0, "hour": 3600.0}
UNIT_LABEL = {"sec": "seconds", "min": "minutes", "hour": "hours"}


def make_plot(args):
    rows = sort_rows(filter_rows(load_final_rows_from_args(args), args))
    require_rows(rows, "final metrics")
    out_dir = ensure_out_dir(args.out_dir)
    set_paper_style()
    palette = get_model_palette()
    scale = UNIT_SCALE[args.unit]
    metric = args.metric
    labels = [row.get("run_id") or pretty_label(row) for row in rows]
    values = [as_float(row.get(metric)) / scale for row in rows]
    plot_rows = []
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    colors = [palette.get(row.get("model_type"), "#4477AA") for row in rows]
    ax.bar(range(len(rows)), values, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(f"{metric} ({UNIT_LABEL[args.unit]})")
    ax.set_title("Training Time")
    style_axis(ax)
    fig.tight_layout()
    out_base = out_dir / (args.name or "training_time_bar")
    save_figure(fig, out_base)
    plt.close(fig)
    for row, value in zip(rows, values):
        plot_rows.append({"run_id": row.get("run_id"), "metric": metric, "unit": args.unit, "value": value})
    save_plot_data(plot_rows, out_base.with_name(out_base.name + "_plot_data.csv"))

    scatter_rows = []
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    for row in rows:
        x = as_float(row.get("total_train_time_sec")) / scale
        y = as_float(row.get("final_test_acc"))
        ax.scatter(x, y, s=45, color=palette.get(row.get("model_type"), "#4477AA"), label=row.get("model_type"))
        ax.annotate(row.get("run_id", ""), (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        scatter_rows.append({"run_id": row.get("run_id"), "total_train_time": x, "unit": args.unit, "final_test_acc": y})
    ax.set_xlabel(f"Total train time ({UNIT_LABEL[args.unit]})")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Accuracy vs Training Time")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False)
    style_axis(ax)
    fig.tight_layout()
    scatter_base = out_dir / ((args.name + "_accuracy_vs_time") if args.name else "accuracy_vs_training_time")
    save_figure(fig, scatter_base)
    plt.close(fig)
    save_plot_data(scatter_rows, scatter_base.with_name(scatter_base.name + "_plot_data.csv"))
    return str(out_base)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot training time comparisons.")
    add_common_args(parser)
    parser.add_argument("--metric", default="total_train_time_sec")
    parser.add_argument("--unit", choices=["sec", "min", "hour"], default="min")
    args = parser.parse_args()
    make_plot(args)


if __name__ == "__main__":
    main()
