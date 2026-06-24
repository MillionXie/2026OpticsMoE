import sys
from collections import defaultdict
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


def make_plot(args):
    rows = sort_rows(filter_rows(load_final_rows_from_args(args), args))
    require_rows(rows, "final metrics")
    out_dir = ensure_out_dir(args.out_dir)
    set_paper_style()
    palette = get_model_palette()
    metric = args.metric
    plot_rows = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(args.x, "")].append(row)
    labels = list(grouped.keys())
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    if args.hue:
        hue_values = sorted({row.get(args.hue, "") for row in rows})
        width = 0.8 / max(1, len(hue_values))
        for hidx, hue in enumerate(hue_values):
            xs, ys = [], []
            for idx, label in enumerate(labels):
                matches = [row for row in grouped[label] if row.get(args.hue, "") == hue]
                value = sum(as_float(row.get(metric)) for row in matches) / max(1, len(matches)) if matches else float("nan")
                xs.append(idx + (hidx - (len(hue_values) - 1) / 2) * width)
                ys.append(value)
                plot_rows.append({args.x: label, args.hue: hue, "metric": metric, "value": value})
            ax.bar(xs, ys, width=width, label=hue, color=palette.get(hue))
    else:
        values = []
        colors = []
        for label in labels:
            values_here = [as_float(row.get(metric)) for row in grouped[label]]
            value = sum(values_here) / max(1, len(values_here))
            values.append(value)
            row0 = grouped[label][0]
            colors.append(palette.get(row0.get("model_type"), "#4477AA"))
            plot_rows.append({args.x: label, "metric": metric, "value": value})
        ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(args.title or metric)
    if args.hue:
        ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    out_base = out_dir / (args.name or f"{metric}_comparison")
    save_figure(fig, out_base)
    plt.close(fig)
    save_plot_data(plot_rows, out_base.with_name(out_base.name + "_plot_data.csv"))
    return str(out_base)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot final metric comparisons.")
    add_common_args(parser)
    parser.add_argument("--metric", default="final_test_acc")
    parser.add_argument("--x", default="model_type")
    parser.add_argument("--hue", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    make_plot(args)


if __name__ == "__main__":
    main()
