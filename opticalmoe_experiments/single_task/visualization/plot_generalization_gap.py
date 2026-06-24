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

from style import get_model_palette, save_figure, set_paper_style, style_axis

add_common_args = _COMMON.add_common_args
as_float = _COMMON.as_float
ensure_out_dir = _COMMON.ensure_out_dir
filter_rows = _COMMON.filter_rows
pretty_label = _COMMON.pretty_label
require_rows = _COMMON.require_rows
sort_rows = _COMMON.sort_rows
load_final_rows_from_args = _IO.load_final_rows_from_args
load_epoch_rows_from_args = _IO.load_epoch_rows_from_args
save_plot_data = _IO.save_plot_data


def make_plot(args):
    out_dir = ensure_out_dir(args.out_dir)
    set_paper_style()
    palette = get_model_palette()
    final_rows = sort_rows(filter_rows(load_final_rows_from_args(args), args))
    if final_rows:
        labels = [row.get("run_id") or pretty_label(row) for row in final_rows]
        values = []
        plot_rows = []
        for row in final_rows:
            gap = as_float(row.get("generalization_gap"))
            if gap != gap:
                gap = as_float(row.get("train_acc_at_best")) - as_float(row.get("best_val_acc"))
            values.append(gap)
            plot_rows.append({"run_id": row.get("run_id"), "generalization_gap": gap})
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        colors = [palette.get(row.get("model_type"), "#4477AA") for row in final_rows]
        ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Train acc at best - best val acc")
        ax.set_title("Generalization Gap")
        style_axis(ax)
        fig.tight_layout()
        out_base = out_dir / (args.name or "generalization_gap")
        save_figure(fig, out_base)
        plt.close(fig)
        save_plot_data(plot_rows, out_base.with_name(out_base.name + "_plot_data.csv"))
        return str(out_base)

    epoch_rows = filter_rows(load_epoch_rows_from_args(args), args)
    require_rows(epoch_rows, "final or epoch metrics")
    grouped = {}
    for row in epoch_rows:
        grouped.setdefault(row.get("run_id", "run"), []).append(row)
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_rows = []
    for run_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(float(row.get("epoch", 0))))
        epochs = [int(float(row.get("epoch", 0))) for row in rows]
        gaps = [as_float(row.get("train_acc")) - as_float(row.get("val_acc")) for row in rows]
        ax.plot(epochs, gaps, label=run_id)
        for epoch, gap in zip(epochs, gaps):
            plot_rows.append({"run_id": run_id, "epoch": epoch, "generalization_gap": gap})
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train acc - val acc")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    out_base = out_dir / (args.name or "generalization_gap_by_epoch")
    save_figure(fig, out_base)
    plt.close(fig)
    save_plot_data(plot_rows, out_base.with_name(out_base.name + "_plot_data.csv"))
    return str(out_base)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot final or epoch-wise generalization gap.")
    add_common_args(parser)
    args = parser.parse_args()
    make_plot(args)


if __name__ == "__main__":
    main()
