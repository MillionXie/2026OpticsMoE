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
as_int = _COMMON.as_int
ensure_out_dir = _COMMON.ensure_out_dir
filter_rows = _COMMON.filter_rows
require_rows = _COMMON.require_rows
load_epoch_rows_from_args = _IO.load_epoch_rows_from_args
save_plot_data = _IO.save_plot_data


def _smooth(values, window):
    if window <= 1:
        return values
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(sum(chunk) / max(1, len(chunk)))
    return out


def _group_epoch_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("run_id", "run")].append(row)
    for run_id in grouped:
        grouped[run_id] = sorted(grouped[run_id], key=lambda row: as_int(row.get("epoch")))
    return grouped


def _metric_columns(metric, show):
    if metric == "acc":
        columns = []
        if "train" in show:
            columns.append(("train_acc", "train"))
        if "val" in show:
            columns.append(("val_acc", "val"))
        return columns, "Accuracy"
    if metric == "loss":
        columns = []
        if "train" in show:
            columns.append(("train_loss", "train"))
        if "val" in show:
            columns.append(("val_loss", "val"))
        return columns, "Loss"
    raise ValueError(f"Unsupported metric: {metric}")


def make_plot(args):
    rows = filter_rows(load_epoch_rows_from_args(args), args)
    require_rows(rows, "epoch metrics")
    out_dir = ensure_out_dir(args.out_dir)
    set_paper_style()
    palette = get_model_palette()
    grouped = _group_epoch_rows(rows)
    metrics = list(dict.fromkeys((getattr(args, "metrics", None) or []) + ([getattr(args, "metric")] if getattr(args, "metric", None) else [])))
    if not metrics:
        metrics = ["acc"]
    outputs = []
    for metric in metrics:
        columns, ylabel = _metric_columns(metric, args.show)
        plot_rows = []
        if args.mode == "grid":
            fig, axes = plt.subplots(len(grouped), 1, figsize=(args.width, max(args.height, 2.2 * len(grouped))), squeeze=False)
            axes = [ax[0] for ax in axes]
        else:
            fig, ax = plt.subplots(figsize=(args.width, args.height))
            axes = [ax] * len(grouped)
        for ax, (run_id, run_rows) in zip(axes, grouped.items()):
            first = run_rows[0]
            color = palette.get(first.get("model_type"), None)
            epochs = [as_int(row.get("epoch")) for row in run_rows]
            for column, split in columns:
                vals = [as_float(row.get(column)) for row in run_rows]
                vals = _smooth(vals, int(args.smooth))
                linestyle = "-" if split == "train" else "--"
                label = f"{run_id} {split}" if args.mode == "overlay" else split
                ax.plot(epochs, vals, linestyle=linestyle, color=color, label=label)
                for epoch, value in zip(epochs, vals):
                    plot_rows.append({"run_id": run_id, "epoch": epoch, "split": split, "metric": metric, "value": value})
            if args.show_best:
                best_row = max(run_rows, key=lambda row: as_float(row.get("val_acc"), -1.0))
                ax.axvline(as_int(best_row.get("epoch")), color="#666666", linestyle=":", linewidth=1.2)
            if args.show_phase_dropout:
                active_epochs = [as_int(row.get("epoch")) for row in run_rows if str(row.get("phase_dropout_active")).lower() in ("true", "1", "yes")]
                if active_epochs:
                    ax.axvline(min(active_epochs), color="#AA4499", linestyle=":", linewidth=1.2)
            style_axis(ax)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            if args.mode == "grid":
                ax.set_title(run_id)
            ax.legend(frameon=False)
        if args.mode == "overlay":
            axes[0].set_title(f"Training {ylabel}")
        fig.tight_layout()
        base_name = args.name or f"training_{'accuracy' if metric == 'acc' else 'loss'}_{args.mode}"
        if args.name:
            base_name = f"{args.name}_{'accuracy' if metric == 'acc' else 'loss'}_{args.mode}"
        out_base = out_dir / base_name
        save_figure(fig, out_base)
        plt.close(fig)
        save_plot_data(plot_rows, out_base.with_name(out_base.name + "_plot_data.csv"))
        outputs.append(str(out_base))
    return outputs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot train/validation curves from run directories or master tables.")
    add_common_args(parser)
    parser.add_argument("--metric", choices=["acc", "loss"], default=None)
    parser.add_argument("--metrics", nargs="*", choices=["acc", "loss"], default=["acc"])
    parser.add_argument("--show", nargs="*", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--smooth", type=int, default=0)
    parser.add_argument("--mode", choices=["overlay", "grid"], default="overlay")
    parser.add_argument("--show_best", action="store_true")
    parser.add_argument("--show_phase_dropout", action="store_true")
    args = parser.parse_args()
    make_plot(args)


if __name__ == "__main__":
    main()
