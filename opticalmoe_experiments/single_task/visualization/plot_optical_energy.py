import csv
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
ensure_out_dir = _COMMON.ensure_out_dir
filter_rows = _COMMON.filter_rows
load_master_table = _IO.load_master_table
resolve_runs_from_args = _IO.resolve_runs_from_args
save_plot_data = _IO.save_plot_data

STAGE_ORDER = [
    "input",
    "after_input_to_prompt",
    "after_prompt",
    "expert_entrance_before_aperture",
    "expert_entrance_after_aperture",
    "after_expert_layer_1",
    "after_expert_layer_last",
    "after_global_fc",
    "detector_plane",
]


def _read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_rows(args):
    rows = []
    for run_dir in resolve_runs_from_args(args):
        rows.extend(_read_csv(Path(run_dir) / "diagnostics" / "optical_energy_by_stage.csv"))
    if not rows and args.master_dir:
        rows = load_master_table(args.master_dir, "optical_energy")
    return filter_rows(rows, args)


def make_plot(args):
    rows = _load_rows(args)
    if not rows:
        print("No optical energy table found. Save optical energy diagnostics or rebuild master_optical_energy.csv first.")
        return None
    out_dir = ensure_out_dir(args.out_dir)
    set_paper_style()
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("run_id", "run")].append(row)
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    plot_rows = []
    for run_id, run_rows in groups.items():
        by_stage = {}
        for row in run_rows:
            by_stage.setdefault(row.get("stage", ""), []).append(row)
        xs, ys = [], []
        for idx, stage in enumerate(STAGE_ORDER):
            vals = [as_float(row.get(args.value)) for row in by_stage.get(stage, [])]
            if vals:
                xs.append(idx)
                ys.append(sum(vals) / len(vals))
                plot_rows.append({"run_id": run_id, "stage": stage, "value_name": args.value, "value": ys[-1]})
        ax.plot(xs, ys, marker="o", label=run_id)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, rotation=35, ha="right")
    ax.set_ylabel(args.value)
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    out_base = out_dir / (args.name or f"optical_energy_{args.value}")
    save_figure(fig, out_base)
    plt.close(fig)
    save_plot_data(plot_rows, out_base.with_name(out_base.name + "_plot_data.csv"))
    return str(out_base)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot optical energy diagnostics by propagation stage.")
    add_common_args(parser)
    parser.add_argument("--value", default="outside_expert_ratio")
    args = parser.parse_args()
    make_plot(args)


if __name__ == "__main__":
    main()
