import csv
import importlib.util
import sys
from pathlib import Path

VIS_DIR = Path(__file__).resolve().parent
if str(VIS_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_DIR))
_COMMON_SPEC = importlib.util.spec_from_file_location("visualization_local_common", VIS_DIR / "common.py")
_COMMON = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON)

import matplotlib.pyplot as plt
import numpy as np

from style import save_figure, set_paper_style

add_common_args = _COMMON.add_common_args
ensure_out_dir = _COMMON.ensure_out_dir

def _load_confusion(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return np.zeros((0, 0), dtype=float)
    labels = [key for key in rows[0].keys() if key != "row"]
    data = []
    for row in rows:
        data.append([float(row.get(label, 0.0) or 0.0) for label in labels])
    return np.asarray(data, dtype=float)


def make_plot(args):
    set_paper_style()
    out_dir = ensure_out_dir(args.out_dir)
    run_dirs = [Path(path) for path in args.run_dirs]
    if not run_dirs:
        raise SystemExit("plot_confusion_matrix requires --run_dirs.")
    outputs = []
    for run_dir in run_dirs:
        path = run_dir / "metrics" / "confusion_matrix.csv"
        if not path.exists():
            print(f"warning: missing confusion matrix: {path}")
            continue
        matrix = _load_confusion(path)
        if matrix.size == 0:
            print(f"warning: empty confusion matrix: {path}")
            continue
        if args.normalize:
            denom = matrix.sum(axis=1, keepdims=True)
            matrix = matrix / np.maximum(denom, 1e-12)
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(run_dir.name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        if matrix.shape[0] <= 12:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}" if args.normalize else f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=7)
        fig.tight_layout()
        out_base = out_dir / (args.name or f"{run_dir.name}_confusion_matrix")
        if len(run_dirs) > 1 and args.name:
            out_base = out_dir / f"{args.name}_{run_dir.name}"
        save_figure(fig, out_base)
        plt.close(fig)
        outputs.append(str(out_base))
    return outputs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot confusion matrix from run_dir/metrics/confusion_matrix.csv.")
    add_common_args(parser)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    make_plot(args)


if __name__ == "__main__":
    main()
