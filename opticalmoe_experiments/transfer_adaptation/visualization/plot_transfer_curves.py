from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = EXPERIMENT_ROOT / "transfer_adaptation" / "scripts"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import transfer_utils as tu


def _read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_dir = tu.resolve_path(args.run_dir, prefer_experiment_root=True)
    rows = _read_rows(run_dir / "metrics" / "epoch_metrics.csv")
    curve_rows = [
        {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "train_acc": float(row["train_acc"]),
            "val_acc": float(row["val_acc"]),
        }
        for row in rows
    ]
    out = Path(args.out) if args.out else run_dir / "figures" / "training_curves.png"
    tu.save_training_curves(curve_rows, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

