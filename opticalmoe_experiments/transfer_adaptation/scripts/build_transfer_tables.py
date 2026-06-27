from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import transfer_utils as tu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="transfer_adaptation/runs")
    parser.add_argument("--out_dir", default="transfer_adaptation/results")
    args = parser.parse_args()
    runs_dir = tu.resolve_path(args.runs_dir, prefer_experiment_root=True)
    out_dir = tu.resolve_path(args.out_dir, prefer_experiment_root=True)
    counts = tu.rebuild_transfer_tables(runs_dir, out_dir)
    print(f"rebuilt transfer tables in {out_dir}: {counts}")


if __name__ == "__main__":
    main()

