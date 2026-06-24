import argparse
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common.reporting.aggregate_results import rebuild_master_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="single_task/runs")
    parser.add_argument("--out_dir", default="single_task/results")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    if not runs_dir.is_absolute():
        runs_dir = EXPERIMENT_ROOT / runs_dir
    if not out_dir.is_absolute():
        out_dir = EXPERIMENT_ROOT / out_dir
    counts = rebuild_master_tables(runs_dir, out_dir)
    print(f"rebuilt master tables in {out_dir}")
    print(counts)


if __name__ == "__main__":
    main()

