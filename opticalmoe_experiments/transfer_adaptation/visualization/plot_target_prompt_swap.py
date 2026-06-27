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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_dir = tu.resolve_path(args.run_dir, prefer_experiment_root=True)
    with open(run_dir / "metrics" / "target_prompt_swap.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = Path(args.out) if args.out else run_dir / "figures" / "target_prompt_swap.png"
    tu.save_target_prompt_swap_plot(rows, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

