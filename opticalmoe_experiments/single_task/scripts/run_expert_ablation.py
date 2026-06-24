import argparse
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common.utils.config import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = run_dir / "diagnostics" / "expert_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "expert_id,ablation_acc,notes\n"
        "all,not_run,placeholder for future expert masking ablation\n",
        encoding="utf-8",
    )
    save_json(
        [{"expert_id": "all", "ablation_acc": None, "notes": "placeholder"}],
        run_dir / "summary_for_master" / "expert_ablation_rows.json",
    )
    print(f"wrote placeholder expert ablation to {out}")


if __name__ == "__main__":
    main()
