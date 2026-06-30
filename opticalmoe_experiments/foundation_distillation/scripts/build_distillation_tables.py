import argparse
import json
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.reporting.metrics_writer import write_rows


TABLES = {
    "runs": "master_distillation_runs.csv",
    "epoch_metrics": "master_distillation_epoch_metrics.csv",
    "final_metrics": "master_distillation_final_metrics.csv",
    "model_params": "master_distillation_model_params.csv",
    "feature_similarity": "master_distillation_feature_similarity.csv",
    "expert_usage": "master_distillation_expert_usage.csv",
}

TEACHER_FIELDS = (
    "teacher_type",
    "teacher_backend",
    "teacher_model_name",
    "feature_type",
    "teacher_feature_dim",
    "teacher_input_mode",
    "final_feature_cosine",
    "final_test_acc",
)


def rebuild_distillation_tables(runs_dir, out_dir):
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for key, filename in TABLES.items():
        rows = []
        for path in sorted(runs_dir.glob(f"*/summary_for_master/{key}_rows.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded_rows = payload if isinstance(payload, list) else [payload]
            for row in loaded_rows:
                if key in {"runs", "final_metrics", "model_params"}:
                    for field in TEACHER_FIELDS:
                        row.setdefault(field, "")
            rows.extend(loaded_rows)
        write_rows(out_dir / filename, rows)
        counts[key] = len(rows)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    print(rebuild_distillation_tables(args.runs_dir, args.out_dir))


if __name__ == "__main__":
    main()
