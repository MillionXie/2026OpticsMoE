import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.reporting.aggregate_results import rebuild_master_tables
from common.utils.config import save_json


TIME_FIELDS = [
    "total_wall_time_sec",
    "total_epoch_time_sec",
    "total_train_time_sec",
    "total_val_time_sec",
    "total_artifact_time_sec",
    "avg_epoch_time_sec",
    "avg_train_time_per_epoch_sec",
    "time_to_best_epoch_sec",
]


def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_training_time_fields_rebuild_into_master_tables(tmp_path):
    summary = tmp_path / "runs" / "run_a" / "summary_for_master"
    summary.mkdir(parents=True)
    row = {
        "run_id": "run_a",
        "dataset_name": "mnist",
        "model_type": "lenet5",
        "final_test_acc": 0.5,
        "status": "completed",
        "completed_at": "2026-06-24T00:00:00+0800",
    }
    for idx, field in enumerate(TIME_FIELDS, start=1):
        row[field] = float(idx)
    save_json(row, summary / "runs_rows.json")
    save_json([row], summary / "final_metrics_rows.json")

    rebuild_master_tables(tmp_path / "runs", tmp_path / "results")
    final_rows = _read_csv(tmp_path / "results" / "master_final_metrics.csv")
    run_rows = _read_csv(tmp_path / "results" / "master_runs.csv")
    assert final_rows and run_rows
    for field in TIME_FIELDS:
        assert field in final_rows[0]
    assert "total_wall_time_sec" in run_rows[0]
    assert "total_train_time_sec" in run_rows[0]
    assert "total_artifact_time_sec" in run_rows[0]
    assert run_rows[0]["status"] == "completed"

