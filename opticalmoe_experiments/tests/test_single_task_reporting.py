import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.reporting.aggregate_results import rebuild_master_tables
from common.utils.config import save_json


def test_rebuild_master_tables(tmp_path):
    summary = tmp_path / "runs" / "run_a" / "summary_for_master"
    summary.mkdir(parents=True)
    save_json({"run_id": "run_a"}, summary / "runs_rows.json")
    save_json([{"run_id": "run_a", "epoch": 1}], summary / "epoch_metrics_rows.json")
    counts = rebuild_master_tables(tmp_path / "runs", tmp_path / "results")
    assert counts["runs"] == 1
    assert (tmp_path / "results" / "master_runs.csv").exists()

