import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_switching.scripts.train_dataset_switching import rebuild_dataset_switching_tables


def test_independent_baseline_is_not_upper_bound_in_summary(tmp_path):
    summary = tmp_path / "runs" / "independent" / "summary_for_master"
    summary.mkdir(parents=True)
    rows = [
        {
            "run_id": "independent",
            "task_name": "mnist",
            "is_upper_bound": False,
            "total_independent_params": 123,
        }
    ]
    (summary / "independent_baseline_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    counts = rebuild_dataset_switching_tables(tmp_path / "runs", tmp_path / "results")
    assert counts["independent_baseline"] == 1
    text = (tmp_path / "results" / "master_independent_baseline.csv").read_text(encoding="utf-8")
    assert "False" in text
