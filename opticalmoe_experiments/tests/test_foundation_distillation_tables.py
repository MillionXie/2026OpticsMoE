import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation_distillation.scripts.build_distillation_tables import rebuild_distillation_tables


def test_distillation_master_tables_are_rebuilt(tmp_path):
    summary = tmp_path / "runs" / "run_a" / "summary_for_master"
    summary.mkdir(parents=True)
    (summary / "final_metrics_rows.json").write_text(json.dumps([{"run_id": "run_a", "final_test_acc": 0.5}]), encoding="utf-8")
    counts = rebuild_distillation_tables(tmp_path / "runs", tmp_path / "results")
    output = tmp_path / "results" / "master_distillation_final_metrics.csv"
    assert counts["final_metrics"] == 1
    assert output.is_file()
    assert "run_a" in output.read_text(encoding="utf-8")
