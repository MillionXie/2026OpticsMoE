import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_switching.scripts.train_dataset_switching import rebuild_dataset_switching_tables


def test_dataset_switching_master_tables_rebuild(tmp_path):
    summary = tmp_path / "runs" / "run_a" / "summary_for_master"
    summary.mkdir(parents=True)
    (summary / "runs_rows.json").write_text(json.dumps({"run_id": "run_a"}), encoding="utf-8")
    (summary / "prompt_swap_rows.json").write_text(json.dumps([{"run_id": "run_a", "eval_dataset": "mnist"}]), encoding="utf-8")
    (summary / "expert_usage_rows.json").write_text(json.dumps([{"run_id": "run_a", "expert_id": "E00"}]), encoding="utf-8")
    counts = rebuild_dataset_switching_tables(tmp_path / "runs", tmp_path / "results")
    assert counts["runs"] == 1
    assert counts["prompt_swap"] == 1
    assert counts["expert_usage"] == 1
    assert (tmp_path / "results" / "master_prompt_swap.csv").exists()
