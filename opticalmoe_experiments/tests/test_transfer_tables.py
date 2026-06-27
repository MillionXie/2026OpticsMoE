import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transfer_adaptation.scripts.transfer_utils import rebuild_transfer_tables


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_transfer_tables_are_built_from_summary_rows(tmp_path):
    summary = tmp_path / "runs" / "run_a" / "summary_for_master"
    _write(summary / "runs_rows.json", {"run_id": "run_a"})
    _write(summary / "epoch_metrics_rows.json", [{"run_id": "run_a", "epoch": 1}])
    _write(summary / "final_metrics_rows.json", [{"run_id": "run_a", "trainable_electronic_params": 0}])
    _write(summary / "prompt_swap_rows.json", [{"run_id": "run_a", "prompt_task": "usps"}])
    _write(summary / "source_retention_rows.json", [{"run_id": "run_a", "source_task": "mnist"}])
    _write(summary / "prompt_similarity_rows.json", [{"run_id": "run_a", "source_task": "mnist"}])
    _write(summary / "expert_usage_rows.json", [{"run_id": "run_a", "expert_id": "E00"}])
    _write(summary / "model_params_rows.json", [{"run_id": "run_a", "trainable_electronic_params": 0}])
    _write(summary / "scaling_rows.json", [{"run_id": "run_a", "trainable_electronic_params": 0}])
    out = tmp_path / "results"
    counts = rebuild_transfer_tables(tmp_path / "runs", out)
    assert counts["final_metrics"] == 1
    for name in [
        "master_transfer_runs.csv",
        "master_transfer_epoch_metrics.csv",
        "master_transfer_final_metrics.csv",
        "master_transfer_prompt_swap.csv",
        "master_transfer_source_retention.csv",
        "master_transfer_prompt_similarity.csv",
        "master_transfer_expert_usage.csv",
        "master_transfer_model_params.csv",
        "master_transfer_scaling.csv",
    ]:
        assert (out / name).exists()
    text = (out / "master_transfer_final_metrics.csv").read_text(encoding="utf-8")
    assert "trainable_electronic_params" in text
    assert "\nrun_a,0" in text or "\n0,run_a" in text or "run_a" in text

