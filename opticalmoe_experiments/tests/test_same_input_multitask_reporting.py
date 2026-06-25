import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from same_input_multitask.scripts.train_same_input_multitask import rebuild_same_input_tables


def test_same_input_master_table_rebuild(tmp_path):
    run = tmp_path / "runs" / "run_a" / "summary_for_master"
    run.mkdir(parents=True)
    payloads = {
        "runs_rows": {"run_id": "run_a", "model_type": "learnable_route_moe"},
        "epoch_metrics_rows": [{"run_id": "run_a", "epoch": 1, "macro_val_acc": 0.5}],
        "task_metrics_rows": [{"run_id": "run_a", "task_name": "shape", "val_acc": 0.5}],
        "final_metrics_rows": [{"run_id": "run_a", "task_name": "shape", "final_test_acc": 0.5}],
        "same_input_switching_rows": [{"run_id": "run_a", "task_name": "shape", "correct": True}],
        "prompt_swap_rows": [{"run_id": "run_a", "eval_task": "shape", "prompt_task": "scale", "accuracy": 0.4}],
        "expert_usage_rows": [{"run_id": "run_a", "task_name": "shape", "expert_id": "E00", "normalized_prompt_power": 0.1}],
        "prompt_similarity_rows": [{"run_id": "run_a", "task_a": "shape", "task_b": "scale", "normalized_power_cosine": 0.9}],
        "model_params_rows": [{"run_id": "run_a", "total_parameter_count": 10}],
        "scaling_results_rows": [{"run_id": "run_a", "num_tasks": 2, "macro_final_test_acc": 0.5}],
    }
    for name, payload in payloads.items():
        (run / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    counts = rebuild_same_input_tables(tmp_path / "runs", tmp_path / "results")
    assert counts["same_input_switching"] == 1
    assert counts["prompt_swap"] == 1
    assert counts["scaling_results"] == 1
    assert not pd.read_csv(tmp_path / "results" / "master_prompt_swap.csv").empty
