"""CPU admission tests only: no model assets or CUDA initialization."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from LightGenV2.tasks.t06_video_quality_assessment import measure_frozen_batch_costs as m


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 6, 8, 9, 10])
def test_exact_full_test_and_cost_boundary(batch):
    plan = m.plan(batch)
    assert [i for start, end in plan for i in range(start, end)] == list(range(558))
    assert sum(end-start for start, end in plan if end <= 200) == 200
    assert all(0 < end-start <= batch and not (start < 200 < end) for start, end in plan)
    assert plan[-1][1] == 558


def test_all_thirty_previously_measured_points_only():
    assert m.config()["batches"] == {
        "qwen": [1,2,3,4,6,8,10], "glm": [1,2], "small": [1,2,3,4],
        "clip": [1,2,3,4,6,8,9,10], "yolo": [1,2,3], "deepseek": [1,2,3,4,6,8]}


def test_qwen_b1_cost_is_exact_anchor_not_new_audit():
    result = m.cost_report("qwen", 1, {"latency_16_videos_ms": 999}, m.config())
    assert result["latency_16_videos_ms"] == 1188.3861464262009
    assert result["energy_16_videos_j"] == 462.1134010767563
    assert result["energy_per_video_j"] == 462.1134010767563/16
    assert "not_this_round" in result["provenance"]


@pytest.mark.parametrize("baseline", ["qwen", "glm", "small", "clip", "yolo", "deepseek"])
def test_fresh_worker_command_reuses_original_environments(baseline):
    args = SimpleNamespace(manifest=Path("manifest.csv"), deadline=123)
    cmd = m.child_command(baseline, 1, args, Path("out"))
    assert cmd[3].endswith("measure_frozen_batch_costs")
    assert "--telemetry" not in cmd  # No second resident experiment in cost run.
    assert cmd[cmd.index("--phase")+1] == "worker"
    if baseline not in ("qwen", "glm", "small"):
        assert "--reproduction-receipt" in cmd


def test_non_anchor_cost_is_matched_not_historical():
    result = m.cost_report("qwen", 2, {"latency_16_videos_ms": 100}, m.config())
    assert result["latency_16_videos_ms"] == 100
    assert result["provenance"] == "fresh_dataset_once_same_round_power_and_wall_time"


def test_tail_batches_not_ceil_sixteen_or_dropped():
    rows = [[i, end-start, True, 20.0] for i, (start, end) in enumerate(m.plan(9)) if end <= 200]
    assert rows[-1][1] == 2
    costs = m.telemetry.workload_cost(rows, 100)
    assert costs["latency_16_videos_ms"] == pytest.approx(23*20*16/200)
    assert costs["energy_16_videos_j"] == pytest.approx(438.2*23*20*16/200/1000)


def test_no_training_graph_or_warmup_and_original_timer():
    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not {"backward", "step", "train", "CUDAGraph", "capture_begin"}.intersection(calls)
    worker = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker"))
    assert "FirstBlockTimer(first)" in worker
    assert worker.index("timer.finish()") < worker.index("values.detach().cpu().tolist()")
    assert "frozen.load_runtime(args)" in worker
    assert "explicit_warmup_forwards=0" in worker
    assert "power['active_mean_w']" in worker


def test_bounded_serial_suite_no_reservation_or_retry():
    text = Path(m.__file__).read_text(encoding="utf-8")
    assert '"maximum_suite_seconds": 28800' in m.CONFIG.read_text()
    assert "timeout=min(c[\"maximum_point_seconds\"], args.deadline-time.time())" in text
    assert "for baseline, batches in c[\"batches\"].items()" in text
    assert "ForeignGuard" in text and "guard.check()" in text
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in text
    assert "range(3)" in text


def test_source_and_pt_guard_not_later_git_head():
    bound = m.sources()
    assert len(bound) >= 16
    assert m.CHECKPOINTS["qwen"] == m.config()["qwen_b1_anchor"]["checkpoint_sha256"]
    assert len(set(m.CHECKPOINTS.values())) == 6
