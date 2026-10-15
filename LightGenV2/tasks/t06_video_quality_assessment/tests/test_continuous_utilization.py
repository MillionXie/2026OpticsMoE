"""CPU contracts for the user-approved old continuous protocol."""
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import run_continuous_utilization_suite as suite
from LightGenV2.tasks.t06_video_quality_assessment import audit_utilization_execution as audit


def test_fixed_phases_and_bounded_small_batches():
    assert audit.MODES == ("legacy_sync_cpu", "sync_gpu_output", "queued_gpu_output")
    assert (audit.DURATION_S, audit.STEADY_START_S, audit.QUEUE_DEPTH) == (30, 5, 8)
    assert suite.INITIAL == dict(qwen=8, clip=9, yolo=3, deepseek=4)
    assert max(suite.LARGER.values()) == 10
    assert suite.MAX_GROUPS == 16


def test_no_unbounded_search_or_repeated_points():
    assert suite.next_batch("qwen", {}) == 8
    assert suite.next_batch("qwen", {8: 92}) == 10
    assert suite.next_batch("qwen", {8: 92, 10: 95}) is None
    assert suite.next_batch("yolo", {3: 100}) == 2
    assert suite.next_batch("yolo", {3: 100, 2: 85}) is None
    assert suite.next_batch("deepseek", {4: 90, 8: 100}) == 7
    assert suite.next_batch("deepseek", {4: 90, 8: 100, 7: 98}) is None


def test_no_special_qwen_cost_audit_for_uniform_utilization():
    source = Path(audit.__file__).read_text()
    assert 'not getattr(args, "utilization_only", False)' in source
    assert '"preceding_forwards": 200 if b1_cost_audit else 0' in source
    suite_source = Path(suite.__file__).read_text()
    assert '"--utilization-only"' in suite_source
    assert 'report["modes"][2]["windows"][1]' in suite_source
    assert 'LOCK_EX | fcntl.LOCK_NB' in suite_source
