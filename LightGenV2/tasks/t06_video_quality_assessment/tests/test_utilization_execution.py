"""CPU contracts for utilization controls; no CUDA/PT/video access."""
import ast
from pathlib import Path
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import audit_utilization_execution as audit
from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample


def test_fixed_workload_and_windows():
    assert audit.DURATION_S == 30 and audit.STEADY_START_S == 5
    assert audit.QUEUE_DEPTH == 8
    assert audit.CASES == (("qwen", 1), ("qwen", 8), ("clip", 8), ("clip", 10),
                           ("yolo", 3), ("yolo", 8), ("deepseek", 8))


def test_fixed_window_keeps_zeros_and_excludes_outside():
    samples = [Sample(0, 100+i, "audit", 50, value, 1000)
               for i, value in enumerate([0, 0, 50, 50, 90, 90, 90])]
    assert audit.window_summary(samples, 100, 6)["utilization_percent"]["mean"] == pytest.approx(280/6)
    assert audit.window_summary(samples, 100, 6, 5)["utilization_percent"]["mean"] == 90
    assert audit.window_summary(samples, 100, 6, 5)["samples"] == 1


def test_no_training_or_source_mutation():
    source = Path(audit.__file__).read_text()
    calls = {n.func.attr for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not {"backward", "step", "train", "float", "half"}.intersection(calls - {"float", "half"})
    assert '"replaces_historical_anchor": True' not in source
    assert '"cuda_guard"' not in source  # no patching/replacing the guard


def test_empty_window_is_missing_not_zero():
    assert audit.window_summary([], 0, 30)["utilization_percent"] is None


def test_queue_not_historical_boundary():
    source = Path(audit.__file__).read_text()
    assert "NOT historical first-block synchronous host latency" in source
    assert "No old cost replacement" in source
