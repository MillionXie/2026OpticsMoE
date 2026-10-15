"""CPU-only contracts; no model, assets or CUDA initialization."""
import ast
from pathlib import Path

import pytest

from LightGenV2.tasks.t06_video_quality_assessment import measure_resident_utilization as measure
from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample


def test_fixed_unified_protocol():
    assert (measure.WARMUP_S, measure.MEASUREMENT_S, measure.QUEUE_DEPTH) == (5, 30, 8)
    source = Path(measure.__file__).read_text()
    assert "cuda_graph=False, streams=1" in source
    assert "range(3)" in source and "LOCK_EX | fcntl.LOCK_NB" in source
    assert "frozen.preflight_files(args)" in source


def test_exact_window_includes_zero_and_excludes_end():
    samples = [Sample(0, i, "measured", 100, util, 5000)
               for i, util in enumerate([100, 0, 40, 60, 100])]
    report = measure.summarize(samples, 1, 4)
    assert report["samples"] == 3
    assert report["mean_gpu_utilization_percent"] == pytest.approx(100/3)
    assert report["peak_gpu_utilization_percent"] == 60
    assert report["duration_s"] == 3


def test_missing_samples_is_failure():
    with pytest.raises(RuntimeError):
        measure.summarize([], 0, 30)


def test_loop_has_no_transfer_or_training_and_closes_timer():
    source = Path(measure.__file__).read_text()
    tree = ast.parse(source)
    phase = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "phase")
    calls = {n.func.attr for n in ast.walk(phase)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not {"cpu", "tolist", "to", "backward", "step", "train", "write_json"}.intersection(calls)
    assert source.index("timer.close()") < source.index("compute = execution.make_gpu_forward")
    assert "final_delta > 0.0002" in source
    assert "historical_costs_unchanged=True" in source
