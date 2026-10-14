import ast
from pathlib import Path
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import diagnose_batch_layers as trace


def test_bounded_samples_and_cpu_reference_memory():
    assert trace.BATCHES==(1,2,4)
    assert trace.REFERENCE_BYTE_LIMIT==8*1024**3


def test_no_training_or_model_forward_patch():
    tree=ast.parse(Path(trace.__file__).read_text(encoding="utf-8"))
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {"backward","step","train"}.intersection(calls)
    assert not any(isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store) and n.attr=="forward" for n in ast.walk(tree))


def test_reference_stage_report_handles_discrete_routes():
    t=trace.Trace(); t.events=["v","gate"]
    t.statistics={"v":dict(kind="vision",max_abs=2.,square_sum=4.,reference_square_sum=16.,entries=4,changed_entries=1),
                  "gate":dict(kind="route",tokens=20,changed_token_routes=2)}
    assert t.report()["v"]["relative_l2"]==.5
    assert t.report()["gate"]["changed_token_routes"]==2


def test_suite_is_bounded_no_occupancy_queue_and_preserves_original_data():
    from LightGenV2.tasks.t06_video_quality_assessment import run_batch_layer_diagnostics as suite
    assert suite.CHILD_TIMEOUT_SECONDS==600
    tree=ast.parse(Path(suite.__file__).read_text(encoding="utf-8"))
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {"sleep","backward","train","cuda"}.intersection(calls)
    assert "do not retry blindly" in Path(suite.__file__).read_text(encoding="utf-8")
