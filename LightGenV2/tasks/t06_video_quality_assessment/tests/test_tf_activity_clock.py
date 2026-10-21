import pytest
from LightGenV2.tasks.t06_video_quality_assessment.tf_activity_clock import epoch_bounds,calibration_error


def test_epoch_correlation_with_declared_kineto_base():
    trace=dict(baseTimeNanoseconds=1_000_000_000,traceEvents=[dict(ph='X',name='a',ts=100,dur=2)])
    anchors=[dict(name='a',before_us=0,after_us=2,wall_ns=1_000_101_000)]
    assert epoch_bounds(trace,anchors)==(99.,101.)
    assert calibration_error(trace,anchors)==1


def test_disjoint_clock_brackets_fail_instead_of_moving_gpu_events():
    trace=dict(baseTimeNanoseconds=0)
    anchors=[dict(before_us=0,after_us=1,wall_ns=1000),dict(before_us=0,after_us=1,wall_ns=100000)]
    with pytest.raises(RuntimeError,match='Inconsistent'):epoch_bounds(trace,anchors)
