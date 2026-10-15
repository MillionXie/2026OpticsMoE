"""Metadata-only tests; do not load model assets or initialize CUDA."""
import ast
from pathlib import Path

import pytest

from LightGenV2.tasks.t06_video_quality_assessment import unified_resident_utilization as u


def test_same_fixed_method_for_six_models():
    c = u.config()
    assert (c['warmup_s'],c['measurement_s'],c['repeats'],c['queue_depth']) == (5,30,3,8)
    assert set(c['batches']) == {'qwen','clip','yolo','deepseek','glm','small'}
    assert not c['cuda_graph']
    assert c['historical_costs_and_metrics_unchanged']
    assert c['batches']['qwen'][:4] == [1,2,3,4]
    assert max(c['batches']['qwen']) == 10


def test_pooled_mean_is_not_maximum_or_unweighted_mean():
    rows=[{'measurement':{'mean_gpu_utilization_percent':a,'samples':n}}
          for a,n in [(40,100),(60,200),(100,100)]]
    assert u.pooled_mean(rows) == pytest.approx(65)


def test_sequential_lower_batch_retained_and_b1_not_stop():
    c=u.config()
    assert not u.stop_at_target(1,100,c)
    assert not u.stop_at_target(2,97.99,c)
    assert u.stop_at_target(2,98,c)


def test_measured_loop_has_no_cpu_output_training_transfer_or_file_writes():
    source=Path(u.__file__).read_text()
    phase=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='phase')
    calls={n.func.attr for n in ast.walk(phase) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {'cpu','to','tolist','backward','train','step','write_json'}.intersection(calls)
    assert 'minimum' in ast.unparse(phase)
    assert 'guard.check()' in ast.unparse(phase)


def test_no_unbounded_search_or_training_and_source_binding():
    source=Path(u.__file__).read_text()
    assert 'for batch in c["batches"][args.baseline]' in source
    assert 'no_quantization_or_offload=True' in source
    assert 'raise RuntimeError("Bound measurement source changed")' in source
    assert 'torch.cuda.OutOfMemoryError' in source
    assert 'max(means)-min(means)' in source
