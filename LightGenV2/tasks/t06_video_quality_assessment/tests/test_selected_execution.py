"""Sparse rerun contract; never alter finalized B1 or the original workers."""
import ast
import os
from types import SimpleNamespace
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import measure_selected_execution as m


def test_selected_points_and_boundaries():
    c=m.config()
    assert sum(map(len,c['batches'].values()))==16
    assert c['batches']['qwen']==[3,4] and all(1 in v for k,v in c['batches'].items() if k!='qwen')
    assert c['maximum_suite_seconds']==10800 and c['maximum_stage_seconds']==1800
    assert c['utilization_is_independent_of_dataset_once_cost']
    for batch in [1,2,3,4,6,8,9,10]:
        plan=m.costs.plan(batch)
        assert sum(end-start for start,end in plan)==558
        assert sum(end-start for start,end in plan if end<=200)==200
        assert all(not (start<200<end) for start,end in plan)


def test_no_monkeypatch_graph_training_or_Qwen_B1():
    text=m.Path(m.__file__).read_text();tree=ast.parse(text)
    calls={x.func.attr for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)}
    assert not {'CUDAGraph','compile','backward','step','load_state_dict','set_device'}.intersection(calls)
    assert 'uniform.measure_repeat' in text and 'costs.worker(args,costs.config())' in text
    assert 'uniform.worker(args,c)' in text and 'core.BoundaryTimer(SimpleNamespace(model=adapter))' in text
    assert 'provisional_until_separate_full558_strict_validation=True' in text
    assert 'os.killpg(p.pid' in text and 'no automatic retry' in text
    with pytest.raises(ValueError,match='protected Qwen B1'):
        m.require_point(SimpleNamespace(baseline='qwen',batch=1))


def test_original_measurement_source_unchanged():
    assert m.frozen.sha256(m.Path(m.uniform.__file__))=='fbaca669c7e92f45d16694210b52f7d82ef3931315be7fe103e84b2ce60792f2'
    assert m.frozen.sha256(m.Path(m.costs.__file__))=='6222b7edbb7654fbfa11cad4c1579e673ef81070ea4fa095dfedd1c45d264ab4'
