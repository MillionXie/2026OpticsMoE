"""Sparse rerun contract; never alter finalized B1 or the original workers."""
import ast
import os
import subprocess
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
    # The approved17Oct sequential wrapper changed the file, not measure_repeat.
    # This is the deployed pre-repair identity, also bound by the original4 points.
    assert m.frozen.sha256(m.Path(m.uniform.__file__))=='875001037eac159fcd540421149b22af185562bdef80a2b810289b81b0d31cbd'
    relative='LightGenV2/tasks/t06_video_quality_assessment/unified_resident_utilization.py'
    original=ast.parse(subprocess.check_output(['git','show','5f73f4e71:'+relative],cwd=m.REPO))
    current=ast.parse(m.Path(m.uniform.__file__).read_text())
    def loop(tree):
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measure_repeat'),include_attributes=False)
    assert loop(original)==loop(current)
    assert m.frozen.sha256(m.Path(m.costs.__file__))=='6222b7edbb7654fbfa11cad4c1579e673ef81070ea4fa095dfedd1c45d264ab4'


def test_compact_service_metrics_keep_all_repeats_and_two_weightings():
    from LightGenV2.tasks.t06_video_quality_assessment.reports.plot_selected_execution import compact_resident_metrics
    repeats=[dict(duration_s=d,calls=c,board_w=w,utilization_percent=u,samples=n)
             for d,c,w,u,n in [(30,10,100,90,60),(40,20,200,95,80),(50,30,300,100,90)]]
    result=compact_resident_metrics(repeats,2)
    assert result['time_ms']==16000
    assert result['board_w']==pytest.approx((3000+8000+15000)/120)
    assert result['utilization_percent']==pytest.approx((5400+7600+9000)/230)
    assert result['energy_j']==pytest.approx((338.2+26000/120)*16)
    with pytest.raises(AssertionError):compact_resident_metrics(repeats[:2],2)


def test_compact_cohort_and_symbols_exclude_tiny_small_rental_qwen():
    from LightGenV2.tasks.t06_video_quality_assessment.reports.plot_selected_execution import COMPACT_HIGH, COMPACT_MARKERS
    assert COMPACT_HIGH==dict(qwen=3,glm=2,clip=10,yolo=3,full=2,kimi=6)
    assert set(COMPACT_MARKERS)==set(COMPACT_HIGH)|{'ours'}
