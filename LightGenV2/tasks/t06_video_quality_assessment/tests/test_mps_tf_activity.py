from copy import deepcopy
import pytest
from LightGenV2.tasks.t06_video_quality_assessment.diagnose_mps_tf_activity import union,duration,summarize_trace,configuration


def evidence():
    trace=dict(traceEvents=[dict(ph='X',cat='user_annotation',name='a',ts=100,dur=2),
        dict(ph='X',cat='kernel',ts=108,dur=5),dict(ph='X',cat='kernel',ts=110,dur=5),
        dict(ph='X',cat='kernel',ts=120,dur=5),dict(ph='X',cat='kernel',ts=135,dur=5)])
    anchors=[dict(name='a',before_us=0,after_us=2)]
    calls=[dict(sample_index=0,sample_id='zero',first_block_start=10e-6,gpu_output=30e-6)]
    return trace,calls,anchors


def test_overlap_union_not_sum_or_first_to_last_envelope():
    assert union([(3,5),(1,4),(9,10)])==[[1,5],[9,10]]
    assert duration([(3,5),(1,4),(9,10)])==5


def test_native_boundary_clips_preblock_postoutput_and_idle():
    trace,calls,anchors=evidence();before=deepcopy(trace)
    r=summarize_trace(trace,calls,anchors)
    assert trace==before and r['conditional_kernel_busy_percent']==50
    assert r['target50_reached'] and not r['target100_reached']
    assert r['kernel_union_us']==10 and r['tf_to_output_us']==20
    assert not r['nvml_utilization_equivalent'] and r['profiled_diagnostic_not_formal_cost']


def test_missing_kernel_or_bad_clock_rejected():
    trace,calls,anchors=evidence();trace['traceEvents']=trace['traceEvents'][:1]
    with pytest.raises(RuntimeError,match='No CUPTI'):summarize_trace(trace,calls,anchors)
    trace,calls,anchors=evidence();anchors[0]['after_us']=-1
    with pytest.raises(RuntimeError,match='Clock'):summarize_trace(trace,calls,anchors)


def test_pinned_finite_budget_and_no_rental_or_protocol_replacement():
    c=configuration()
    assert c['caps']==[100,50,25,10] and c['videos']==12 and c['batch']==1
    assert c['no_new_rental_or_ds_kimi_measurement'] and c['no_pacing_or_synthetic_load']


def test_cuda_only_clock_without_inventing_cpu_trace_events():
    trace,calls,_=evidence();trace['baseTimeNanoseconds']=1_000_000_000
    trace['traceEvents']=trace['traceEvents'][1:]
    anchors=[dict(before_us=0,after_us=2,wall_ns=1_000_101_000)]
    r=summarize_trace(trace,calls,anchors,0)
    assert r['conditional_kernel_busy_percent']==50 and r['trace_mode']=='cuda_only'
    assert r['clock_bound_is_empirical']
