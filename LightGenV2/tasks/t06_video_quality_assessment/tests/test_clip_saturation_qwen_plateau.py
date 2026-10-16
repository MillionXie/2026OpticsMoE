"""CPU-only protocol and trace analysis tests; no model or CUDA imports."""
import ast
from pathlib import Path
import subprocess
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import extend_clip_diagnose_qwen as m


def test_bounded_extension_preserves_old_contracts():
    assert m.config()['clip_candidates']==[11,12,13,14,15,16]
    assert max(m.resident.config()['batches']['clip'])==10
    assert sum(map(len,m.costs.config()['batches'].values()))==30
    c=m.extension_protocol('util')
    assert (c['warmup_s'],c['measurement_s'],c['queue_depth'],c['repeats'])==(5,30,8,3)
    assert c['batches']=={'clip':[11,12,13,14,15,16]}
    assert not c['cuda_graph']


@pytest.mark.parametrize('batch',range(11,17))
def test_cost_formula_tail_and_historical_anchor_unchanged(batch):
    plan=m.costs.plan(batch)
    assert [i for a,b in plan for i in range(a,b)]==list(range(558))
    assert sum(b-a for a,b in plan if b<=200)==200
    assert not any(a<200<b for a,b in plan)
    c=m.extension_protocol('cost',batch)
    assert (c['host_proxy_w'],c['explicit_warmup_forwards'])==(338.2,0)
    assert c['qwen_b1_anchor']['latency_16_videos_ms']==1188.3861464262009


def test_stop_bracket_is_only_last_two_new_batches():
    assert m.selected_cost_batches([{'batch':11}])==[11]
    assert m.selected_cost_batches([{'batch':11},{'batch':12},{'batch':13}])==[12,13]
    with pytest.raises(ValueError):m.selected_cost_batches([{'batch':13}])


def test_kernel_union_excludes_overlaps_and_does_not_claim_nvml():
    trace={'traceEvents':[{'ph':'X','cat':'kernel','ts':0,'dur':4},
        {'ph':'X','cat':'kernel','ts':2,'dur':4},{'ph':'X','cat':'kernel','ts':8,'dur':2},
        {'ph':'X','cat':'cuda_runtime','name':'cudaStreamSynchronize','dur':3}]}
    x=m.trace_summary(trace)
    assert x['kernel_union_us']==8
    assert x['kernel_gap_us']==2
    assert x['kernel_union_fraction']==.8
    assert x['cuda_api'][0]['calls']==1
    assert 'not NVML' in x['interpretation']


def test_reuses_measurement_workers_without_patching_them():
    text=Path(m.__file__).read_text()
    assert 'resident.worker(args,extension_protocol("util"))' in text
    assert 'costs.worker(args,extension_protocol("cost",args.batch))' in text
    calls={n.func.attr for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {'backward','CUDAGraph','train','set_device'}.intersection(calls)
    assert 'LOCK_EX | fcntl.LOCK_NB' in text and 'costs.wait_for_idle' in text


@pytest.mark.parametrize('module,function',[('unified_resident_utilization.py','measure_repeat'),('measure_frozen_batch_costs.py','worker')])
def test_historical_core_measurement_ast_is_unchanged(module,function):
    path=Path(m.__file__).parent/module
    rel=path.relative_to(m.REPO).as_posix()
    old=subprocess.check_output(['git','show','28f3ceb69582bd03da9c3ddf2168c759917e203b:'+rel],cwd=m.REPO,text=True)
    def core(text):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==function))
    assert core(old)==core(path.read_text())
