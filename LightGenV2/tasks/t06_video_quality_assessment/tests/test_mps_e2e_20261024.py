"""CPU contracts only; never allocate a CUDA context or load a model."""
from copy import deepcopy
from types import SimpleNamespace
import json
import math
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as m


def fixture():
    calls=[dict(sample_index=i,sample_id=f'video{i}',preprocess_start=10+i*6,
        prepare_end=11+i*6,model_call_start=12+i*6,first_block_start=13+i*6,
        gpu_output=14+i*6,call_return=15+i*6) for i in range(200)]
    samples=[SimpleNamespace(host_monotonic_s=8+i*.1,watts=50.) for i in range(12030)]
    return calls,samples


def test_continuous_wall_counts_input_frontend_return_and_all199_gaps():
    calls,power=fixture();value=m.summarize(calls,power)
    assert value['continuous_wall_s']==1199
    assert value['components']==dict(prepare_s=200,h2d_s=200,model_front_s=200,
        first_block_s=200,cpu_return_s=200,internal_gap_s=199)
    assert value['videos']==200 and value['frames']==800
    assert value['ms_per_video']==1199*5
    assert value['equivalent_ms_per_input_frame']==1199*1.25
    assert math.isclose(value['estimated_system_energy_j'],1199*388.2,rel_tol=1e-10)
    assert math.isclose(value['gpu_energy_j'],1199*50,rel_tol=1e-10)
    assert 'time16_ms' not in value and value['no_historical_cost_override']


def test_preparation_cannot_be_removed_without_changing_cost():
    calls,power=fixture();old=m.summarize(calls,power)
    calls[0]['preprocess_start']-=1
    new=m.summarize(calls,power)
    assert new['continuous_wall_s']==old['continuous_wall_s']+1
    assert math.isclose(new['estimated_system_energy_j']-old['estimated_system_energy_j'],388.2,abs_tol=1e-6)


def test_terminal_cpu_result_is_included_but_nothing_after_it():
    calls,power=fixture();old=m.summarize(calls,power)
    calls[-1]['call_return']+=.5
    new=m.summarize(calls,power)
    assert new['continuous_wall_s']-old['continuous_wall_s']==.5


@pytest.mark.parametrize('kind',['duplicate','order','nan','overlap'])
def test_rejects_bad_sample_identity_or_clocks(kind):
    calls,power=fixture()
    if kind=='duplicate':calls[1]['sample_id']=calls[0]['sample_id']
    elif kind=='order':calls[1]['sample_index']=9
    elif kind=='nan':calls[1]['model_call_start']=float('nan')
    else:calls[1]['preprocess_start']=calls[0]['call_return']-.5
    with pytest.raises(ValueError):m.summarize(calls,power)


def test_power_must_cover_prepare_not_only_gpu_compute():
    calls,power=fixture()
    with pytest.raises(ValueError):m.summarize(calls,[v for v in power if v.host_monotonic_s>=13])
    with pytest.raises(ValueError):m.summarize(calls,power[::4])


def test_plan_starts_with_shared_qwen100_three_fresh_workers():
    value=m.plan('lab')
    assert len(value)==24
    assert value[:3]==[('qwen',100,r) for r in (1,2,3)]
    assert value[3:6]==[('qwen',50,r) for r in (1,2,3)]
    assert len(m.plan('rental'))==12
    with pytest.raises(ValueError):m.plan('lab',['deepseek'])
    with pytest.raises(ValueError):m.plan('lab',caps=[25])
    with pytest.raises(ValueError):m.plan('lab',repeats=[1,1])


def test_immutable_output_requires_canonical_new_run(tmp_path,monkeypatch):
    allowed=tmp_path/'canonical';allowed.mkdir();monkeypatch.setattr(m,'CANONICAL',allowed)
    with pytest.raises(ValueError):m.validate_output(tmp_path/'elsewhere')
    with pytest.raises(FileExistsError):m.validate_output(allowed)
    assert m.validate_output(allowed/'point')==allowed/'point'


def test_collect_preserves_partial_records_and_never_runs_warmup():
    ticks=iter([0,1,2,5,6,7,8,11,12]);records=[];seen=[];checks=[]
    rows=[dict(sample_id=str(i),temporal=1) for i in range(3)]
    def prepare(selected):
        seen.append(selected[0]['sample_id'])
        if len(seen)==3:raise RuntimeError('simulated read failure')
        return None,[[1,2,3,4]]
    def forward(_):return [1.],1000,990
    first=iter([3,9])
    with pytest.raises(RuntimeError):
        m.collect(rows,prepare,lambda x:x,forward,lambda:next(first),lambda:None,
            lambda:checks.append(1),clock=lambda:next(ticks),records=records)
    assert len(records)==2 and seen==['0','1','2'] and len(checks)==3
    assert [v['prediction'] for v in records]==[1,1]


def test_reference_heads_use_full_not_tiny_deepseek():
    c=m.config()
    assert c['references']['deepseek']['head_sha256'].startswith('93e6f0de')
    assert set(c['references'])=={'qwen','glm','clip','yolo','deepseek','kimi'}
    assert c['no_historical_qwen_cost_override']


def test_input_bytes_are_not_prehashed_to_warm_cache(tmp_path,monkeypatch):
    path=tmp_path/'video.mp4';path.write_bytes(b'test video content')
    rows=[dict(sample_id='sample',video_path=str(path))];reads=[]
    original=m.sha
    def watched(path):reads.append(path);return original(path)
    monkeypatch.setattr(m,'sha',watched)
    before=m.file_identities(rows)
    assert not reads and 'sha256' not in before['sample']
    after=m.file_identities(rows,hash_content=True)
    assert reads==[path] and len(after['sample']['sha256'])==64
    assert {k:v for k,v in after['sample'].items() if k!='sha256'}==before['sample']


def test_cpu_metadata_records_full_shapes_grid_and_nonpadding_ids():
    import numpy as np
    inputs=dict(input_ids=np.array([[11,12,13,0]]),attention_mask=np.array([[1,1,1,0]]),
        pixel_values=np.zeros((4,3,224,224)),image_grid_thw=np.array([[1,16,16]]))
    value=m.cpu_input_evidence(inputs)
    assert value['effective_input_tokens']==3
    assert value['small_cpu_values']['input_ids']==[[11,12,13,0]]
    assert value['tensors']['pixel_values']['shape']==[4,3,224,224]
    assert value['small_cpu_values']['image_grid_thw']==[[1,16,16]]
    assert 'pixel_values' not in value['small_cpu_values']
    assert len(value['effective_input_ids_sha256'])==64


def test_input_evidence_refuses_cuda_without_running_any_tensor_method():
    class CUDAInput:
        shape=(1,4);dtype='int64';device=SimpleNamespace(type='cuda')
        def tolist(self):raise AssertionError('Never copy CUDA audit data')
    with pytest.raises(ValueError,match='CUDA'):m.cpu_input_evidence(dict(input_ids=CUDAInput()))
