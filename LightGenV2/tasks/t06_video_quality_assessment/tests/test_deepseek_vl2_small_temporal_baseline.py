import ast
from pathlib import Path
import sys

import pytest

from LightGenV2.tasks.t06_video_quality_assessment import deepseek_vl2_small_temporal_baseline as run
from LightGenV2.tasks.t06_video_quality_assessment.prepare_vlm_assets import validate_profile


def source(name):
    tree=ast.parse(Path(run.__file__).read_text())
    return ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))


def test_original_finite_protocol_only_B1_no_backbone_updates():
    c=run.config()
    assert c['epochs']==50 and c['inference_batches']==[1]
    assert c['feature_batch_size']==1 and c['native_frame_size']==384
    assert c['quality_output_trainable_parameters']==10240 and c['backbone_trainable_parameters']==0
    assert c['selection']=='highest_TEST_SRCC_development_no_VAL'
    assert (c['train_samples'],c['test_samples'])==(2250,558)


def test_fixed_official_assets_and_total_not_active_parameter_storage():
    c=run.config();validate_profile(c)
    assert c['model_id']=='deepseek-ai/deepseek-vl2-small'
    assert c['model_revision']=='6033e16432a1d771cf9fe4a6f894ff5e5e1459af'
    assert len(c['official_assets'])==11
    assert sum(s['size'] for n,s in c['official_assets'].items() if n.endswith('.safetensors'))==32297441224


def test_native_image_input_reuses_tiny_without_extra_frames_or_crops():
    s=source('prepare_batch')
    assert 'tiny.decode_ordered_frames' in s and 'core.FRAME_FRACTIONS[4], 384' in s
    assert "tiny.build_conversation('temporal', 4)" in s
    assert '(1, 8, 3, 384, 384)' in s and '(1, 4, 2)' in s
    assert 'len(rows) != 1' in s and 'force_batchify=True' in s
    assert 'labels' not in s


def test_native_full_forward_and_last_valid_readout_no_generation():
    s=source('features')
    assert 'model.prepare_inputs_embeds(**inputs)' in s and 'model.language.model(' in s
    assert 'use_cache=False' in s and 'last_hidden_state' in s and 'masked_fill(~mask, -1)' in s
    n=source('native_model')
    assert '4.38.2' in source('runtime_sources') and '0.0.29.post2' in source('runtime_sources')
    assert 'requires_grad_(False)' in n and 'torch_dtype=torch.bfloat16' in n
    assert 'reproduction_receipt.json' in n
    assert '.generate(' not in Path(run.__file__).read_text()


def test_same_fifty_epoch_five_row_optimizer_and_cpu_validation():
    s=source('train_rows')
    assert 'range(1, 51)' in s and 'cross_entropy' in s
    assert 'lr=0.001, weight_decay=0.0' in s and 'T_max=50' in s
    assert '10240' in s and 'strict_cpu_metrics' in s


def test_prepare_stays_CPU_only_and_respects_official_hash_and_retry_limits():
    s=source('prepare_assets')
    assert 'range(1, 4)' in s and "'Retry-After'" in s
    assert 'range_request(' in s and 'shared.asset_valid' in s
    assert 'Preserve partial; no blind restart' in s and '3600' in s
    assert 'guard(' not in s and '.cuda(' not in s
    assert source('main').count("('cpu-preflight', 'suite', 'prepare')")==2


def test_finite_total_and_utilization_yield_before_GPU_dispatch():
    s=source('suite')
    assert '10800' in s and '7200' in s and '1800' in s and 'no automatic retry' in s
    assert s.index('args.gpu_not_before_unix')<s.index('for phase, name, bound')
    assert 'yielding_for_utilization' in s and 'deadline' in s
    assert 'a100_batch04' not in s and 'a100_batch08' not in s


def test_source_identity_memory_admission_and_fresh_full_test():
    assert all(len(v)==64 for v in run.sources().values())
    assert any(n.endswith('deepseek_vl2_quality.py') for n in run.sources())
    assert '37000' in source('guard')
    s=source('run')
    assert 'for i, row in enumerate(test)' in s and 'test_videos=558' in s
    assert 'formal_latency_energy_measured=False' in s
    assert 'native_forward_last_logits_vs_feature_tap_max_abs' in s


@pytest.mark.skipif(sys.platform=='win32',reason='Real Torch regression uses unchanged server environment')
def test_actual_CPU_2048_head_only_native_copy_updates():
    import torch
    from LightGenV2.tasks.t06_video_quality_assessment import quality_token_train as q
    torch.manual_seed(42)
    native=torch.randn(5,2048,dtype=torch.bfloat16,requires_grad=True)
    original=native.detach().clone();head=run.five_native_rows(native)
    x=torch.randn(12,2048,dtype=torch.float16);labels=torch.arange(12)%5
    assert sum(p.numel() for p in head.parameters())==10240
    assert list(head.state_dict())==['weight'] and head.weight.dtype==torch.float32
    torch.testing.assert_close(head(x),torch.nn.functional.linear(x.float(),original.float()),rtol=0,atol=0)
    loss=torch.nn.functional.cross_entropy(head(x),labels);loss.backward()
    assert native.grad is None and torch.isfinite(head.weight.grad).all()
    torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0.).step()
    assert not torch.equal(head.weight.detach(),original.float())
    torch.testing.assert_close(native.detach(),original,rtol=0,atol=0)
    clone=run.five_native_rows(head.state_dict()['weight']).eval().requires_grad_(False)
    metrics,mos,*_=q.evaluate(clone,x,torch.linspace(0.,1.,12),labels,torch.linspace(0.,1.,5),torch.device('cpu'))
    assert torch.isfinite(mos).all() and all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    with pytest.raises(ValueError):run.five_native_rows(torch.zeros(5,4096))


def test_existing_output_never_reused(tmp_path):
    with pytest.raises(FileExistsError):run.shared.new_output(tmp_path)


def test_allocator_recovery_changes_memory_policy_not_native_graph():
    s=source('suite');m=source('main')
    assert 'waiting_for_memory_without_reservation' in s
    assert 'free >= 37000' in s and 'Original total deadline during memory wait' in s
    assert s.index('waiting_for_memory_without_reservation')<s.index('for phase, name, bound')
    assert "cmd += ['--allocator-recovery']" in s
    assert "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'" in m
    assert m.index("os.environ['PYTORCH_CUDA_ALLOC_CONF']")<m.index('import torch, transformers')
    assert 'use_cache=False' in source('features') and 'incremental_prefilling' not in source('features')


def test_recovery_waits_CPU_only_then_dispatches_once_and_preserves_failure(tmp_path,monkeypatch):
    import json
    import time
    from types import SimpleNamespace
    output=tmp_path/'recovery';output.mkdir()
    prepare=tmp_path/'prepare';prepare.mkdir()
    (prepare/'status.json').write_text(json.dumps(dict(status='complete')))
    args=SimpleNamespace(deadline_unix=time.time()+100,prepare_run=prepare,output=output,
        gpu_not_before_unix=0,allocator_recovery=True,model=tmp_path/'model',manifest=tmp_path/'split.csv')
    calls=[];states=[];free=iter(['33000','37000'])
    monkeypatch.setattr(run.frozen,'write_json',lambda path,payload:states.append(json.loads(json.dumps(payload))))
    monkeypatch.setattr(run,'sources',lambda:{'entry':'unchanged'})
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**k:next(free))
    monkeypatch.setattr(run.time,'sleep',lambda seconds:calls.append(('CPU_wait',seconds)))
    def dispatch(cmd,**kwargs):
        calls.append(('dispatch',cmd,kwargs))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(run.subprocess,'run',dispatch)
    with pytest.raises(RuntimeError,match='no automatic retry'):
        run.suite(args,dict(source_sha256={'entry':'unchanged'}))
    assert calls[0][0]=='CPU_wait'
    dispatched=[c for c in calls if c[0]=='dispatch']
    assert len(dispatched)==1 and '--allocator-recovery' in dispatched[0][1]
    assert dispatched[0][2]['env']['CUDA_VISIBLE_DEVICES']==''
    assert states[-1]['status']=='failed'
    assert any(s.get('status')=='waiting_for_memory_without_reservation' for s in states)


@pytest.mark.skipif(sys.platform=='win32',reason='Actual mixed-dtype Torch audit uses unchanged server environment')
def test_native_parity_accepts_FP32_BF16_values_without_changing_input_dtypes():
    import torch
    tapped=torch.tensor([[.125,1.,-3.5]],dtype=torch.bfloat16)
    native=tapped.float()
    audit=run.native_logit_parity(native,tapped)
    assert audit['status']=='passed' and audit['max_abs']==0
    assert audit['native_dtype']=='torch.float32' and audit['tapped_dtype']=='torch.bfloat16'
    assert native.dtype==torch.float32 and tapped.dtype==torch.bfloat16
    assert audit['rtol']==.01 and audit['atol']==.125 and audit['audit_only_cast']


@pytest.mark.skipif(sys.platform=='win32',reason='Actual Torch audit uses unchanged server environment')
def test_native_parity_still_rejects_changed_nonfinite_or_wrong_shape_values():
    import torch
    with pytest.raises(RuntimeError,match='parity failed'):
        run.native_logit_parity(torch.zeros(1,3),torch.ones(1,3,dtype=torch.bfloat16))
    with pytest.raises(RuntimeError,match='Nonfinite'):
        run.native_logit_parity(torch.tensor([[float('nan')]]),torch.ones(1,1,dtype=torch.bfloat16))
    with pytest.raises(RuntimeError,match='shape mismatch'):
        run.native_logit_parity(torch.ones(1,3),torch.ones(1,2,dtype=torch.bfloat16))


def test_official_assets_hashed_once_with_progress_and_no_mtime_cache(tmp_path,monkeypatch):
    import hashlib
    from types import SimpleNamespace
    weight=b'original fixed official weight';metadata=b'{"native":true}'
    (tmp_path/'weight.safetensors').write_bytes(weight)
    (tmp_path/'config.json').write_bytes(metadata)
    spec={'official_assets':{
        'weight.safetensors':dict(size=len(weight),sha256=hashlib.sha256(weight).hexdigest()),
        'config.json':dict(git_blob_sha1=hashlib.sha1(b'blob '+str(len(metadata)).encode()+b'\0'+metadata).hexdigest())}}
    calls=[];states=[]
    def hash_once(path):
        calls.append(path.name)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(run.frozen,'sha256',hash_once)
    monkeypatch.setattr(run.frozen,'write_json',lambda path,payload:states.append(payload.copy()))
    args=SimpleNamespace(model=tmp_path,output=tmp_path/'out')
    expected={n:hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in spec['official_assets']}
    assert run.verify_model_assets(args,spec,'before_GPU')==expected
    assert calls==['weight.safetensors'] and states[-1]['verified_assets']==2
    assert run.verify_model_assets(args,spec,'after_GPU')==expected
    assert calls==['weight.safetensors','weight.safetensors']
    assert states[-1]['status']=='verifying_model_assets_after_GPU'
    (tmp_path/'weight.safetensors').write_bytes(b'x'*len(weight))
    with pytest.raises(ValueError,match='SHA mismatch'):run.verify_model_assets(args,spec,'after_GPU')


def test_smoke_bound_includes_asset_IO_but_never_extends_absolute_total_deadline():
    s=source('suite')
    assert "('gpu-smoke', 'gpu_smoke', 600)" in s
    assert 'str(min(bound, remaining))' in s and 'deadline - time.time()' in s
    assert 'shared.model_files' not in source('run') and 'shared.model_files' not in source('main')
    assert "verify_model_assets(args, c, 'before_GPU')" in source('main')
    assert "verify_model_assets(args, c, 'after_GPU')" in source('run')
