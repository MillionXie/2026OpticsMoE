import ast
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import rental_strict_tf_dual_20261020 as m
from LightGenV2.tasks.t06_video_quality_assessment import rental_precision_recheck as runtime


def test_exact_devices_batches_and_finite_budget():
    c=m.config()
    assert len(c['gpu_uuids'])==len(set(c['gpu_uuids']))==2
    assert len(m.placement('deepseek','fp32'))==2
    assert all(len(m.placement(model,precision))==1 for model,precision in [('kimi','fp32'),('kimi','native'),('deepseek','native')])
    assert c['mps_batch']==1 and c['mps_caps']==[100,50]
    assert c['temporal_batches']==dict(deepseek_native=2,kimi_native=6,deepseek_fp32=1,kimi_fp32=1)
    assert c['compute_deadline']<c['backup_release_deadline']
    assert len(c['temporal_batches'])*4==16


def test_dual_runtime_is_explicit_whole_blocks_gpu_only_not_auto_offload():
    text=Path(runtime.__file__).read_text();ast.parse(text)
    assert "torch.cuda.device_count()!=2" in text
    assert "'language.model.norm':1,'language.lm_head':1" in text
    assert "0 if i<15 else 1 for i in range(30)" in text
    assert "{p.device.type for p in model.parameters()}!={'cuda'}" in text
    assert "value.to(head.weight.device)" in text
    assert "dtype!=torch.float32" in text
    assert "if not dual:model=model.cuda()" in text


def test_actual_smoke_precedes_formal_cost_and_new_full558_is_bound():
    source=Path(m.__file__).read_text();ast.parse(source)
    assert source.index('smoke(args,compute,inputs,first,args.output,guard)')<source.index('report=timing.measure(')
    assert 'for delay in (0.,.05,0.):' in source
    assert 'all_used_devices_synchronized=True' in source
    assert "args.accuracy" in source and "cmd.append('--accuracy')" in source
    assert "len(rows)!=558" in source
    assert "float(row['temporal'])" in source
    assert 'np.array(targets),np.array(predictions)' in source
    assert "checkpoint_sha256=identity['checkpoint_sha256']" in source
    assert "with (ROOT/(name+'.log')).open('x')" in source
    assert 'no automatic retry' in source


def test_never_changes_power_precision_or_formal_b1():
    source=Path(m.__file__).read_text()
    assert 'allow_tf32=False' in source and "torch.set_float32_matmul_precision('highest')" in source
    assert 'nvidia-smi -pl' not in source and 'nvidia-smi -mig' not in source
    assert 'full558' in source
    assert 'qwen_batch01' not in source
