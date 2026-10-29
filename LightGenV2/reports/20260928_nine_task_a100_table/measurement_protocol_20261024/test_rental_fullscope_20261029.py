"""CPU-only authorization/plan/adapter tests; no model, CUDA or network."""
import importlib.util
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_rental_fullscope_20261029 as entry

def test_exact_plan():
    points=entry.plan('rental')
    assert len(points)==len(set(points))==12
    assert set(points)=={(m,c,r) for m in ('kimi','deepseek') for c in (50,100) for r in (1,2,3)}
    try:entry.plan('lab')
    except ValueError:pass
    else:raise AssertionError('Lab must not be authorized')

def test_cpu_scope():
    entry.configure()
    c=entry.config()
    assert c['videos_per_repeat']==512 and c['power_maximum_support_age_s']==.25
    assert c['explicit_warmup_forwards']==0 and c['host_proxy_w']==338.2
    assert 'all_511_internal_gaps' in c['included']
    assert 'torch' not in sys.modules
    assert entry.base.initialize_profile is entry.initialize

def test_receiver():
    command=entry.observer.observer_command(entry.GPU,Path('/tmp/data.jsonl'))
    assert command[2].endswith('rental_nvml_observer_20261029.py')
    assert entry.GPU in command
    try:entry.observer.observer_command('foreign',Path('/tmp/data'))
    except ValueError:pass
    else:raise AssertionError('Other GPU must be rejected')

def test_native_rows_subset():
    from types import SimpleNamespace
    manifest=entry.base.TASK/'runs/simulation/lgvq_full_deepseek_kimi_rental_20261017/provisioning/original_manifest.csv'
    args=SimpleNamespace(profile='rental',gpu_uuid=entry.GPU,pipe='/data/tmp/rental_fullscope_20261029_03',suite_root=entry.ROOT,manifest=manifest)
    legacy=entry.initialize(args)
    rows=legacy.engine.base.old.rows()
    # Original native runtime insists on558 metadata rows. The outer
    # measurement collector alone selects512; no unused file is opened here.
    assert len(rows)==558 and all(r['split']=='test' for r in rows)
    assert entry.base.ordered_id_sha(rows[:512])==entry.config()['first512_ids_sha256']
    assert rows[:512]==entry.base.input_rows(manifest)
    assert 'torch' not in sys.modules

def test_finite_launcher_cpu():
    import launch_rental_when_ready_20261029 as launcher
    assert launcher.entry.ROOT==entry.ROOT
    assert launcher.DEADLINE==1793278441.7376518
    assert launcher.entry.GPU=='GPU-30fecb94-0222-a3d5-d952-a91f2cab2f43'
    assert 'torch' not in sys.modules

def test_pipe_parent_provision():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as folder:
        pipe=Path(folder)/'previously_missing_parent'/'private_pipe'
        entry.prepare_pipe_parent(pipe);entry.prepare_pipe_parent(pipe)
        assert pipe.parent.is_dir() and not pipe.exists()
    assert 'torch' not in sys.modules

def test_rental_NVML400_profile_actual_sampler_start():
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch
    original=entry.observer.original
    old_gpu,old_cap,old_command=original.GPU,original.POWER_LIMIT_MW,original.observer_command
    fake=SimpleNamespace(nvmlInit=lambda:None,nvmlShutdown=lambda:None,
        nvmlDeviceGetHandleByUUID=lambda x:x,nvmlDeviceGetUUID=lambda h:entry.GPU,
        nvmlDeviceGetPowerManagementLimit=lambda h:400000,
        nvmlDeviceGetPowerUsage=lambda h:65837,nvmlDeviceGetUtilizationRates=lambda h:SimpleNamespace(gpu=37),
        nvmlDeviceGetMemoryInfo=lambda h:SimpleNamespace(used=1536*1024**2),
        nvmlSystemGetDriverVersion=lambda:'CPU fake',nvmlSystemGetNVMLVersion=lambda:'CPU fake')
    try:
        entry.observer.configure_receiver()
        with TemporaryDirectory() as folder,patch.dict(sys.modules,{'pynvml':fake}):
            sampler=original.Sampler(entry.GPU,Path(folder)/'new.jsonl');sampler.start()
            assert sampler.identity['power_limit_milliwatts']==400000
            sampler.stop()
            fake.nvmlDeviceGetPowerManagementLimit=lambda h:250000
            try:original.Sampler(entry.GPU,Path(folder)/'must_not_create.jsonl').start()
            except ValueError:pass
            else:raise AssertionError('Lab250 cap must not qualify for rental400')
            fake.nvmlDeviceGetPowerManagementLimit=lambda h:400000
            fake.nvmlDeviceGetUUID=lambda h:'foreign'
            try:original.Sampler(entry.GPU,Path(folder)/'foreign.jsonl').start()
            except ValueError:pass
            else:raise AssertionError('Foreign UUID must be rejected')
    finally:original.GPU,original.POWER_LIMIT_MW,original.observer_command=old_gpu,old_cap,old_command
    assert old_cap==250000 and 'torch' not in sys.modules

if __name__=='__main__':
    test_exact_plan();test_cpu_scope();test_receiver();test_native_rows_subset();test_finite_launcher_cpu();test_pipe_parent_provision();test_rental_NVML400_profile_actual_sampler_start()
    print('{"status":"passed","tests":7,"torch_imported":false}')
