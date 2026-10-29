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
    args=SimpleNamespace(profile='rental',gpu_uuid=entry.GPU,pipe='/data/tmp/rental_fullscope_20261029_01',suite_root=entry.ROOT,manifest=manifest)
    legacy=entry.initialize(args)
    rows=legacy.engine.base.old.rows()
    assert len(rows)==512 and all(r['split']=='test' for r in rows)
    assert entry.base.ordered_id_sha(rows)==entry.config()['first512_ids_sha256']
    assert 'torch' not in sys.modules

if __name__=='__main__':
    test_exact_plan();test_cpu_scope();test_receiver();test_native_rows_subset()
    print('{"status":"passed","tests":4,"torch_imported":false}')
