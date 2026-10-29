"""Authorized NEW A10080: native Kimi/DS full-input B1, 512 videos x3 x2 caps.

Reuse original arithmetic, exact assets/head, cold input timing, actual NVML
timestamps, 250ms gate and serialized original MPS guard. No LG/lab rerun.
"""
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
import sys
import time
import run_temporal_qwen_fullscope_20261029 as adapter
import git_blob_identity_20261029 as identity
import rental_nvml_observer_20261029 as observer
import nvml_direct_observer_20261029 as direct
import serialized_mps_guard_20261029 as serialized

ENTRY=Path(__file__).resolve();HERE=ENTRY.parent;REPO=adapter.REPO;base=adapter.base
ROOT=base.TASK/'runs/simulation/rental_fullscope_20261029_03'
GPU=observer.GPU
PROTOCOL='rental80_cold_file_to_CPU_B1_512x3_20261029'
NATIVE_INITIALIZE=base.initialize_profile
FROZEN_SOURCES=None

def prepare_pipe_parent(pipe):
    """New image has no /data/tmp; never leave creation to the old daemon."""
    parent=Path(pipe).parent
    if parent.is_symlink():raise ValueError('Private pipe parent may not be redirected')
    parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    if not parent.is_dir():raise ValueError('Private pipe parent is not a directory')

def verified_admission_predecessor():
    """Exact preserved zero-inference01/02 failures; never resume either root."""
    old=ROOT.with_name('rental_fullscope_20261029_01');p=old/'suite_status.json'
    if base.sha(p)!='76b8d63325c64ef9df2351fcf5266bc72a28afb99c52745f3d53dacbe14a2002':
        raise ValueError('Original stopped admission evidence differs')
    state=json.loads(p.read_text())
    if (state['status']!='stopped' or state['completed'] or state.get('phase') or state.get('child_pid')
            or state['error']!="FileNotFoundError(2, 'No such file or directory')"):
        raise ValueError('No measured worker may be revived or skipped')
    if any(Path('/proc/'+str(pid)).exists() for pid in (2836,2837)):
        raise ValueError('Original admission processes must truly exit')
    if Path('/data/tmp/rental_fullscope_20261029_01').exists():raise ValueError('Old private pipe unexpectedly exists')
    old=ROOT.with_name('rental_fullscope_20261029_02');p=old/'suite_status.json'
    if base.sha(p)!='e09855bd1656679dbfd7b16792d7c6f79152a26ae6124f588729dbcfd270b4b6':
        raise ValueError('Original observer-profile failure evidence differs')
    state=json.loads(p.read_text());leaf=old/'kimi_cap50_repeat01'
    if state['status']!='stopped' or state['completed'] or state.get('mps_released') is not True:
        raise ValueError('Original02 must be stopped and fully released')
    status=json.loads((leaf/'status.json').read_text())
    if (status['status']!='failed' or status['samples']!=0 or (leaf/'calls.csv').read_text().strip()
            or 'Original physical UUID and250W cap required' not in status['error']):
        raise ValueError('Only exact pre-inference resource-profile error is authorized here')
    if any(Path('/proc/'+str(pid)).exists() for pid in (2970,2971,2989)):
        raise ValueError('Original02 parent/worker must truly exit')
    check=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True,timeout=5,
        env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY='/data/tmp/rental_fullscope_20261029_02'))
    if check.returncode!=1:raise ValueError('Original02 private daemon must be absent')

def plan(profile,models=None,caps=None,repeats=None):
    if profile!='rental':raise ValueError('Only explicit new rental authorization')
    if models and models!=['kimi','deepseek']:raise ValueError('Exact two-model plan required')
    return [(m,c,r) for m in ('kimi','deepseek') for c in (50,100) for r in (1,2,3)]

def sources():
    if FROZEN_SOURCES is None:raise RuntimeError('Configure exact rental sources first')
    return {name:base.sha(REPO/name) for name in FROZEN_SOURCES}

def config():
    c=adapter.ORIGINAL_CONFIG()
    return dict(c,protocol_id=PROTOCOL,videos_per_repeat=512,
        authorization='2026-10-29 human supplied new ebcloud rental endpoint; measure Kimi/DS MPS50/100',
        models=['kimi','deepseek'],resource_caps=[50,100],
        additional_units=['ms_per_video','estimated_joules_per_video','512_video_continuous_wall_seconds'],
        included=[s.replace('199_internal_gaps','511_internal_gaps') for s in c['included']],
        first512_ids_sha256='48b7fea193abc07f48bec9b091db0351b949387c9b44fbd228399ce3a03b655a',
        gpu_uuid=GPU,power_limit_w=400,host_proxy_is_not_measured_rental_host_power=True,
        guard_query_serialization='per_worker_RLock_whole_original_check_no_retry_v1',
        observer='direct_NVML_actual_API_receipt_no_stdout_hop_v1')

def initialize(args):
    legacy=NATIVE_INITIALIZE(args)
    legacy.engine.MPSGuard=serialized.serialized_guard_class(legacy.engine.MPSGuard)
    # Only these already-approved TEST inputs are measured. The legacy rows()
    # demands presence of all2250 unused TRAIN files; avoid that storage-only
    # requirement without creating placeholders or changing any input/forward.
    # Native runtime retains its558-row manifest/order identity check. Only
    # the outer fullscope collector prepares/executes the first512 rows.
    legacy.engine.base.old.rows=lambda: native_manifest_rows(args.manifest)
    return legacy


def native_manifest_rows(manifest):
    selected=base.input_rows(manifest)  # exact original manifest and first512
    with Path(manifest).open(newline='',encoding='utf-8-sig') as stream:
        rows=[r for r in csv.DictReader(stream) if r['split']=='test']
    if (len(rows)!=558 or len({r['sample_id'] for r in rows})!=558
            or rows[:512]!=selected):
        raise ValueError('Original558 metadata with exact first512 required')
    return rows

def configure():
    global FROZEN_SOURCES
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_dataset_once as legacy
    legacy.select_profile('rentalb')
    paths=[ENTRY,Path(adapter.__file__),Path(identity.__file__),Path(observer.__file__),Path(direct.__file__),Path(serialized.__file__)]
    FROZEN_SOURCES={**adapter.ORIGINAL_SOURCES(),**{p.relative_to(REPO).as_posix():base.sha(p) for p in paths}}
    adapter.ENTRY=ENTRY;adapter.NEW_PROTOCOL=PROTOCOL;adapter.source_map=sources
    adapter.configure(ROOT)
    base.plan=plan;base.config=config;base.initialize_profile=initialize
    from LightGenV2.tasks.t06_video_quality_assessment import e2e_telemetry_observer as old_observer
    old_observer.observer_command=observer.observer_command
    for fn in (base.worker,base.suite,base.summarize,base.audit_point):
        fn.__globals__.update(vars(base));fn.__globals__.update(NEW_PROTOCOL=PROTOCOL,ENTRY=ENTRY)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('suite','worker','audit'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--suite-root',type=Path)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--deadline',type=float)
    p.add_argument('--profile',choices=('rental',),default='rental')
    p.add_argument('--model',choices=('kimi','deepseek'));p.add_argument('--cap',type=int,choices=(50,100))
    p.add_argument('--repeat',type=int,choices=(1,2,3));p.add_argument('--model-path',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--reproduction-receipt',type=Path)
    p.add_argument('--gpu-uuid',choices=(GPU,),default=GPU);p.add_argument('--pipe')
    a=p.parse_args();a.output=a.output.resolve();a.suite_root=(a.suite_root or a.output).resolve()
    if a.suite_root!=ROOT or (a.phase=='suite' and a.output!=ROOT):raise ValueError('Single new canonical rental root only')
    if a.phase!='suite' and a.output.parent!=ROOT:raise ValueError('Exact terminal leaf required')
    if a.phase!='audit' and (a.deadline is None or not 0<a.deadline-time.time()<=28800):raise ValueError('Finite at most eight-hour deadline required')
    if a.phase=='worker' and (a.model,a.cap,a.repeat) not in plan('rental'):raise ValueError('Exact native worker required')
    a.models=['kimi','deepseek'];a.caps=[50,100];a.repeats=[1,2,3]
    a.pipe=a.pipe or '/data/tmp/rental_fullscope_20261029_03'
    if a.pipe!='/data/tmp/rental_fullscope_20261029_03':raise ValueError('Exact private MPS pipe required')
    configure()
    if a.phase=='audit':
        import json
        print(json.dumps(base.audit_point(a.output,a.manifest),ensure_ascii=True));return
    identity.verify(os.environ['MEASUREMENT_PUBLISHED_COMMIT'],sources(),REPO)
    if a.phase=='suite':
        verified_admission_predecessor()
        prepare_pipe_parent(a.pipe)
    (base.suite if a.phase=='suite' else base.worker)(a)

if __name__=='__main__':main()
