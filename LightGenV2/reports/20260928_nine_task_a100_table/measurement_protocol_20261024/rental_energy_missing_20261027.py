"""Missing-only rental continuation; unchanged qualified native energy workers.

No Torch import, source deployment, lease operation or automatic retry.
Retained point cost hashes are supplied as data, checked before GPU admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
import rental_energy_triples_20261026 as energy
import rental_energy_cap100_20261026 as full
ENTRY=Path(__file__).resolve();SIM=energy.original.TASK/'runs/simulation'
EXPECTED={(m,c,r) for m in ('kimi','deepseek') for c in (50,100) for r in (1,2,3)}


def missing_plan(retained):
    identities=[(p['model'],p['cap'],p['repeat']) for p in retained]
    if len(identities)!=len(set(identities)) or not set(identities)<=EXPECTED:
        raise ValueError('Unknown or duplicate retained point')
    order=[('deepseek',50,3)]+[(m,100,r) for r in (1,2,3) for m in ('kimi','deepseek')]
    order += sorted(EXPECTED-set(order))
    return [p for p in order if p not in set(identities)]


def source_map():
    result=energy.original.sources()
    for p in (ENTRY,Path(energy.__file__),Path(full.__file__),
              REPO/'LightGenV2/tasks/t06_video_quality_assessment/e2e_telemetry_observer.py',
              REPO/'LightGenV2/tasks/t06_video_quality_assessment/nvml_observer_process.py',
              REPO/'LightGenV2/tasks/t06_video_quality_assessment/batch_telemetry.py',
              REPO/'LightGenV2/tasks/t06_video_quality_assessment/strict_tf_window_20261020.py'):
        result[p.relative_to(REPO).as_posix()]=energy.original.sha(p)
    return result


def verify_sources(commit,sources):
    if len(commit)!=40:raise ValueError('Exact published Git commit required')
    for name,digest in sources.items():
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO,timeout=10)
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Unpublished source: '+name)


def validate_retained(records):
    for row in records:
        leaf=(REPO/row['relative']).resolve()
        if not leaf.is_relative_to(SIM) or leaf.is_symlink():raise ValueError('Canonical retained leaf required')
        read=lambda n:json.loads((leaf/n).read_text())
        cost=read('cost_report.json');identity=read('identity.json')
        if (energy.original.sha(leaf/'cost_report.json')!=row['cost_sha256']
                or read('status.json')['status']!='complete' or read('energy_status.json')['status']!='complete'
                or cost['status']!='complete' or cost['video_calls']!=2048
                or (cost['model'],cost['cap'],cost['round'])!=(row['model'],row['cap'],row['repeat'])
                or identity['gpu_uuid']!=energy.original.GPU or identity['power_limit_w']!=400):
            raise ValueError('Retained point identity or original cost SHA differs')


def unique():
    names={ENTRY.name,Path(energy.__file__).name,Path(full.__file__).name}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=os.geteuid():continue
            argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if any(Path(a).name in names for a in argv):raise RuntimeError('Existing rental entry: '+proc.name)
        except (FileNotFoundError,ProcessLookupError):continue


def worker(args):
    receipt=json.loads((args.root/'source_receipt.json').read_text())
    if source_map()!=receipt['sources']:raise ValueError('Frozen runtime source changed')
    verify_sources(receipt['published_commit'],receipt['sources'])
    if args.output!=args.root/f'{args.model}{args.cap}_repeat{args.repeat:02d}':raise ValueError('Exact new leaf required')
    if args.output.exists():raise FileExistsError('Preserve previous leaf')
    previous=energy.original.sources
    energy.original.sources=lambda:dict(previous(),**{ENTRY.relative_to(REPO).as_posix():energy.original.sha(ENTRY)})
    try:(full.worker if args.cap==100 else energy.worker)(args)
    finally:energy.original.sources=previous


def stop_verified(child,leaf):
    if child is None or child.poll() is not None:return
    proc=Path('/proc',str(child.pid));argv=(proc/'cmdline').read_bytes().split(b'\0')
    if proc.stat().st_uid!=os.geteuid() or os.getpgid(child.pid)!=child.pid or str(ENTRY).encode() not in argv or str(leaf).encode() not in argv:
        raise RuntimeError('Refuse signal to unverified worker')
    os.killpg(child.pid,signal.SIGINT);child.wait(timeout=60)


def suite(args):
    import fcntl
    unique()
    if args.root.exists():raise FileExistsError('Preserve existing suite')
    retained=json.loads(args.retained.read_text());validate_retained(retained)
    sources=source_map();commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'];verify_sources(commit,sources)
    plan=missing_plan(retained)
    if not plan:raise ValueError('No missing points; do not rerun qualified data')
    args.root.mkdir();energy.write(args.root/'source_receipt.json',dict(published_commit=commit,sources=sources))
    state=dict(status='admission',pid=os.getpid(),deadline=args.deadline,plan=plan,completed=[],retained=retained,
        sources=sources,published_commit=commit,gpu_uuid=energy.original.GPU,common_tokens=913408)
    save=lambda:energy.write(args.root/'suite_status.json',state)
    child=None;leaf=None;active=False;active_cap=None;save()
    def bounded():
        if time.time()>=args.deadline:raise TimeoutError('Finite rental deadline')
        if source_map()!=sources:raise RuntimeError('Active source changed')
    try:
        with (SIM/'.temporal_telemetry_a100.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if Path(energy.original.PIPE).exists():raise FileExistsError('New private MPS pipe required')
            Path(energy.original.PIPE).mkdir(parents=True);(args.root/'mps_logs').mkdir()
            for model,cap,repeat in plan:
                bounded()
                if time.time()>=args.deadline-600:raise TimeoutError('Reserve data backup margin')
                if active_cap!=cap:
                    if active:
                        energy.original.control('quit');active=False
                    for _ in range(12):
                        bounded()
                        if energy.original.gpu_processes():raise RuntimeError('Foreign GPU process')
                        time.sleep(.25)
                    if energy.original.control('get_server_list').returncode==0:raise RuntimeError('Existing private MPS daemon')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=energy.original.GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        CUDA_MPS_PIPE_DIRECTORY=energy.original.PIPE,CUDA_MPS_LOG_DIRECTORY=str(args.root/'mps_logs'),
                        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
                    subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True,timeout=10)
                    active=True;active_cap=cap
                leaf=args.root/f'{model}{cap}_repeat{repeat:02d}'
                command=['/data/venvs/'+model+'/bin/python','-u',str(ENTRY),'--phase','worker','--root',str(args.root),
                    '--output',str(leaf),'--model',model,'--cap',str(cap),'--repeat',str(repeat),
                    '--deadline',str(args.deadline),'--common-tokens','913408']
                energy.write(args.root/(leaf.name+'.command.json'),command)
                with (args.root/(leaf.name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=leaf.name,child=child.pid);save()
                    while child.poll() is None:bounded();time.sleep(1)
                if child.returncode!=0:raise RuntimeError('Preserve failed point; no blind retry')
                report=json.loads((leaf/'cost_report.json').read_text())
                if report['status']!='complete' or report['video_calls']!=2048 or report['cap']!=cap:
                    raise RuntimeError('Missing complete qualified cost')
                state['completed'].append(leaf.name);save()
            state.update(status='complete',finished=time.time())
    except BaseException as error:
        state.update(status='stopped',error=repr(error));stop_verified(child,leaf);raise
    finally:
        if active:
            release=energy.original.control('quit');state['mps_release_returncode']=release.returncode
        state['mps_released']=energy.original.control('get_server_list').returncode!=0;save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','worker'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--retained',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--model',choices=['kimi','deepseek'])
    p.add_argument('--cap',type=int,choices=[50,100]);p.add_argument('--repeat',type=int,choices=[1,2,3])
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--common-tokens',type=int,default=913408)
    args=p.parse_args();args.root=args.root.resolve()
    if REPO!=Path('/data/2026OpticsMoE') or args.root.parent!=SIM or not args.root.name.startswith('a10080_energy_missing_20261027_'):
        raise ValueError('Explicit canonical rental continuation required')
    if not 0<args.deadline-time.time()<=36000 or args.common_tokens!=913408:raise ValueError('Original finite scope required')
    energy.bind(args.root)
    if args.phase=='suite':suite(args)
    else:args.output=args.output.resolve();worker(args)


if __name__=='__main__':main()
