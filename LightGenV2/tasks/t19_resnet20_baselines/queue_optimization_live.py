"""Attach to existing stage identities and admit the one queued OpenMoji job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .optimization_shared_worker import validate_shared
from .shared_idle_gpu import snapshot
from .train_easy100 import TASK,ROOT,require_published_source,write_json,git
from .frozen_resnet18 import file_sha256


def actual(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return ''


def main(args):
    require_published_source()
    if not args.suite.resolve().is_relative_to((TASK/'runs/simulation').resolve()):raise ValueError('Suite outside task')
    path=args.suite/'queue_status.json'
    lock=(TASK/'runs/.t19_optimization_live_share2_20261024.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=json.loads(path.read_text())
    if actual(q['pid']):raise RuntimeError('Previous coordinator is still alive; refuse overlap')
    if q.get('resource_handoff')!='verified_waiting_OpenMoji_zero_updates':raise RuntimeError('Missing safe handoff evidence')
    candidates=q['gpu_uuids']
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    allowed={r.split(',')[1].strip() for r in inventory.splitlines() if int(r.split(',')[0]) in (2,4,5,6)}
    if set(candidates)!=allowed:raise RuntimeError('GPU set differs from original four cards')
    q.update(pid=os.getpid(),status='running',coordinator_commit=git('rev-parse','HEAD'),
        coordinator_source_SHA=file_sha256(Path(__file__)),resource_policy='max2_T19_on_stable_idle_cards',
        command=[sys.executable,'-m',__spec__.name,*sys.argv[1:]])
    observations={};logs={};children={}
    def save():write_json(path,q)
    save()
    try:
        while any(j['status'] in ('queued','running') for j in q['jobs']):
            for j in q['jobs']:
                if j['status']!='running':continue
                if j['pid'] in children:children[j['pid']].poll()
                command=actual(j['pid'])
                if command:
                    if j['name'] not in command or j['gpu_uuid'] not in command or 't19_resnet20_baselines.' not in command:
                        raise RuntimeError('PID command identity changed; refuse treating it as the worker')
                    continue
                base=TASK/'runs'/('smoke' if j['phase']=='probe' else 'simulation')/j['name']
                result=json.loads((base/'result.json').read_text()) if (base/'result.json').exists() else {}
                manifest=json.loads((base/'run_manifest.json').read_text()) if (base/'run_manifest.json').exists() else {}
                j.update(status='complete' if result.get('status')==manifest.get('status')=='complete' else 'failed',
                    finished_unix=time.time(),completion_verified_from_result_and_manifest=True)
                if j['pid'] in logs:logs.pop(j['pid']).close()
                save()
            for uuid in candidates:
                ready=[j for j in q['jobs'] if j['status']=='queued']
                if not ready:break
                if any(j['task']!='openmoji' or j['phase']!='formal' for j in ready):raise RuntimeError('Unexpected queued scientific job')
                peers=[j['pid'] for j in q['jobs'] if j['status']=='running' and j['gpu_uuid']==uuid]
                state=snapshot(uuid)
                try:validate_shared(state,peers,observations.get(uuid))
                except (RuntimeError,FileNotFoundError):
                    old=observations.get(uuid)
                    if old is None or state['observed_monotonic']-old['observed_monotonic']>=15:observations[uuid]=state
                    continue
                j=ready[0]
                command=[sys.executable,'-u','-m','LightGenV2.tasks.t19_resnet20_baselines.optimization_shared_worker',
                    '--task','openmoji','--phase','formal','--plan',str(args.plan),
                    '--run-dir',str(TASK/'runs/simulation'/j['name']),'--gpu-uuid',uuid,'--queue-status-path',str(path),
                    '--probe-result',str(TASK/'runs/smoke'/j['probe_name']/'result.json')]
                log=(args.suite/(j['name']+'.log')).open('x')
                process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[process.pid]=process
                logs[process.pid]=log;j.update(pid=process.pid,status='running',gpu_uuid=uuid,command=command,started_unix=time.time());save()
            time.sleep(5)
        q['status']='complete' if all(j['status']=='complete' for j in q['jobs']) else 'completed_with_attention'
    except BaseException as exc:q.update(status='supervisor_detached_workers_preserved',error=repr(exc));raise
    finally:
        save()
        for log in logs.values():log.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--suite',type=Path,required=True)
    main(parser.parse_args())
