"""One bounded probe/formal pair per task, on the existing four-GPU whitelist."""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from .shared_idle_gpu import snapshot,validate
from .train_easy100 import TASK,ROOT,git,require_published_source,write_json
from .frozen_resnet18 import file_sha256


def main(args):
    require_published_source()
    if not args.suite.resolve().is_relative_to((TASK/'runs/simulation').resolve()):raise ValueError('Suite outside task')
    plan=json.loads(args.plan.read_text())
    if set(plan['candidate_gpu_indices'])!={2,4,5,6} or plan['max_physical_gpus']!=4:
        raise ValueError('Cannot expand four-GPU resource authorization')
    tag=plan.get('run_tag','')
    if tag and re.fullmatch('[a-z0-9_]+',tag) is None:raise ValueError('Unsafe run tag')
    suffix='_'+tag if tag else ''
    lock=(TASK/f'runs/.t19_optimization_stage_20261024{suffix}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.suite.mkdir(parents=True,exist_ok=False)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    inventory={int(r.split(',')[0]):r.split(',')[1].strip() for r in rows.splitlines()}
    candidates=[inventory[i] for i in plan['candidate_gpu_indices']]
    jobs=[]
    for task in plan['tasks']:
        probe=f'{task}_optimization_probe_s42_20261024{suffix}'
        for phase in ('probe','formal'):
            name=probe if phase=='probe' else f'{task}_optimization_fixed30_formal_s42_20261024{suffix}'
            base=TASK/'runs'/('smoke' if phase=='probe' else 'simulation')
            if (base/name).exists():raise RuntimeError('Existing run evidence; refuse duplicate')
            jobs.append(dict(task=task,phase=phase,name=name,probe_name=probe,status='queued'))
    state=dict(status='running',pid=os.getpid(),jobs=jobs,command=[sys.executable,'-m',__spec__.name,*sys.argv[1:]],
        started_unix=time.time(),git_commit=git('rev-parse','HEAD'),profile_SHA=file_sha256(args.plan),
        gpu_uuids=candidates,max_physical_gpus=4,max_jobs_per_gpu=2)
    path=args.suite/'queue_status.json';running={};observations={}
    def save():write_json(path,state)
    save()
    try:
        while any(j['status'] in {'queued','running'} for j in jobs):
            for pid,(process,job,log) in list(running.items()):
                code=process.poll()
                if code is None:continue
                log.close();del running[pid]
                category='smoke' if job['phase']=='probe' else 'simulation'
                result_path=TASK/'runs'/category/job['name']/'result.json'
                result=json.loads(result_path.read_text()) if result_path.exists() else {}
                passed=code==0 and result.get('status')=='complete'
                if job['phase']=='probe':passed=passed and result.get('gate_passed') is True
                job.update(status='complete' if passed else ('probe_not_fit' if code==0 else 'failed'),exit_code=code,
                    finished_unix=time.time(),result_status=result.get('status'))
                if job['phase']=='probe' and not passed:
                    next(j for j in jobs if j['task']==job['task'] and j['phase']=='formal')['status']='blocked_by_TRAIN_probe'
                save()
            for uuid in candidates:
                owned=[pid for pid,(_,j,_) in running.items() if j['gpu_uuid']==uuid]
                # Respect CPU preflight reservations in the other live T19
                # registry as well as CUDA allocations in snapshot().
                for registry in (TASK/'runs/simulation').glob('*/queue_status.json'):
                    if registry==path:continue
                    try:other=json.loads(registry.read_text())
                    except (OSError,ValueError):continue
                    for peer in other.get('jobs',[]):
                        if peer.get('status')!='running' or peer.get('gpu_uuid')!=uuid or not peer.get('pid'):continue
                        try:command=Path(f"/proc/{peer['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
                        except (FileNotFoundError,PermissionError):continue
                        if ('t19_resnet20_baselines.' in command and uuid in command and str(TASK/'runs') in command):
                            owned.append(peer['pid'])
                if len(set(owned))>=2:continue
                ready=[j for j in jobs if j['status']=='queued' and (j['phase']=='probe' or
                    next(p for p in jobs if p['task']==j['task'] and p['phase']=='probe')['status']=='complete')]
                if not ready:continue
                state_gpu=snapshot(uuid)
                try:validate(state_gpu,owned,observations.get(uuid))
                except (RuntimeError,FileNotFoundError):
                    old=observations.get(uuid)
                    if old is None or state_gpu['observed_monotonic']-old['observed_monotonic']>=15:observations[uuid]=state_gpu
                    continue
                j=ready[0];category='smoke' if j['phase']=='probe' else 'simulation'
                command=[sys.executable,'-u','-m','LightGenV2.tasks.t19_resnet20_baselines.optimization_stage',
                    '--task',j['task'],'--phase',j['phase'],'--plan',str(args.plan),'--run-dir',str(TASK/'runs'/category/j['name']),
                    '--gpu-uuid',uuid,'--queue-status-path',str(path)]
                if j['phase']=='formal':command+=['--probe-result',str(TASK/'runs/smoke'/j['probe_name']/'result.json')]
                log=(args.suite/(j['name']+'.log')).open('x')
                process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,
                    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                j.update(status='running',pid=process.pid,command=command,gpu_uuid=uuid,started_unix=time.time())
                running[process.pid]=(process,j,log);save()
            time.sleep(5)
        state['status']='complete' if all(j['status']=='complete' for j in jobs) else 'completed_with_attention'
    except BaseException as exc:
        state.update(status='supervisor_detached_workers_preserved',error=repr(exc));raise
    finally:
        state['updated_unix']=time.time();save()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--suite',required=True,type=Path)
    main(parser.parse_args())
