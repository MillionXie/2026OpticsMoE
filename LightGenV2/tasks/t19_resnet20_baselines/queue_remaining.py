"""One bounded queue, LGVQ first; CPU-prepared tasks can be appended to its plan."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .train_easy100 import TASK,ROOT,PROFILE,admit_gpu,require_published_source,write_json,git

ALLOWED = ('lgvq_temporal','lgvq_spatial','abo_image_to_image','lsp','salicon','openmoji','abo_conditional_edit')


def read_plan(path):
    plan=json.loads(path.read_text())
    tasks=plan['tasks']
    if len(set(tasks))!=len(tasks) or any(t not in ALLOWED for t in tasks):
        raise ValueError('Invalid/duplicate task in remaining-seven plan')
    return plan


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuids',nargs='+',required=True)
    p.add_argument('--suite-id',required=True)
    p.add_argument('--task-plan',type=Path,required=True)
    args=p.parse_args()
    if not 1<=len(set(args.gpu_uuids))==len(args.gpu_uuids)<=3:
        raise ValueError('One to three unique physical GPUs required')
    if Path(args.suite_id).name!=args.suite_id: raise ValueError('Invalid suite-id')
    require_published_source()
    lock=(TASK/'runs/.t19_gpu_suite.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    suite=TASK/'runs/simulation'/args.suite_id
    suite.mkdir(exist_ok=False)
    report={'status':'running','pid':os.getpid(),'gpu_uuids':args.gpu_uuids,'jobs':[],
            'max_physical_gpus':3,'max_jobs_per_gpu':2,'execution_profile':PROFILE,
            'started_unix':time.time(),'initial_commit':git('rev-parse','HEAD')}
    running,children={},[]
    def save(): write_json(suite/'queue_status.json',report)
    def interrupt(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        while True:
            plan=read_plan(args.task_plan)
            for task in plan['tasks']:
                for phase in ('smoke','formal'):
                    if not any(j['task']==task and j['phase']==phase for j in report['jobs']):
                        report['jobs'].append({'task':task,'phase':phase,'status':'queued',
                            'name':f'{task}_{phase}_s42_{args.suite_id}'})
            for pid,(proc,job,log) in list(running.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del running[pid]
                root=TASK/('runs/smoke' if job['phase']=='smoke' else 'runs/simulation')/job['name']
                result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {}
                job.update(status='complete' if code==0 and result.get('status')=='complete' else 'failed',
                           exit_code=code,finished_unix=time.time(),result=result)
                # A failed task does not kill independent tasks, but its formal
                # training is blocked. Never retry failures silently.
            for uuid in [u for u in args.gpu_uuids for _ in range(2)]:
                owned=[pid for pid,(_,j,_) in running.items() if j['gpu_uuid']==uuid]
                if len(owned)>=2:continue
                eligible=[]
                for j in report['jobs']:
                    if j['status']!='queued':continue
                    if j['phase']=='formal':
                        smoke=next(x for x in report['jobs'] if x['task']==j['task'] and x['phase']=='smoke')
                        if smoke['status']=='failed':
                            j['status']='blocked_by_failed_smoke'
                            continue
                        if smoke['status']!='complete':continue
                    eligible.append(j)
                if not eligible:continue
                job=eligible[0]
                old=os.environ.get('CUDA_VISIBLE_DEVICES')
                os.environ['CUDA_VISIBLE_DEVICES']=uuid
                try:admit_gpu(uuid,owned)
                except (RuntimeError,FileNotFoundError) as exc:
                    report.setdefault('waiting_for_headroom',{})[uuid]=str(exc)
                    continue
                finally:
                    if old is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
                    else:os.environ['CUDA_VISIBLE_DEVICES']=old
                report.setdefault('waiting_for_headroom',{}).pop(uuid,None)
                base=TASK/('runs/smoke' if job['phase']=='smoke' else 'runs/simulation')
                command=[sys.executable,'-m','LightGenV2.tasks.t19_resnet20_baselines.train_tasks',
                    '--task',job['task'],'--run-dir',str(base/job['name']),'--gpu-uuid',uuid,
                    '--epochs','1' if job['phase']=='smoke' else '30',
                    '--smoke-batches','2' if job['phase']=='smoke' else '0',
                    '--queue-status-path',str(suite/'queue_status.json')]
                log=(suite/(job['name']+'.log')).open('x')
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,
                    OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(status='running',pid=proc.pid,gpu_uuid=uuid,command=command,
                           started_unix=time.time(),published_commit_at_launch=git('rev-parse','HEAD'))
                running[proc.pid]=(proc,job,log);children.append(proc)
            save()
            if plan.get('sealed') and not running and not any(j['status']=='queued' for j in report['jobs']):
                report['status']='complete' if all(j['status']=='complete' for j in report['jobs']) else 'complete_with_failures'
                break
            time.sleep(5)
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        for proc,job,log in running.values():
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            job['status']='interrupted'
        for proc,job,log in running.values():
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            log.close()
        raise
    finally:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        pids={proc.pid for proc in children};remaining=[]
        for line in apps.splitlines():
            parts=line.split(',')
            if len(parts)!=2:continue
            pid=int(parts[1].strip())
            try:owned=pid in pids or os.getpgid(pid) in pids
            except ProcessLookupError:owned=False
            if owned:remaining.append(line)
        report.update(finished_unix=time.time(),remaining_owned_gpu_processes=remaining,
                      owned_gpu_processes_released=not remaining)
        save();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
