"""Three-card six-fit queue; no test, no duplicate start, no automatic retry."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import caltech_deploy234 as t


def gpu_available(uuid):
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    line=next(x.split(',') for x in lines if x.startswith(uuid))
    assert 'A100' not in line[1] and uuid in t.ALLOWED.values()
    return int(line[2])>=3*1024+800


def suite(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    profile=t.read(t.PROFILE);assert profile['maximum_gpu_children']==3
    manifest=t.manifest(a.data);src=t.sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=t.ROOT,text=True).strip();t.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False)
    t.save(a.out/'training_started.json',dict(pid=os.getpid(),time_unix=time.time(),sources=src,git_commit=commit,gpu_indices=[0,1,2,4],maximum_gpu_children=3,A100_allowed=False,data_sha256=manifest['cache_sha256']),exclusive=True)
    t.save(a.out/'metadata.json',dict(pid=os.getpid(),command=sys.argv,sources=src,git_commit=commit,config=profile,test_evaluations=0))
    jobs=[dict(depth=d,rho=r,name=f'L{d}_'+('rho03' if r else 'rho0')) for r in (.3,0.) for d in (4,3,2)]
    pending=jobs[:];active={};completed=[];failures=[];start=time.time()
    while active or (pending and not failures):
        for gpu,(proc,job) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            if code==0 and (a.out/job['name']/'fit_result.json').exists():completed.append(dict(job,exit_code=code))
            else:failures.append(dict(job,exit_code=code))
            del active[gpu]
        if (a.out/'stop_dispatch.json').exists():break
        if not failures:
            for index in (2,1,4,0):
                gpu=t.ALLOWED[index]
                if gpu in active or not pending or len(active)>=3 or not gpu_available(gpu):continue
                t.verify_sources(src,commit);job=pending.pop(0)
                cmd=[sys.executable,str(t.HERE/'caltech_deploy234.py'),'--out',str(a.out/job['name']),'--data',str(a.data),'--depth',str(job['depth']),'--rho',str(job['rho']),'--gpu',gpu]
                with (a.out/(job['name']+'.log')).open('x') as log:
                    proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(pid=proc.pid,gpu_index=index,gpu_uuid=gpu,command=cmd);active[gpu]=(proc,job)
        t.save(a.out/'current_processes.json',dict(active=[job for proc,job in active.values()],pending=pending))
        t.save(a.out/'status.json',dict(state='fit_failure_draining' if failures else 'fit_running',completed=completed,failures=failures,active=len(active),pending=len(pending),seconds=time.time()-start,test_evaluations=0))
        if active or (pending and not failures):time.sleep(5)
    t.save(a.out/'current_processes.json',dict(active=[job for proc,job in active.values()],pending=pending))
    t.save(a.out/'status.json',dict(state='fit_failed_no_automatic_retry' if failures else ('dispatch_stopped' if pending else 'fit_complete_awaiting_cpu_review'),completed=completed,failures=failures,active=len(active),pending=pending,seconds=time.time()-start,test_evaluations=0))
    assert not failures,failures

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True);suite(p.parse_args())