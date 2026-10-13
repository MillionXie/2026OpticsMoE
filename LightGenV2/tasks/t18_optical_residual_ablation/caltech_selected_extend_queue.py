"""Four-GPU bounded common200 continuation queue, fit only, no automatic retry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import caltech_selected_extend as e


def suite(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=e.read(e.HERE/'caltech_selected_extend_plan.json')
    assert plan['total_epochs']==200 and plan['additional_epochs']==100 and plan['regularization_change'] is None
    assert e.read(a.parent/'status.json')['state']=='fit_complete_awaiting_validation_review'
    assert e.read(a.parent/'current_processes.json')['active']==[]
    assert all(e.parent_tools.idle(gpu) for gpu in e.GPUS),'No other project is displaced'
    parents={key:a.parent/key for key in e.KEYS};fits={};receipts={}
    for key,parent in parents.items():
        fit,last,manifest=e.verify_parent(parent,a.data);fits[key]=fit
        receipts[key]=dict(last_checkpoint_sha256=e.sha(parent/'last_checkpoint.pt'),best_checkpoint_sha256=e.sha(parent/'best_checkpoint.pt'),
            raw_state_sha256=e.state_digest(last['model']),ema_state_sha256=e.state_digest(last['ema']),
            optimizer_state_sha256=e.state_digest(last['optimizer']),scheduler_state_sha256=e.state_digest(last['scheduler']),
            original_pid=e.read(parent/'metadata.json')['pid'])
        assert not Path('/proc',str(receipts[key]['original_pid'])).exists()
        del last
    for depth in e.DEPTHS:
        e.parent_tools.verify_pair([fits[f'L{depth}_{r}'] for r in ('rho0','rho03')],depth,e.read(a.parent/'metadata.json')['sources'],manifest)
    sources=e.source_identity(next(iter(parents.values())))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=e.ROOT,text=True).strip();e.verify_sources(sources,commit)
    a.out.mkdir(parents=True,exist_ok=False)
    metadata=dict(pid=os.getpid(),command=sys.argv,git_commit=commit,sources=sources,parent=str(a.parent.resolve()),
        profile_sha256=e.sha(e.parent_tools.PROFILE),plan_sha256=e.sha(e.HERE/'caltech_selected_extend_plan.json'),
        data_sha256=e.sha(a.data),manifest_sha256=e.sha(a.data.parent/'data_manifest.json'),parent_receipts=receipts,
        modes={str(d):4 if d in (2,4) else 1 for d in e.DEPTHS},total_epochs=200,additional_epochs=100,
        maximum_seconds_per_continuation=36000,gpu_uuids=e.GPUS,test_evaluations=0,time_unix=time.time(),scope=e.SCOPE)
    e.r.save(a.out/'metadata.json',metadata)
    jobs=[dict(depth=d,rho=r,name=f'L{d}_'+('rho0' if r==0 else 'rho03')) for d in (16,12,4,2) for r in (0.,.3)]
    pending,active,completed,failures,arms=jobs[:],{},[],[],{};started=time.time()
    while active or (pending and not failures):
        for gpu,(child,job) in list(active.items()):
            code=child.poll()
            if code is None:continue
            try:
                assert code==0,code
                fit=e.read(a.out/job['name']/'fit_result.json')
                assert fit['epochs_completed']==200 and fit['stop_reason']=='maximum_epochs' and fit['test_evaluations']==0
                completed.append(dict(job,exit_code=code));arms[job['name']]=str((a.out/job['name']).resolve())
            except Exception as error:failures.append(dict(job,exit_code=code,error=repr(error)))
            del active[gpu]
        if not failures:
            for gpu in e.GPUS:
                if gpu in active or not pending or not e.parent_tools.idle(gpu):continue
                e.verify_sources(sources,commit);job=pending.pop(0)
                cmd=[sys.executable,str(e.HERE/'caltech_selected_extend.py'),'--parent',str(parents[job['name']]),
                    '--out',str(a.out/job['name']),'--data',str(a.data),'--gpu',gpu]
                with (a.out/(job['name']+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(pid=child.pid,gpu_uuid=gpu,command=cmd);active[gpu]=(child,job)
        e.r.save(a.out/'current_processes.json',dict(active=[j for p,j in active.values()],pending=pending))
        e.r.save(a.out/'status.json',dict(state='fit_failure_draining' if failures else 'fit_running',completed=completed,failures=failures,
            pending=len(pending),active=len(active),seconds=time.time()-started,test_evaluations=0))
        if active or (pending and not failures):time.sleep(5)
    e.r.save(a.out/'fit_manifest.json',dict(arms=arms,data_sha256=e.sha(a.data),test_evaluations=0))
    e.r.save(a.out/'status.json',dict(state='fit_failed_no_automatic_retry' if failures else 'fit_complete_awaiting_validation_review',
        completed=completed,failures=failures,active=0,pending=pending,seconds=time.time()-started,test_evaluations=0))
    assert not failures,failures


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('parent','out','data'):p.add_argument('--'+name,type=Path,required=True)
    suite(p.parse_args())
