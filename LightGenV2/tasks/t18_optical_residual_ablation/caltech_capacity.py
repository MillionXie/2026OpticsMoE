"""Exactly three human-authorized phase-capacity fits; frozen engines unchanged."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import caltech_repair234 as repaired
from caltech_capacity_model import CapacityMoE, GRIDS, COUNTS, input_amplitude, deployment_contract
import gpu_budget
engine=repaired.engine
read,save,sha,digest=repaired.read,repaired.save,repaired.sha,repaired.digest
PLAN=HERE/'caltech_capacity_plan.json'; ALLOWED=repaired.ALLOWED
NEW_FILES=('caltech_capacity_model.py','caltech_capacity.py','caltech_capacity_plan.json',
           'check_caltech_capacity.py','audit_caltech_capacity.py','export_caltech_capacity.py',
           'CALTECH_CAPACITY_PROTOCOL_20261014.md')

def sources():
    src=repaired.sources()
    for name in ('export_caltech_repair234.py','check_caltech_repair234_export.py')+NEW_FILES:
        src[(HERE/name).relative_to(ROOT).as_posix()]=sha(HERE/name)
    return src

def config(grid):
    assert grid in GRIDS
    cfg=read(repaired.PLAN)
    cfg.update(read(PLAN)['profile_overrides'])
    cfg.update(phase_grid=grid, study_case='phase_'+grid,
               scope='New matched-initialization phase-capacity mechanism probe, not formal six or adoption of a smaller architecture')
    return cfg

def gate(depth,rho,data,cfg,device,micro,state=None):
    assert depth==4 and rho==.3 and micro==1 and state is None
    engine.seed(cfg['seed']); model=CapacityMoE(depth,rho,cfg['phase_grid']).to(device)
    initial=digest(model.state_dict()); weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).to(device)
    idx=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)])
    _,theta=engine.epoch_plan(1,len(data[1]),cfg); x,y,_=engine.mixed_batch(data,idx,theta,1,0,cfg,device)
    def compute(order):
        model.zero_grad(set_to_none=True); ps={}
        for k in order:
            p,c=model(x[k:k+1]); (engine.objective(model,p,c,y[k:k+1],weights,cfg)/20).backward(); ps[k]=p.detach()
        return torch.cat([ps[k] for k in range(20)]), {n:p.grad.detach().clone() for n,p in model.named_parameters()}
    p,g=compute(range(20)); q,h=compute(reversed(range(20)))
    error=float((p-q).abs().max()); finite=all(torch.isfinite(v).all() and v.norm()>0 for v in h.values())
    rel=max(float((g[n]-v).norm()/v.norm().clamp_min(1e-30)) for n,v in h.items())
    cos=min(float(torch.sum(g[n].double()*v.double())/(g[n].double().norm()*v.double().norm()).clamp_min(1e-30)) for n,v in h.items())
    assert error<=1e-6 and rel<=.02 and cos>=.999 and finite,(error,rel,cos,finite)
    result=dict(passed=True,depth=depth,rho=rho,phase_grid=cfg['phase_grid'],micro=1,classes_checked=20,
                probability_max_error=error,gradient_max_relative_l2=rel,gradient_min_cosine=cos,
                all_phase_gradients_finite_nonzero=bool(finite),phase_gradients={n:float(v.norm()) for n,v in h.items()},
                parameter_shapes={n:list(v.shape) for n,v in model.named_parameters()},
                initial_parameter_sha256=initial,router_initial_sha256=digest(model.raw_router.detach().cpu().numpy()),
                parameters=sum(p.numel() for p in model.parameters()),test_evaluations=0)
    del model; torch.cuda.empty_cache() if device=='cuda' else None
    return result

def bind(grid):
    cfg=config(grid)
    original_read=read
    def profile_read(path):
        return config(grid) if Path(path)==PLAN else original_read(path)
    engine.PROFILE=PLAN;engine.read=profile_read
    engine.DeployMoE=lambda depth,rho:CapacityMoE(depth,rho,grid)
    engine.input_amplitude=input_amplitude;engine.deployment_contract=lambda depth,rho:deployment_contract(depth,rho,grid)
    engine.ALLOWED=ALLOWED;engine.sources=sources;engine.manifest=repaired.manifest
    engine.evaluate=repaired.evaluate;engine.fresh_gate=gate
    return cfg

def eligibility(fit,hist):
    tm,vm=fit['metrics']['train'],fit['metrics']['val'];tail=hist[-min(20,len(hist)):]
    checks=dict(val_accuracy=vm['accuracy']>=.65,val_macro=vm['balanced_accuracy']>=.60,
                no_zero_train_val=min(tm['recall'])>0 and min(vm['recall'])>0,val_each_recall=min(vm['recall'])>=.20,
                no_persistent_zero=all(sum(h['val']['recall'][k]==0 for h in tail)<=len(tail)/2 for k in range(20)),
                gap=100*(tm['accuracy']-vm['accuracy'])<=15,
                routing_all_slots=all(v==1 for split in (tm,vm) for v in split['routing']['nonzero_fraction']),
                routing_input_dependent=all(max(split['routing']['weight_std'])>1e-4 for split in (tm,vm)))
    return dict(passed=all(checks.values()),checks=checks,selected_epoch=fit['selected_epoch'],
                gap_pp=100*(tm['accuracy']-vm['accuracy']),test_evaluations=0)

def prerequisites():
    cfg=read(PLAN)
    for kind in ('cpu_native','gpu_numeric'):
        path=ROOT/cfg['gates'][kind]; proof=read(path)
        assert proof['passed'] and proof['sources']==sources(),kind
        engine.verify_sources(proof['sources'],proof['git_commit'])
    return cfg

def fit(a):
    prerequisites();assert a.depth==4 and a.rho==.3 and a.data==Path(config(a.grid)['data'])
    bind(a.grid);a.resume_parent=None;engine.fit(a)
    f=read(a.out/'fit_result.json');save(a.out/'validation_gate.json',eligibility(f,read(a.out/'history.json')))

def preflight(a):
    assert a.gpu in ALLOWED.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit)
    gpu_budget.ALLOWED.update(ALLOWED.values());policy=gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2)
    torch.set_num_threads(4);cfg=config('full');m=repaired.manifest(Path(cfg['data']));data=engine.load_fit(Path(cfg['data']),'train',m)
    a.out.mkdir(parents=True,exist_ok=False);checks=[]
    for grid in GRIDS:
        bind(grid);checks.append(gate(4,.3,data,config(grid),'cuda',1))
    assert len({c['router_initial_sha256'] for c in checks})==1
    save(a.out/'checks.json',dict(passed=True,checks=checks,policy=policy,sources=src,git_commit=commit,test_arrays_materialized=0,test_evaluations=0),exclusive=True)

def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=prerequisites();src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit)
    assert not a.out.exists();a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,plan=plan,command=sys.argv,
                                   phase='authorized_bounded_capacity_probe_not_formal',test_evaluations=0))
    pending=list(GRIDS);active=[];complete=[];failures=[]
    while pending or active:
        for job in list(active):
            rc=job['process'].poll()
            if rc is not None:
                job['log'].close();record={k:v for k,v in job.items() if k not in ('process','log')};record['exit_code']=rc
                (complete if rc==0 else failures).append(record);active.remove(job)
        if failures:pending=[]
        busy={j['gpu_uuid'] for j in active}
        for index in (6,2,5):
            gpu=ALLOWED[index]
            if not pending or gpu in busy or failures:continue
            gpu_budget.ALLOWED.update(ALLOWED.values())
            try:gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            grid=pending.pop(0);case='L4_rho03_phase_'+grid;dest=a.out/case;assert not dest.exists()
            log=(a.out/(case+'.log')).open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
            cmd=[sys.executable,str(HERE/'caltech_capacity.py'),'--phase','fit','--depth','4','--rho','.3','--grid',grid,
                 '--gpu',gpu,'--out',str(dest),'--data',config(grid)['data']]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append(dict(case=case,grid=grid,pid=proc.pid,gpu_uuid=gpu,process=proc,log=log));busy.add(gpu)
        save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending))
        save(a.out/'status.json',dict(state='capacity_failed' if failures else 'capacity_running' if active or pending else 'capacity_complete_awaiting_CPU_audit',
             completed=complete,failures=failures,active=len(active),pending=len(pending),test_evaluations=0))
        if active or pending:time.sleep(5)
    engine.verify_sources(src,commit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','queue','preflight'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--grid',choices=GRIDS,default='full');p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),pid=os.getpid(),automatic_retry=False,test_evaluations=0))
        raise
