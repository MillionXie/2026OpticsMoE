"""Bounded repaired-graph validation. Reuses frozen fit primitives in this child only."""
import argparse,copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech_deploy234 as engine
from caltech_repair234_model import CorrectedMoE,input_amplitude,deployment_contract
import gpu_budget
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PLAN=HERE/'caltech_repair234_plan.json'
ALLOWED={2:'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b',5:'GPU-d53ce4c8-272d-c2fb-dc09-f182d586c4eb',6:'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'}
read,save,sha,digest=engine.read,engine.save,engine.sha,engine.digest
NEW_FILES=('caltech_repair234_model.py','caltech_repair234.py','caltech_repair234_plan.json','check_caltech_repair234.py','audit_caltech_repair234.py')

def sources():
    cfg=read(PLAN);src=dict(cfg['reference_sources'])
    for name,h in src.items():assert sha(ROOT/name)==h,name
    for name in NEW_FILES:src[(HERE/name).relative_to(ROOT).as_posix()]=sha(HERE/name)
    return src

def manifest(data):
    cfg=read(PLAN);m=read(data.parent/'data_manifest.json')
    assert sha(data)==cfg['cache_sha256']==m['cache_sha256'] and sha(data.parent/'data_manifest.json')==cfg['manifest_sha256']
    assert m['total_records']==1937 and len(m['classes'])==20 and sum(m['supports']['train'])==1246 and sum(m['supports']['val'])==261
    return m

@torch.no_grad()
def evaluate(model,data,micro):
    model.eval();device=model.raw_router.device;ps=[];captures=[];routes=[]
    for idx in torch.arange(len(data[1])).split(micro):
        p,c=model(input_amplitude(data[0][idx].to(device)));ps.append(p.cpu().numpy());captures.append(c.cpu().numpy());routes.append(model.routing_weights.cpu().numpy())
    p=np.concatenate(ps).astype(np.float64);w=np.concatenate(routes).astype(np.float64);y=data[1].numpy();metric=engine.metrics(y,p);metric['detector_capture']=float(np.concatenate(captures).mean())
    metric['routing']=dict(nonzero_fraction=(w>0).mean(0).tolist(),weight_mean=w.mean(0).tolist(),weight_std=w.std(0).tolist(),weight_min=w.min(0).tolist(),weight_max=w.max(0).tolist(),per_class_weight_mean=[w[y==k].mean(0).tolist() for k in range(20)],ranked_top2_diagnostic_not_hard_gate=True)
    rows=[dict(sample_id=str(sid),label_true=int(label),label_pred=int(v.argmax()),**{f'score{k}':float(v[k]) for k in range(20)}) for sid,label,v in zip(data[2],data[1],p)]
    return metric,rows

def fresh_gate(depth,rho,data,cfg,device,micro,state=None):
    assert micro==1;engine.seed(cfg['seed']);model=CorrectedMoE(depth,rho).to(device)
    if state is not None:model.load_state_dict(state,strict=True)
    initial=digest(model.state_dict());weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).to(device)
    idx=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]);_,theta=engine.epoch_plan(1,len(data[1]),cfg);x,y,_=engine.mixed_batch(data,idx,theta,1,0,cfg,device)
    def compute():
        model.zero_grad(set_to_none=True);ps=[]
        for k in range(20):
            p,c=model(x[k:k+1]);(engine.objective(model,p,c,y[k:k+1],weights,cfg)/20).backward();ps.append(p.detach())
        return torch.cat(ps),{n:p.grad.detach().clone() for n,p in model.named_parameters()}
    p,g=compute();q,h=compute();error=float((p-q).abs().max());finite=all(torch.isfinite(v).all() and v.norm()>0 for v in h.values())
    rel=max(float((g[n]-v).norm()/v.norm().clamp_min(1e-30)) for n,v in h.items());cos=min(float(torch.sum(g[n].double()*v.double())/(g[n].double().norm()*v.double().norm()).clamp_min(1e-30)) for n,v in h.items())
    assert error<=1e-6 and rel<=.02 and cos>=.999 and finite,(error,rel,cos,finite)
    result=dict(passed=True,depth=depth,rho=rho,micro=1,classes_checked=20,probability_max_error=error,gradient_max_relative_l2=rel,gradient_min_cosine=cos,all_phase_gradients_finite_nonzero=bool(finite),phase_gradients={n:float(v.norm()) for n,v in h.items()},initial_parameter_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),test_evaluations=0)
    del model;torch.cuda.empty_cache() if device=='cuda' else None
    return result

def bind(alpha):
    # Explicit process-local dependencies; sealed engine/model source files are never edited.
    cfg=read(PLAN);assert alpha in cfg['candidates']
    original_read=read
    def profile_read(path):
        result=original_read(path)
        if Path(path)==PLAN:result=dict(result,mixup_alpha=alpha,study_case='mixup'+str(alpha))
        return result
    engine.PROFILE=PLAN;engine.read=profile_read;engine.DeployMoE=CorrectedMoE;engine.input_amplitude=input_amplitude;engine.deployment_contract=deployment_contract;engine.ALLOWED=ALLOWED;engine.sources=sources;engine.manifest=manifest;engine.evaluate=evaluate;engine.fresh_gate=fresh_gate
    return cfg

def fit(a):
    assert a.depth==4 and a.rho==.3,'This runner fits only the three declared L4 residual study candidates'
    bind(a.alpha);a.resume_parent=None;engine.fit(a)
    result=read(a.out/'fit_result.json');hist=read(a.out/'history.json');tm=result['metrics']['train'];vm=result['metrics']['val'];g=read(PLAN)['formal_gate'];tail=hist[-min(20,len(hist)):]
    checks=dict(val_accuracy=vm['accuracy']>=g['L4_val_accuracy_at_selected_ema_min'],val_macro=vm['balanced_accuracy']>=g['L4_val_macro_recall_min'],no_zero_train_val=min(tm['recall'])>0 and min(vm['recall'])>0,val_each_recall=min(vm['recall'])>=g['selected_val_each_recall_min'],no_persistent_zero=all(sum(h['val']['recall'][k]==0 for h in tail)<=len(tail)/2 for k in range(20)),gap=100*(tm['accuracy']-vm['accuracy'])<=15,routing_all_slots=all(v==1 for split in (tm,vm) for v in split['routing']['nonzero_fraction']),routing_input_dependent=all(max(split['routing']['weight_std'])>1e-4 for split in (tm,vm)))
    save(a.out/'validation_gate.json',dict(passed=all(checks.values()),checks=checks,selected_epoch=result['selected_epoch'],gap_pp=100*(tm['accuracy']-vm['accuracy']),test_evaluations=0,scope='Performance eligibility only; failure does not retry or test. Study results require saved CPU audit before choosing profile.'))

def preflight(a):
    cfg=bind(.4);assert a.gpu in ALLOWED.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;gpu_budget.ALLOWED.update(ALLOWED.values());policy=gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);m=manifest(Path(cfg['data']));data=engine.load_fit(Path(cfg['data']),'train',m);a.out.mkdir(exist_ok=False,parents=True);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);checks=[]
    for depth in (2,3,4):
        for rho in (0.,.3):checks.append(fresh_gate(depth,rho,data,cfg,'cuda',1))
    for depth in (2,3,4):assert checks[(depth-2)*2]['initial_parameter_sha256']==checks[(depth-2)*2+1]['initial_parameter_sha256']
    save(a.out/'checks.json',dict(passed=True,checks=checks,policy=policy,sources=src,git_commit=commit,test_arrays_materialized=0,test_evaluations=0))

def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cfg=read(PLAN);a.out.mkdir(exist_ok=False,parents=True);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,config=cfg,command=sys.argv,phase='bounded_L4_residual_validation',test_evaluations=0));pending=list(cfg['candidates']);active=[];complete=[];failures=[]
    while pending or active:
        for job in list(active):
            rc=job['process'].poll()
            if rc is not None:
                job['log'].close();record={k:v for k,v in job.items() if k not in ('process','log')};record['exit_code']=rc;(complete if rc==0 else failures).append(record);active.remove(job)
        if failures:pending=[]
        busy={j['gpu_uuid'] for j in active}
        for index in (6,2,5):
            gpu=ALLOWED[index]
            if not pending or gpu in busy or failures:continue
            gpu_budget.ALLOWED.update(ALLOWED.values())
            try:gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            alpha=pending.pop(0);case='L4_rho03_mixup'+str(int(round(alpha*10))).zfill(2);dest=a.out/case;assert not dest.exists();log=(a.out/(case+'.log')).open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu);cmd=[sys.executable,str(HERE/'caltech_repair234.py'),'--phase','fit','--depth','4','--rho','.3','--alpha',str(alpha),'--gpu',gpu,'--out',str(dest),'--data',cfg['data']];proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);active.append(dict(case=case,alpha=alpha,pid=proc.pid,gpu_uuid=gpu,process=proc,log=log));busy.add(gpu)
        save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending))
        save(a.out/'status.json',dict(state='study_failed' if failures else 'study_running' if active or pending else 'study_complete_awaiting_CPU_review',completed=complete,failures=failures,active=len(active),pending=len(pending),test_evaluations=0))
        if active or pending:time.sleep(5)
    engine.verify_sources(src,commit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','queue','preflight'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path);p.add_argument('--depth',type=int);p.add_argument('--rho',type=float);p.add_argument('--alpha',type=float,default=.4);p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),pid=os.getpid(),automatic_retry=False,test_evaluations=0))
        raise
