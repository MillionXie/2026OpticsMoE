import copy, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.vision_transfer import model as M, engine as E
from experiments.expert_merge import core as C
from experiments.office_transfer.runtime import setup
from eurosat_data import ROOT,PLAN,RECORDS,STAGES,Images,atomic,eval_batches,signature,file_sha

MOE=ROOT/'runs/moe'
D2=Path(PLAN['d2nn_root'])/'runs'

def build(architecture,out):
    loaded,s=setup(42,out)
    r,h=C.build(loaded,s) if architecture=='moe' else M.build(loaded,s,'d2nn')
    return loaded,s,r,h

def forward(loaded,r,h,inputs,mode='automatic',domain=None):
    return C.forward(loaded,r,h,inputs,mode,domain) if r.transfer_architecture=='moe' else M.predict(loaded,r,h,inputs)

def optimizer(r,h,architecture,stage,epoch,epochs,old=None):
    scale=.2+.8*.5*(1+math.cos(math.pi*(epoch-1)/max(1,epochs-1)))
    if architecture=='moe':
        if stage=='shared':
            rates=dict(electronic=2e-4,head=1e-3)
            if epoch>4:rates['shared_phase']=1e-3
        elif stage=='expert_A':rates={'expert_a':3e-3}
        elif stage=='expert_B':rates={'expert_b':3e-3}
        elif stage=='router':rates={'router':1e-3}
        else:raise ValueError(stage)
    else:
        rates=dict(electronic=2e-4,head=1e-3)
        if stage not in ('shared','A_only','B_only') or epoch>4:rates['shared_phase']=1e-3
    rates={g:lr*scale for g,lr in rates.items()};groups={}
    for name,p in M.named(r,h).items():
        group=M.group_of(name);p.grad=None;p.requires_grad_(name in r.transfer_eligible and group in rates)
        if p.requires_grad:groups.setdefault(group,[]).append(p)
    if old is not None and set(groups)=={g['group_name'] for g in old.param_groups}:
        for g in old.param_groups:g['lr']=rates[g['group_name']]
        return old
    return torch.optim.AdamW([dict(params=ps,group_name=g,lr=rates[g],weight_decay=0.) for g,ps in groups.items()])

@torch.no_grad()
def evaluate(loaded,r,h,s,domain,partition='validation',mode='automatic',dest=None,limit=None):
    start=time.perf_counter();labels=[];preds=[];logs=[];ids=[];prob=[];weights=[];total=0.
    for inputs,meta in E.base._prepared_batches(eval_batches(domain,partition,limit=limit),loaded,s):
        y=meta[:,0].to(loaded.device)
        with E.autocast(loaded,s):z=forward(loaded,r,h,inputs,mode,meta[:,1].to(loaded.device) if mode=='isolated' else None)
        if not bool(torch.isfinite(z).all()):raise RuntimeError('Nonfinite evaluation logits')
        total+=float(F.cross_entropy(z.float(),y,reduction='sum'))
        labels.append(meta[:,0].numpy());ids.append(meta[:,2].numpy());preds.append(z.argmax(1).cpu().numpy());logs.append(z.float().cpu().numpy())
        if r.transfer_architecture=='moe':
            q=M.routes(r)['vision'];prob.append(q['probabilities'].float().cpu().numpy());weights.append(q['weights'].float().cpu().numpy())
    y=np.concatenate(labels);p=np.concatenate(preds);idx=np.concatenate(ids);z=np.concatenate(logs)
    result=E.metrics_from_predictions(y,p,total/len(y),time.perf_counter()-start)
    result.update(domain=domain,partition=partition,routing_mode=mode,per_class_recall=(np.diag(np.array(result['confusion_matrix']))/np.maximum(1,np.bincount(y,minlength=10))).tolist())
    for key in ('spatial_group',):
        g=np.asarray([RECORDS[i][key] for i in idx]);result[key+'_accuracy']={str(v):dict(samples=int((g==v).sum()),accuracy=float((p[g==v]==y[g==v]).mean())) for v in np.unique(g)}
    extra={}
    if prob:
        pr=np.concatenate(prob);w=np.concatenate(weights)
        result['routes']=dict(mean_probability=pr.mean(0).tolist(),mean_power=np.square(w).mean(0).tolist(),maximum_power_error=float(np.abs(np.square(w).sum(1)-1).max()))
        extra=dict(route_probabilities=pr,route_weights=w)
    if dest:
        dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,indices=idx,labels=y,predictions=p,logits=z,**extra)
    return result

def validate(loaded,r,h,s,domains=('A','B'),mode='automatic'):
    val={d:evaluate(loaded,r,h,s,d,mode=mode) for d in domains}
    val['mean']=float(np.mean([val[d]['accuracy'] for d in domains]));val['mean_ce']=float(np.mean([val[d]['loss'] for d in domains]))
    return val

def checkpoint(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['plan_sha256']!=signature(PLAN) or state['split_sha256']!=PLAN['split_records_sha256']:raise RuntimeError('Checkpoint provenance mismatch')
    return state

def stamped(r,h,**extra):
    return dict(**M.clone(r,h),plan_sha256=signature(PLAN),split_sha256=PLAN['split_records_sha256'],**extra)

def phase_snapshot(r,h):return {n:p.detach().cpu().clone() for n,p in M.named(r,h).items() if n.endswith(('raw_phase','raw_router_phase'))}

def paths(model):
    return MOE/'router'/'selected.pt' if model=='moe' else D2/model/'selected.pt'
