import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Subset
from . import preprocessing as base
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.data import build_train_loader, build_eval_loader
from . import model as M
from .data import atomic_json, pair_loader, validation_b, ArrayDataset, CORRUPTIONS

from .protocol import load_policy
from .routing_objective import routing_loss as route_target_loss

VERSION=load_policy()['version']

def append(path,row):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def save(path,payload):
    path=Path(path);t=path.with_suffix(path.suffix+'.tmp');torch.save(payload,t);os.replace(t,path)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def autocast(loaded,s):
    return torch.autocast(device_type=loaded.device.type,dtype=torch.bfloat16,enabled=s.amp_enabled)

def routing_loss(r,domain,teacher=None,clean=None):
    # Teacher route imitation is removed; logit KD remains in train.
    return route_target_loss(M.routes(r),domain,load_policy()['loss'])


def metrics_from_predictions(y,p,loss,elapsed,route_info=None):
    confusion=np.bincount(10*y+p,minlength=100).reshape(10,10)
    tp=np.diag(confusion);f1=2*tp/np.maximum(1,confusion.sum(0)+confusion.sum(1))
    return dict(accuracy=float((y==p).mean()),loss=float(loss),macro_f1=float(f1.mean()),
                samples=len(y),confusion_matrix=confusion.tolist(),elapsed_sec=elapsed,
                images_per_sec=len(y)/max(elapsed,1e-9),routes=route_info or {})

@torch.no_grad()
def evaluate(loaded,r,h,ds,s,*,predictions_path=None,return_outputs=False,oracle_domain=None):
    M.set_mode(loaded,r,h,False);M.force_route(r,None)
    ys=[];ps=[];logs=[];route_probs={};sel={};power={};total=0.;count=0;start=time.perf_counter()
    for inputs,ycpu in base._prepared_batches(build_eval_loader(ds,s),loaded,s):
        y=ycpu.to(loaded.device)
        if oracle_domain is not None:
            for _,sur in M.surrogates(r):sur.core.optical_branch.core.router.train(True)
            M.force_route(r,torch.full_like(y,oracle_domain))
        with autocast(loaded,s):logits=M.classification_logits(loaded.model,r,h,inputs)[0]
        total+=float(F.cross_entropy(logits,y,reduction='sum'));count+=len(y)
        ys.append(ycpu.numpy());ps.append(logits.argmax(1).cpu().numpy())
        if return_outputs:logs.append(logits.float().cpu())
        for mod,q in M.routes(r).items():
            selected=q['selected_mask'].float().sum(0).cpu().numpy();pw=q['weights'].detach().float().square().sum(0).cpu().numpy()
            sel[mod]=sel.get(mod,np.zeros(4))+selected;power[mod]=power.get(mod,np.zeros(4))+pw
            if return_outputs:route_probs.setdefault(mod,[]).append(q['probabilities'].float().cpu())
    y=np.concatenate(ys);p=np.concatenate(ps)
    info={m:dict(selection_share=(v/(count*4)).tolist(),power_share=(power[m]/count).tolist()) for m,v in sel.items()}
    metrics=metrics_from_predictions(y,p,total/count,time.perf_counter()-start,info)
    M.set_mode(loaded,r,h,False)
    if predictions_path:
        Path(predictions_path).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(predictions_path,labels=y,predictions=p)
    if return_outputs:return metrics,torch.cat(logs),{m:torch.cat(v) for m,v in route_probs.items()}
    return metrics

def evaluate_corrupted(loaded,r,h,s,conditions,folder=None):
    records=[]
    for kind,sev,ds in conditions:
        dest=None if folder is None else Path(folder)/f'{kind}_{sev}.npz'
        result=evaluate(loaded,r,h,ds,s,predictions_path=dest)
        records.append(dict(corruption=kind,severity=sev,**result))
    return dict(accuracy=float(np.mean([v['accuracy'] for v in records])),
                macro_f1=float(np.mean([v['macro_f1'] for v in records])),conditions=records)

def phase_stats(r,h,initial):
    result={}
    for n,p in M.named(r,h).items():
        if not n.endswith(('raw_phase','raw_router_phase')):continue
        pref,path=n.split('.',1); module=dict(M.modules(r,h))[pref].get_submodule(path.rsplit('.',1)[0])
        phase=lambda x: x.float() if getattr(module,'parameterization','sigmoid')=='unconstrained' else 2*torch.pi*x.float().sigmoid()
        a=phase(p.detach());b=phase(initial[n].to(p.device));delta=torch.atan2(torch.sin(a-b),torch.cos(a-b))
        result[n]=dict(rms_change_rad=float(delta.square().mean().sqrt()),phase_std_rad=float(a.std()))
    return result

def phase_gradients(r,h,ce):
    ps={n:p for n,p in M.named(r,h).items() if p.requires_grad and n.endswith(('raw_phase','raw_router_phase'))}
    grads=torch.autograd.grad(ce,list(ps.values()),retain_graph=True,allow_unused=True)
    values={n:dict(l2=0. if g is None else float(g.float().norm()),finite=g is None or bool(torch.isfinite(g).all())) for n,g in zip(ps,grads)}
    if any(not x['finite'] for x in values.values()):raise RuntimeError('Nonfinite CE-only phase gradients')
    return values

def export_visuals(r,h,initial,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    masks={};changes={};fields={}
    for n,p in M.named(r,h).items():
        if not n.endswith(('raw_phase','raw_router_phase')):continue
        pref,path=n.split('.',1);module=dict(M.modules(r,h))[pref].get_submodule(path.rsplit('.',1)[0])
        convert=lambda v:v.float() if getattr(module,'parameterization','sigmoid')=='unconstrained' else 2*torch.pi*v.float().sigmoid()
        a=convert(p.detach()).cpu().numpy();b=convert(initial[n]).numpy()
        masks[n]=a;changes[n]=np.angle(np.exp(1j*(a-b)))
    for mod,sur in M.surrogates(r):
        path=sur.core.optical_branch
        for key in ('last_expert_input_amplitude','last_global_input_amplitude','last_raw_expert_ccd','last_raw_ccd'):
            value=getattr(path,key,None)
            if value is not None:fields[mod+'.'+key]=value[0].float().cpu().numpy()
        value=getattr(path.core.router,'last_detector_intensity',None)
        if value is not None:fields[mod+'.router_ccd']=value[0].float().cpu().numpy()
    for label,values in [('phase_masks',masks),('phase_changes',changes),('fields',fields)]:
        np.savez_compressed(folder/(label+'.npz'),**values)
        cols=4;fig,axs=plt.subplots(math.ceil(len(values)/cols),cols,figsize=(15,3.5*math.ceil(len(values)/cols)),squeeze=False)
        for ax,(name,v) in zip(axs.flat,values.items()):
            im=np.log1p(np.maximum(v,0)) if label=='fields' else v
            opts={} if label=='fields' else dict(vmin=0 if label=='phase_masks' else -.5,vmax=2*np.pi if label=='phase_masks' else .5)
            ax.imshow(im,cmap='magma' if label=='fields' else 'twilight' if label=='phase_masks' else 'RdBu_r',**opts)
            ax.set_title(name.replace('core.optical_branch.core.',''),fontsize=7);ax.axis('off')
        for ax in list(axs.flat)[len(values):]:ax.axis('off')
        fig.tight_layout();fig.savefig(folder/(label+'.png'),dpi=120);plt.close(fig)

def train(*args,**kwargs):
    from .train import train as implementation
    return implementation(*args,**kwargs)
