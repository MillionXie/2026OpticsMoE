"""Train/validate the scaling study. Test data are never loaded by this entry."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import ScalingOptics
from .plan import load_protocol


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');os.replace(tmp,p)


def data(path):
    manifest=json.loads((path.parent/'data_manifest.json').read_text())
    if manifest.get('training_ready') is False:
        raise ValueError('Dataset audit is incomplete; training_ready=false')
    assert sha(path)==manifest['cache_sha256']
    z=np.load(path,allow_pickle=False);splits={}
    for split in ['train','val']:
        x=z[split+'_images'];y=z[split+'_labels'];ids=z[split+'_ids'].astype(str)
        assert x.ndim==4 and x.shape[-1]==3 and len(x)==len(y)==len(ids)
        assert ids.tolist()==manifest['split_ids'][split]
        splits[split]=(torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y.astype(np.int64)),ids)
    z.close()
    assert not set(splits['train'][2])&set(splits['val'][2])
    return splits,manifest


def images(x,indices,epoch,seed,augment=False):
    batch=x[indices].float().cuda()/255
    if augment:
        # Per-image/epoch keyed augmentation independent of architecture RNG.
        for j,index in enumerate(indices.tolist()):
            rng=random.Random((seed*1000003+epoch*1000033+index)*1000037)
            if rng.random()<.5:batch[j]=batch[j].flip(-1)
            if rng.random()<.5:batch[j]=batch[j].flip(-2)
            batch[j]=batch[j].rot90(rng.randrange(4),(-2,-1))
    return batch


@torch.no_grad()
def evaluate(model,split,batch=4):
    model.eval();x,y,ids=split;pred=[];captures=[];routes=[];masks=[]
    for start in range(0,len(y),batch):
        idx=torch.arange(start,min(start+batch,len(y)))
        out=model(images(x,idx,0,0))
        pred.append(out['probabilities'].cpu());captures.append(out['capture'].cpu())
        if out['route_mask'] is not None:
            routes.append(out['route_probabilities'].cpu());masks.append(out['route_mask'].cpu())
    p=torch.cat(pred);yp=p.argmax(1);nll=-p[torch.arange(len(y)),y].clamp_min(1e-12).log()
    cm=torch.bincount(y*model.classes+yp,minlength=model.classes**2).reshape(model.classes,model.classes).float()
    recall=cm.diag()/cm.sum(1).clamp_min(1);precision=cm.diag()/cm.sum(0).clamp_min(1)
    result=dict(accuracy=float((yp==y).float().mean()),balanced_accuracy=float(recall.mean()),
                macro_f1=float((2*precision*recall/(precision+recall).clamp_min(1e-12)).mean()),
                macro_nll=float(torch.stack([nll[y==i].mean() for i in range(model.classes)]).mean()),
                capture_mean=float(torch.cat(captures).mean()),confusion_matrix=cm.int().tolist())
    if masks:
        selected=torch.cat(masks);rp=torch.cat(routes)
        result.update(route_load=selected.mean(0).tolist(),route_probability=rp.mean(0).tolist(),
                      distinct_selected_sets=len(torch.unique(selected,dim=0)))
    return result,p


def objective(model,out,y,class_weights,cfg):
    c=model.classes;logp=out['probabilities'].clamp_min(1e-12).log()
    smooth=cfg['label_smoothing']
    ce=-((1-smooth)*logp[torch.arange(len(y),device=y.device),y]+smooth*logp.mean(1))
    ce=(ce*class_weights[y]).mean()
    loss=ce+cfg['capture_loss_weight']*(-out['capture'].clamp_min(1e-12).log()).mean()
    loss=loss+cfg['phase_tv_weight']*model.regularizer()
    if model.is_moe:
        mean=out['route_probabilities'].mean(0)
        loss=loss+cfg['router_balance_weight']*(model.n*mean.square().sum()-1)
    return loss


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arch',choices=['moe_oeo','d2nn_total_parameter','d2nn_same_aperture','d2nn_expert_global'],required=True)
    ap.add_argument('--experts',type=int,default=4);ap.add_argument('--top-k',type=int,default=4)
    ap.add_argument('--layers',type=int,default=6);ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--lr',type=float,default=.002);ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--microbatch',type=int,default=2);ap.add_argument('--padding',type=int,default=2)
    ap.add_argument('--smoke-limit',type=int,default=0)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException:
        save(a.out/'status.json',dict(state='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
    finally:
        torch.cuda.empty_cache()


def run(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    cfg=load_protocol();tc=cfg['training_proposal'];tc['learning_rate']=a.lr;tc['epochs']=a.epochs
    splits,manifest=data(a.data);classes=len(manifest['classes'])
    if a.smoke_limit:
        for key,(x,y,ids) in splits.items():
            ix=torch.cat([torch.where(y==i)[0][:max(1,a.smoke_limit//classes)] for i in range(classes)])
            splits[key]=(x[ix],y[ix],ids[ix.numpy()])
    model=ScalingOptics(cfg,a.experts,a.top_k,a.arch,classes,a.layers,a.padding).cuda()
    ema=copy.deepcopy(model).eval();ema.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0)
    x,y,ids=splits['train'];counts=torch.bincount(y,minlength=classes).float().cuda()
    weights=len(y)/(classes*counts)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    metadata=dict(command=sys.argv,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  protocol=cfg,geometry=model.geo,git_commit=commit,data_sha256=sha(a.data),
                  data_manifest_sha256=sha(a.data.parent/'data_manifest.json'),test_read=False,
                  device=torch.cuda.get_device_name(),torch_version=torch.__version__,pid=os.getpid(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),parameter_count=sum(p.numel() for p in model.parameters()),
                  dataset_limitations={k:manifest.get(k) for k in ['patient_independence_verified','mirror_repackaged','original_zip_pixel_equivalence_verified']})
    save(a.out/'metadata.json',metadata)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    best=math.inf;best_epoch=0;history=[];started=time.time()
    for epoch in range(1,a.epochs+1):
        model.train();order=torch.randperm(len(y),generator=torch.Generator().manual_seed(a.seed+epoch*100003))
        lr=a.lr*(.01+.99*(1+math.cos(math.pi*(epoch-1)/a.epochs))/2)
        for group in optimizer.param_groups:group['lr']=lr
        total=0.;steps=0
        for start in range(0,len(order),16):
            effective=order[start:start+16];optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(effective),a.microbatch):
                ix=effective[offset:offset+a.microbatch];yb=y[ix].cuda()
                out=model(images(x,ix,epoch,a.seed,True),dense=epoch<=tc['dense_warmup_epochs_included_in_total'])
                loss=objective(model,out,yb,weights,tc)*len(ix)/len(effective)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();total+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,1-tc['ema_decay'])
            steps+=1
        vm,vp=evaluate(ema,splits['val'],a.microbatch)
        entry=dict(epoch=epoch,train_objective=total/steps,val=vm,lr=lr,elapsed_seconds=time.time()-started,
                   peak_memory_bytes=torch.cuda.max_memory_allocated(),last_gradient_norm=float(norm))
        history.append(entry);save(a.out/'history.json',history)
        payload=dict(model=ema.state_dict(),live_model=model.state_dict(),optimizer=optimizer.state_dict(),
                     epoch=epoch,metadata=metadata,val=vm)
        torch.save(payload,a.out/'last_checkpoint.pt')
        if vm['macro_nll']<best:
            best=vm['macro_nll'];best_epoch=epoch;torch.save(payload,a.out/'best_checkpoint.pt')
            np.savez_compressed(a.out/'val_predictions.npz',ids=splits['val'][2],labels=splits['val'][1].numpy(),scores=vp.numpy())
        save(a.out/'status.json',dict(state='training',epoch=epoch,total_epochs=a.epochs,best_epoch=best_epoch,
                                     best_val_macro_nll=best,pid=os.getpid(),elapsed_seconds=time.time()-started))
        print(json.dumps(entry),flush=True)
    ck=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);ema.load_state_dict(ck['model'])
    train_metrics,_=evaluate(ema,splits['train'],a.microbatch)
    del ck
    audit={n:dict(rms_change=float((p.detach().cpu()-initial[n]).square().mean().sqrt()),
                  changed_fraction=float(((p.detach().cpu()-initial[n]).abs()>1e-7).float().mean()))
           for n,p in ema.named_parameters()}
    save(a.out/'phase_audit.json',audit)
    result=dict(best_epoch=best_epoch,val=history[best_epoch-1]['val'],train=train_metrics,
                checkpoint_sha256=sha(a.out/'best_checkpoint.pt'),test_read=False,git_commit=commit,
                elapsed_seconds=time.time()-started)
    save(a.out/'result.json',result);save(a.out/'status.json',dict(state='complete',pid=os.getpid(),**result))


if __name__=='__main__':main()
