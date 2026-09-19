"""Reproducible validation-only optical continual-learning experiment."""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .model import OpticalMoE,loss
from .data import load,balanced_indices,domain,sha
HERE=Path(__file__).parent


def save(path, value):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2,allow_nan=False)); temp.replace(path)


@torch.no_grad()
def evaluate(model, x, y, task, batch_size, mask=None):
    model.eval(); probs=[]; routes=[]
    device=next(model.parameters()).device
    for start in range(0,len(y),batch_size):
        out=model(domain(x[start:start+batch_size].to(device),task,model.cfg.get("task_b_view","color_shift")),mask=mask)
        probs.append(out['probabilities'].cpu()); routes.append(out['routes'].cpu())
    p=torch.cat(probs); q=torch.cat(routes); pred=p.argmax(1)
    confusion=torch.bincount(y*8+pred,minlength=64).reshape(8,8)
    return dict(accuracy=float((pred==y).float().mean()),loss=float(torch.nn.functional.nll_loss(p.clamp_min(1e-12).log(),y)),confusion=confusion.tolist(),mean_route=q.mean(0).tolist(),route_std=q.std(0,unbiased=False).tolist(),dominant_expert_counts=torch.bincount(q.argmax(1),minlength=12).tolist()),p,q


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=HERE/'configs/kather.json')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--pilot',action='store_true',help='One epoch per stage; structural/data validation only')
    args=parser.parse_args(); cfg=json.loads(args.config.read_text())
    if args.pilot: cfg.update(epochs_A=1,epochs_warmup=1,epochs_B=1)
    out=args.out.resolve(); base=(HERE/'runs'/('smoke' if args.pilot else 'simulation')).resolve()
    if base not in out.parents: raise ValueError('Output must be a fresh task runs/smoke or runs/simulation directory')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'status.json',dict(state='preparing'))
    try:
        torch.set_num_threads(4); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
        torch.backends.cudnn.benchmark=False
        arrays,manifest=load(args.data,args.manifest,cfg['seed'])
        x=torch.from_numpy(arrays['train_images']); y=torch.from_numpy(arrays['train_labels']).long()
        vx=torch.from_numpy(arrays['val_images']); vy=torch.from_numpy(arrays['val_labels']).long()
        aid,bid=arrays['A'],arrays['B']
        rid=aid[balanced_indices(arrays['train_labels'][aid],cfg['replay_capacity'],cfg['seed']+1)]
        split={name:arrays['train_ids'][ids].tolist() for name,ids in [('A',aid),('B',bid),('replay',rid)]}
        split['validation']=arrays['val_ids'].tolist()
        split['counts']={name:np.bincount(arrays['train_labels'][ids],minlength=8).tolist() for name,ids in [('A',aid),('B',bid),('replay',rid)]}
        save(out/'split.json',split)
        save(out/'config.json',cfg)
        save(out/'metadata.json',dict(command=sys.argv,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.version,torch=torch.__version__,platform=platform.platform(),device=args.device,data_sha256=sha(args.data),manifest=manifest,source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},scope='validation-only pilot' if args.pilot else 'validation-only experiment',test_images_read=False))
        model=OpticalMoE(cfg).to(args.device); initial_shapes={k:list(v.shape) for k,v in model.state_dict().items()}
        rng=np.random.default_rng(cfg['seed']); history=[]; audit=[]; a_before=None
        for stage in ('A','warmup','B'):
            model.configure(stage)
            frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
            groups=[dict(params=[p for p in model.experts if p.requires_grad],lr=cfg['lr_expert'])]
            if stage!='warmup': groups.append(dict(params=[model.router,model.global_phase],lr=cfg['lr_shared']))
            optimizer=torch.optim.Adam(groups,weight_decay=0.)
            best=-float('inf'); epochs=cfg['epochs_'+stage]; stage_dir=out/stage; stage_dir.mkdir()
            for epoch in range(1,epochs+1):
                model.train(); order=rng.permutation(aid if stage=='A' else bid)
                replay_n=round(cfg['batch_size']*cfg['replay_fraction']) if stage=='B' else 0
                current_n=cfg['batch_size']-replay_n; total=0.; count=0
                for start in range(0,len(order),current_n):
                    ids=order[start:start+current_n]; xb=domain(x[ids].to(args.device),'A' if stage=='A' else 'B',cfg.get('task_b_view','color_shift')); yb=y[ids].to(args.device)
                    if replay_n:
                        ri=rng.choice(rid,size=replay_n,replace=False)
                        xb=torch.cat((xb,x[ri].to(args.device))); yb=torch.cat((yb,y[ri].to(args.device)))
                    optimizer.zero_grad(set_to_none=True); output=model(xb,warmup=stage=='warmup'); value=loss(output,yb)
                    if not torch.isfinite(value): raise RuntimeError('Nonfinite loss')
                    value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
                    total+=value.item()*len(yb); count+=len(yb)
                ma,_,_=evaluate(model,vx,vy,'A',cfg['batch_size'])
                mb,_,_=evaluate(model,vx,vy,'B',cfg['batch_size'])
                row=dict(stage=stage,epoch=epoch,train_loss=total/count,val_A=ma,val_B=mb)
                history.append(row); save(out/'history.json',history)
                score=ma['accuracy'] if stage=='A' else (ma['accuracy']+mb['accuracy'])/2
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,stage=stage,epoch=epoch,validation=row)
                torch.save(state,stage_dir/'last_checkpoint.pt')
                if score>best:
                    best=score; torch.save(state,stage_dir/'best_checkpoint.pt')
                save(out/'status.json',dict(state='training',stage=stage,epoch=epoch))
                print(json.dumps(dict(stage=stage,epoch=epoch,train_loss=total/count,val_A=ma['accuracy'],val_B=mb['accuracy'])),flush=True)
            for n,p in model.named_parameters():
                if n in frozen and not torch.equal(p,frozen[n]): raise RuntimeError('Frozen parameter changed: '+n)
            if initial_shapes!={k:list(v.shape) for k,v in model.state_dict().items()}: raise RuntimeError('Geometry changed')
            audit.append(dict(stage=stage,frozen_unchanged=list(frozen),geometry_unchanged=True))
            # Warmup is fixed-duration; A/B use validation-selected checkpoints.
            if stage!='warmup':
                state=torch.load(stage_dir/'best_checkpoint.pt',map_location=args.device,weights_only=False); model.load_state_dict(state['model'])
            if stage=='A':
                a_before,prob,routes=evaluate(model,vx,vy,'A',cfg['batch_size'])
                np.savez_compressed(out/'A_before.npz',ids=arrays['val_ids'],labels=vy.numpy(),probabilities=prob.numpy(),routes=routes.numpy())
                b_before,_,_=evaluate(model,vx,vy,'B',cfg['batch_size'])
                save(out/'before_B.json',b_before)
        results={'A_before':a_before}
        for task in ('A','B'):
            for label,mask in [('all',None),('old_only',[True]*4+[False]*8),('new_only',[False]*4+[True]*4+[False]*4)]:
                metrics,prob,routes=evaluate(model,vx,vy,task,cfg['batch_size'],mask)
                results[task+'_'+label]=metrics
                np.savez_compressed(out/(task+'_'+label+'.npz'),ids=arrays['val_ids'],labels=vy.numpy(),probabilities=prob.numpy(),routes=routes.numpy())
        results['BWT']=results['A_all']['accuracy']-a_before['accuracy']
        results['new_to_old_ablation']=results['A_all']['accuracy']-results['A_old_only']['accuracy']
        results['old_to_new_ablation']=results['B_all']['accuracy']-results['B_new_only']['accuracy']
        results['interpretation']='Mask ablations change coherent interference; they do not isolate additive expert knowledge.'
        save(out/'metrics.json',results); save(out/'audit.json',audit); save(out/'status.json',dict(state='complete'))
    except BaseException as error:
        save(out/'status.json',dict(state='failed',error=repr(error))); raise

if __name__=='__main__': main()
