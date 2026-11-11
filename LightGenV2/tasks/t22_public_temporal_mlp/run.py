"""Train/evaluate the approved serial model. Training never constructs Qwen."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tempfile

import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr, kendalltau, pearsonr

if __package__:
    from .model import LightGenTemporal, ModelConfig, load_lightgen
    from .data import load_inputs, groups, batch, sha256
else:
    from model import LightGenTemporal, ModelConfig, load_lightgen
    from data import load_inputs, groups, batch, sha256

ROOT=Path(__file__).resolve().parent


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf-8')
    temp.replace(path)


def save_checkpoint(path,payload):
    """Complete and flush a temporary file before replacing a valid checkpoint."""
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='w+b',dir=path.parent,
            prefix='.'+path.name+'.',suffix='.writing',delete=False) as f:
        temp=Path(f.name)
        torch.save(payload,f)
        f.flush();os.fsync(f.fileno())
    for attempt in range(8):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.25*(attempt+1))


def correlation(x,y):
    x=x-x.mean();y=y-y.mean()
    return (x*y).sum()/(x.square().sum()*y.square().sum()).clamp_min(1e-12).sqrt()


def router_regularization(routes,valid):
    values={k:[] for k in ['balance','importance','diversity']}
    stats={}
    for name,route in routes.items():
        p=route['probabilities'];s=route['selected'].float();v=valid
        if name=='frame':v=v[...,None].expand(*v.shape,4)
        # Keep physical slot/frame identities while averaging valid samples.
        count=v.float().sum(0).clamp_min(1)[...,None]
        importance=(p*v[...,None]).sum(0)/count
        load=(s*v[...,None]).sum(0)/count/2
        conditional=(-(p.clamp_min(1e-8).log()*p).sum(-1)*v).sum()/v.sum().clamp_min(1)/math.log(4)
        marginal=-(importance.clamp_min(1e-8).log()*importance).sum(-1).mean()/math.log(4)
        values['balance'].append((4*(importance*load).sum(-1)).mean())
        values['importance'].append((4*importance.square().sum(-1)-1).mean())
        values['diversity'].append(conditional-marginal)
        stats[name]={'mean_probability':importance.mean(tuple(range(importance.ndim-1))).detach().cpu().tolist(),
                     'selection_share':load.mean(tuple(range(load.ndim-1))).detach().cpu().tolist()}
    return {k:torch.stack(v).mean() for k,v in values.items()},stats


def regression_metrics(prediction,target):
    x=np.asarray(prediction,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
    out={'n':len(y),'srcc':float(spearmanr(x,y).statistic),'krcc':float(kendalltau(x,y).statistic),
         'plcc':float(pearsonr(x,y).statistic),'rmse':float(np.sqrt(np.mean((x-y)**2))),
         'mae':float(np.mean(np.abs(x-y)))}
    if not all(np.isfinite(v) for v in out.values()):raise ValueError('Nonfinite evaluation metric')
    return out


@torch.no_grad()
def evaluate(model,payload,device,batch_fields,out=None,epoch=None,slot_shift=0):
    model.eval();idx,valid=groups(payload,'test');preds=[];truth=[];all_indices=[]
    route_samples={'frame':[],'video':[]}
    for start in range(0,len(idx),batch_fields):
        ii=idx[start:start+batch_fields];vv=valid[start:start+batch_fields]
        if slot_shift:ii=ii.roll(slot_shift,1);vv=vv.roll(slot_shift,1)
        item=batch(payload,ii,vv,device)
        result=model(item['vision']);keep=item['valid']
        preds.extend(result['prediction'][keep].cpu().tolist());truth.extend(item['target'][keep].cpu().tolist())
        all_indices.extend(ii[vv].tolist())
        for name,r in result['routing'].items():
            selected=r['selected'][keep].cpu()
            route_samples[name].append(selected)
    if len(set(all_indices))!=558 or len(all_indices)!=558:raise ValueError('Evaluation IDs/padding mismatch')
    metrics=regression_metrics(preds,truth)
    for name,pieces in route_samples.items():
        selected=torch.cat(pieces,0).reshape(-1,4)
        codes=(selected.long()*torch.tensor([1,2,4,8])).sum(-1)
        counts=torch.bincount(codes,minlength=16)
        metrics[name+'_routing']={'selection_share':(selected.float().mean(0)/2).tolist(),
            'selection_variation':float(1-counts.max()/len(codes)),
            'unique_top2_pairs':int((counts>0).sum())}
    if out:
        with (Path(out)/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['sample_id','target','prediction','epoch'])
            for index,target,pred in zip(all_indices,truth,preds):writer.writerow([payload['ids'][index],target,pred,epoch])
        write(Path(out)/'metrics.json',metrics)
    return metrics


def ensure_idle_gpu():
    """Fail before creating a CUDA context if the selected GPU is occupied."""
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible or ',' in visible:raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one GPU UUID/index')
    text=subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=uuid,memory.used,memory.total,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True).strip()
    uuid,memory,total,util=[v.strip() for v in text.split(',')]
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'],text=True)
    if float(total)-float(memory)<6144 or float(util)>5 or any(line.startswith(uuid+',') for line in procs.splitlines()):
        raise RuntimeError('Selected GPU occupied; refusing to share: '+text)
    return text


def train(args):
    cfg=json.loads(args.config.read_text('utf-8'));paths=json.loads(args.paths.read_text('utf-8'))
    out=args.out.resolve()
    if out.exists():raise FileExistsError('Refusing to overwrite run: '+str(out))
    if args.device.startswith('cuda') and not args.allow_shared_gpu:admission=ensure_idle_gpu()
    else:admission='local smoke or explicitly allowed device; no exclusive-resource measurement claim'
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    payload=load_inputs(paths)
    small_front=torch.load(paths['front'],map_location='cpu',weights_only=True)
    if not torch.equal(small_front['prompt_embedding'].to(payload['prompt']),payload['prompt']):
        raise ValueError('Exported prompt/cache mismatch')
    mc=ModelConfig(**cfg.get('model',{}))
    if mc.prompt_length!=len(payload['prompt']):raise ValueError('Prompt length mismatch')
    model=LightGenTemporal(mc);model.front.load_state_dict(small_front['front'],strict=True)
    model.set_prompt(payload['prompt'])
    train_idx=[i for i,s in enumerate(payload['splits']) if s=='train']
    train_y=payload['target'][train_idx]
    model.target_mean.copy_(train_y.mean());model.target_std.copy_(train_y.std(unbiased=False))
    initialization=None
    if args.paper_initialize:
        if args.initialize:raise ValueError('Select one initialization source')
        paper=torch.load(args.paper_initialize,map_location='cpu',weights_only=False)
        source=paper['state_dict'];target=model.state_dict();copied=[]
        direct=['vision_adapter.','language_adapter.','visual_input_norm.','prompt_to_visual.','frame_merger.']
        globals_map={'frame_global.':'parallel_optics.','video_global.':'serial_optics.'}
        for name in target:
            original=None
            if any(name.startswith(prefix) for prefix in direct) or name in ['frame_position','sequence_position']:
                original=name
            for new_prefix,old_prefix in globals_map.items():
                if name.startswith(new_prefix):
                    suffix=name[len(new_prefix):]
                    if suffix.startswith(('width_to_field.','tokens_to_field.')) or suffix=='raw_global_phase':
                        original=old_prefix+suffix
                    elif suffix.startswith('norm.'):
                        original=old_prefix+'global_readout.'+suffix
                    elif suffix.startswith('output.'):
                        original=old_prefix+'global_readout.'+suffix
            if original is not None:
                if original not in source or source[original].shape!=target[name].shape:
                    raise ValueError('Paper initialization shape mismatch: '+str(original))
                target[name].copy_(source[original]);copied.append({'target':name,'source':original})
        model.load_state_dict(target,strict=True)
        initialization={'path':str(args.paper_initialize),'sha256':sha256(args.paper_initialize),
                        'kind':'explicit front/merger/global whitelist; fresh MLP router/experts and final scoring head',
                        'copied_tensors':copied,'optimizer':'new AdamW and cosine schedule'}
    if args.initialize:
        parent,record=load_lightgen(args.initialize,'cpu')
        if parent.cfg!=mc:raise ValueError('Initialization architecture differs')
        if record['dataset']['manifest']['sha256']!=payload['identity']['manifest']['sha256']:
            raise ValueError('Initialization data identity differs')
        model.load_state_dict(parent.state_dict(),strict=True)
        initialization={'path':str(args.initialize),'sha256':sha256(args.initialize),
                        'parent_epoch':record.get('epoch'),'optimizer':'new AdamW and cosine schedule; weights-only initialization'}
    if args.device.startswith('cuda') and not args.allow_shared_gpu:
        admission=ensure_idle_gpu()  # Recheck after cache/hash work, before CUDA allocation.
    model=model.to(args.device)
    phase=[];electronic=[]
    for name,param in model.named_parameters():
        (phase if name.endswith('raw_global_phase') else electronic).append(param)
    optimizer=torch.optim.AdamW([{'params':electronic,'lr':cfg['learning_rate']},
                                 {'params':phase,'lr':cfg['phase_learning_rate']}],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['epochs'])
    out.mkdir(parents=True)
    try:head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):head='unavailable'
    metadata={'role':'PUBLIC_SURROGATE_DEVELOPMENT','test_used_for_selection':True,
              'test_used_for_gradient':False,'validation_used':False,'config':cfg,
              'source_commit':os.environ.get('LIGHTGEN_SOURCE_COMMIT',head),'checkout_HEAD':head,
              'source_SHA256':{p.name:sha256(p) for p in ROOT.glob('*.py')},
              'dataset':payload['identity'],'front_source':small_front['source'],
              'python':platform.python_version(),'torch':str(torch.__version__),'platform':platform.platform(),
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu_name':torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
              'gpu_admission':admission,'command':sys.argv,'pid':os.getpid(),
              'model_parameters':sum(p.numel() for p in model.parameters()),'model_config':asdict(mc)}
    metadata['initialization']=initialization
    write(out/'metadata.json',metadata)
    history=[];best=-float('inf');started=time.time()
    write(out/'status.json',{'state':'running','epoch':0,'pid':os.getpid()})
    for epoch in range(1,cfg['epochs']+1):
        model.train();indices,valid=groups(payload,'train',cfg['seed']+epoch)
        # A smoke remains clearly separate and does not yield formal metrics.
        batches=range(0,len(indices),cfg['batch_fields'])
        if args.smoke_steps:batches=list(batches)[:args.smoke_steps]
        losses=[]
        for step,start in enumerate(batches):
            item=batch(payload,indices[start:start+cfg['batch_fields']],valid[start:start+cfg['batch_fields']],args.device)
            optimizer.zero_grad(set_to_none=True)
            result=model(item['vision']);keep=item['valid']
            y=((item['target']-model.target_mean)/model.target_std)[keep]
            pred=result['normalized_prediction'][keep]
            loss=F.smooth_l1_loss(pred,y)+cfg['correlation_weight']*(1-correlation(pred,y))
            pair=torch.sign(y[:,None]-y[None,:]);mask=pair!=0
            if bool(mask.any()):loss=loss+cfg['ranking_weight']*F.relu(.1-pair*(pred[:,None]-pred[None,:]))[mask].mean()
            reg,_=router_regularization(result['routing'],keep)
            loss=loss+cfg['router_balance_weight']*reg['balance']+cfg['router_importance_weight']*reg['importance']
            loss=loss+cfg['router_diversity_weight']*reg['diversity']+cfg['guard_weight']*result['guard_energy']
            if cfg['slot_consistency_weight'] and step%cfg['slot_consistency_interval']==0:
                # Two deterministic forwards: penalize physical-slot sensitivity,
                # not different random phase/shift/noise augmentation draws.
                model.eval()
                base=model(item['vision'])['normalized_prediction']
                shifted=model(item['vision'].roll(1,1))['normalized_prediction'].roll(-1,1)
                loss=loss+cfg['slot_consistency_weight']*F.mse_loss(base[keep],shifted[keep])
                model.train()
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();losses.append(float(loss.detach()))
        scheduler.step()
        row={'epoch':epoch,'loss':float(np.mean(losses)),'elapsed_seconds':time.time()-started}
        if not args.smoke_steps and (epoch==1 or epoch%cfg['eval_interval']==0 or epoch==cfg['epochs']):
            metrics=evaluate(model,payload,args.device,cfg['eval_batch_fields'])
            row['test_development']=metrics
            if metrics['srcc']>best:
                best=metrics['srcc']
                ck=model.checkpoint(**metadata,epoch=epoch,metrics=metrics)
                save_checkpoint(out/'best_checkpoint.pt',ck)
                evaluate(model,payload,args.device,cfg['eval_batch_fields'],out,epoch)
        history.append(row);write(out/'history.json',history)
        last=model.checkpoint(**metadata,epoch=epoch)
        last['optimizer']=optimizer.state_dict();last['scheduler']=scheduler.state_dict()
        save_checkpoint(out/'last_checkpoint.pt',last)
        write(out/'status.json',{'state':'running','epoch':epoch,'pid':os.getpid(),
                               'best_srcc':best if np.isfinite(best) else None,'elapsed_seconds':time.time()-started})
        print(json.dumps(row),flush=True)
        if args.smoke_steps:break
    if args.smoke_steps:
        write(out/'status.json',{'state':'smoke_passed','steps':args.smoke_steps,'formal_metrics':False})
        return
    selected,_=load_lightgen(out/'best_checkpoint.pt',args.device)
    final=evaluate(selected,payload,args.device,cfg['eval_batch_fields'],out)
    final['reference_srcc']=cfg['reference_srcc'];final['absolute_srcc_difference']=abs(final['srcc']-cfg['reference_srcc'])
    final['within_target']=final['absolute_srcc_difference']<=cfg['allowed_absolute_srcc_difference']
    cycles=[evaluate(selected,payload,args.device,cfg['eval_batch_fields'],slot_shift=k)['srcc'] for k in range(16)]
    write(out/'slot_cycle_audit.json',{'srcc_by_shift':cycles,'mean_srcc':float(np.mean(cycles)),
                                    'minimum_srcc':float(np.min(cycles))})
    release=selected.checkpoint(**metadata,metrics=final,selection='best_original_TEST_development_SRCC')
    save_checkpoint(out/'LightGen.pt',release)
    write(out/'metrics.json',final)
    write(out/'status.json',{'state':'completed','epochs':cfg['epochs'],'elapsed_seconds':time.time()-started,**final})
    print('COMPLETED',json.dumps(final),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/trial_relu_s163.json')
    p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--initialize',type=Path,help='Weights-only continuation of the same surrogate; never the paper model')
    p.add_argument('--paper-initialize',type=Path,help='Explicitly whitelist matching front, merger, global tensors for initialization')
    p.add_argument('--smoke-steps',type=int,default=0)
    p.add_argument('--allow-shared-gpu',action='store_true',help='Only for explicitly authorized nonexclusive correctness smoke')
    args=p.parse_args()
    try:train(args)
    except BaseException as error:
        if args.out.is_dir():write(args.out/'failure.json',{'type':type(error).__name__,'error':str(error),'time':time.time()})
        raise


if __name__=='__main__':main()
