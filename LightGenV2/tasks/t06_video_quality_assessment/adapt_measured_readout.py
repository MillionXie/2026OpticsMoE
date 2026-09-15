"""Offline Spatial readout adaptation. Never opens or changes lab hardware.

Extract the exact readout inputs by replaying ALL SIX measured CCDs, then train
only the existing readout. Full-data scores are resubstitution, not test scores.
The 80% run selects on its fixed 20% holdout (thus not an untouched final test).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .lab_runtime import STAGES, PINS, load_model, read, replay, sha, write


def split_indices(count, fraction, seed):
    if not 0 < fraction <= 1 or count < 3:
        raise ValueError('Need >=3 samples and 0 < fraction <= 1')
    order = np.random.default_rng(seed).permutation(count)
    n = count if fraction == 1 else int(count * fraction)
    if n < 3 or (fraction < 1 and count - n < 3):
        raise ValueError('Both partitions need at least three videos')
    return np.sort(order[:n]), np.sort(order[n:])


def metrics(pred, target):
    from scipy.stats import spearmanr, pearsonr
    p, y = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite predictions/labels')
    corr = np.std(p) > 0 and np.std(y) > 0
    return dict(count=len(y), srcc=float(spearmanr(p,y).statistic) if corr else 0.,
                plcc=float(pearsonr(p,y).statistic) if corr else 0.,
                rmse=float(np.sqrt(np.mean((p-y)**2))), mae=float(np.mean(abs(p-y))))


def frozen_digest(state):
    h = hashlib.sha256()
    for k,v in sorted(state.items()):
        if not k.startswith('readout.'):
            h.update(k.encode()); h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def replay_audit(prediction, reference, targets):
    """Portable FP32 gate: bound both per-MOS error and ranking drift.

    Windows torch2.8/cu126 and Linux torch2.6/cu124 select different cuDNN
    convolution kernels. IEEE GPU/CPU probes agree within 1e-5 MOS, whereas
    the original TF32-enabled acquisition predictions differ by ~0.01 MOS.
    This gate does not optimize scores or change any measured pixel.
    """
    current, original = metrics(prediction,targets),metrics(reference,targets)
    error=np.abs(np.asarray(prediction)-np.asarray(reference))
    limits=dict(max_mos_error=.03,max_srcc_delta=.0001,max_rmse_delta=.005)
    srcc_delta=abs(current['srcc']-original['srcc'])
    rmse_delta=abs(current['rmse']-original['rmse'])
    passed=bool(error.max()<=limits['max_mos_error'] and srcc_delta<=limits['max_srcc_delta'] and rmse_delta<=limits['max_rmse_delta'])
    return dict(passed=passed,limits=limits,max_mos_error=float(error.max()),mean_mos_error=float(error.mean()),
                srcc_delta=srcc_delta,rmse_delta=rmse_delta,current=current,original=original)


def trainable_readout_names(names,scope):
    if scope=='head':return set(names)
    if scope!='terminal':raise ValueError('Unknown training scope')
    allowed={'output.4.weight','output.4.bias','compact_output.4.weight','compact_output.4.bias'}
    if not allowed.issubset(set(names)):raise ValueError('Original terminal readout structure changed')
    return allowed


def training_order(indices, targets, strategy='random'):
    """Permutation of TRAIN identities only; stratification never duplicates samples."""
    indices=np.asarray(indices,dtype=np.int64)
    if strategy=='random':return np.random.permutation(indices)
    if strategy!='mos_stratified':raise ValueError('Unknown batch order')
    values=np.asarray(targets)[indices]
    if not len(indices) or not np.isfinite(values).all():raise ValueError('Invalid training targets')
    ordered=indices[np.argsort(values,kind='stable')]
    bins=[np.random.permutation(b) for b in np.array_split(ordered,min(10,len(ordered)))]
    # Each consecutive group covers all populated MOS quantile bins.  The
    # final partial batch is retained, so each training identity occurs once.
    return np.asarray([b[i] for i in range(max(map(len,bins)))
                       for b in [bins[j] for j in np.random.permutation(len(bins))]
                       if i<len(b)],dtype=np.int64)


def loss_weights(args):
    values=tuple(float(getattr(args,k,d)) for k,d in
                 [('reg_weight',1.),('rank_weight',.2),('corr_weight',.1)])
    if not all(np.isfinite(x) and x>=0 for x in values) or sum(values)==0:
        raise ValueError('Loss weights must be finite, nonnegative and not all zero')
    return values


def official_partitions(training, evaluation):
    """Original train/test identities, not a random split of measured test data."""
    tr,te=training['video_ids'],evaluation['video_ids']
    if training.get('dataset_split')!='train' or evaluation.get('dataset_split','test')!='test':
        raise ValueError('Require original training and test caches')
    if len(tr)!=2250 or len(te)!=558 or len(set(tr))!=2250 or len(set(te))!=558 or set(tr)&set(te):
        raise ValueError('Expected disjoint original 2250/558 video identities')
    for key in ('contract','checkpoint_sha256','target_mean','target_std'):
        if training[key]!=evaluation[key]:raise ValueError('Train/test cache contract differs: '+key)
    return np.arange(2250),np.arange(2250,2808)


def partial_test_indices(original_train, original_test, fraction, seed):
    """Explicit deployment adaptation, never relabel adapted test IDs as unseen."""
    if not np.isfinite(fraction) or not 0<=fraction<1:
        raise ValueError('Test adaptation fraction must be in [0,1)')
    original_train=np.asarray(original_train,dtype=np.int64)
    original_test=np.asarray(original_test,dtype=np.int64)
    if len(set(original_train)&set(original_test)):
        raise ValueError('Original partitions overlap')
    count=int(np.ceil(len(original_test)*fraction))
    if len(original_test)-count<3:raise ValueError('At least three held-out videos required')
    shuffled=np.random.default_rng(seed).permutation(original_test)
    adapted=np.sort(shuffled[:count]);holdout=np.sort(shuffled[count:])
    return np.sort(np.concatenate((original_train,adapted))),holdout,adapted


def extract(a):
    import torch
    from .lab_bench import verified_ccd, identity
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root, s, dest = Path(a.project), Path(a.session_dir), Path(a.output)
    if dest.exists():
        raise FileExistsError(dest)
    state, release = read(s/'session.json'), read(root/'release.json')
    if state['target'] != 'spatial' or state['measured_stages'] != list(STAGES):
        raise ValueError('Requires complete six-pass Spatial measurements')
    if state['release_sha256'] != sha(root/'release.json'):
        raise ValueError('Release identity mismatch')
    c=state['hardware_config']
    if state['hardware_sha256'] != identity(c):
        raise ValueError('Hardware identity mismatch')
    dataset_split=release.get('dataset_split','test')
    expected=2250 if dataset_split=='train' else 558
    if len(state['fields']) != expected:
        raise ValueError(f'Expected the entire {expected}-video measured {dataset_split} dataset')
    model, _ = load_model('spatial',root/'weights/best_checkpoint.pt',a.device)
    model.requires_grad_(False)
    if model.late_input_correction is not None:
        raise ValueError('Extra post-readout correction requires an explicit contract')
    captured=[]
    def hook(module, args):
        captured.append(tuple(x.detach().cpu().clone() for x in args))
    handle=model.readout.register_forward_pre_hook(hook)
    features=[[],[],[]]; targets=[]; ids=[]; predictions=[]; evidence=[]
    reference={r['video']:r['prediction'] for r in read(s/'results.json')['rows']}
    try:
        for i,item in enumerate(state['fields']):
            if sum(item['valid']) != 1 or len(item['valid']) != 1:
                raise ValueError('Spatial must have one video per field')
            source=root/item['file']
            if sha(source)!=item['sha256']:
                raise ValueError('Input cache identity mismatch')
            batch=torch.load(source,map_location='cpu',weights_only=False)
            measured={}; digests={}
            for stage in STAGES:
                p,rec=verified_ccd(s,stage,item['key'])
                # The immutable original acquisition audit and full ZIP hash
                # cover inherited-record identities; recheck every PNG here.
                digests[stage]=rec['sha256']
                measured[stage]=torch.from_numpy(np.array(Image.open(p),dtype=np.float32))[None]*float(c.get('detector_intensity_scale',{}).get(stage,1/255))
            captured.clear()
            result,tap=replay(model,'spatial',batch,measured)
            if len(captured)!=1 or tap.index!=6:
                raise ValueError('Readout/propagation boundary changed')
            for k,tensor in enumerate(captured[0]): features[k].append(tensor)
            score=float(result['prediction'].item())
            predictions.append(score); targets.append(item['targets'][0]); ids.append(item['sample_ids'][0])
            evidence.append(dict(field=item['key'],input_sha256=item['sha256'],ccd_sha256=digests))
            if (i+1)%50==0: print('EXTRACT',i+1,'/',expected,flush=True)
    finally:
        handle.remove()
    if len(set(ids))!=len(ids): raise ValueError('Duplicate video identities')
    max_error=max(abs(p-reference[v]) for p,v in zip(predictions,ids))
    audit=replay_audit(predictions,[reference[v] for v in ids],targets)
    audit['precision']='FP32, CUDA matmul TF32 off, cuDNN TF32 off'
    audit['rows']=[dict(video=v,server=p,acquisition=reference[v]) for p,v in zip(predictions,ids)]
    write(dest.with_suffix('.replay_audit.json'),audit)
    if not audit['passed']:
        raise ValueError(f'Measured replay mismatch; see replay_audit.json: {max_error}')
    payload=dict(contract='spatial_six_real_ccd_readout_inputs_v1',dataset_split=dataset_split,
                 checkpoint_sha256=PINS['spatial']['sha256'],
                 hardware_sha256=state['hardware_sha256'],release_sha256=state['release_sha256'],
                 vision=torch.cat(features[0]),language=torch.cat(features[1]),mask=torch.cat(features[2]),
                 targets=torch.tensor(targets,dtype=torch.float32),video_ids=ids,
                 original_prediction=torch.tensor(predictions),target_mean=float(model.target_mean),target_std=float(model.target_std),
                 evidence=evidence,source_session=s.name,max_replay_score_error=max_error)
    dest.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,dest)
    write(dest.with_suffix('.json'),dict(cache_sha256=sha(dest),count=len(ids),
          feature_shapes={k:list(payload[k].shape) for k in ('vision','language','mask')},
          replay_metrics=metrics(predictions,targets),max_replay_score_error=max_error,
          source_session=s.name,checkpoint_sha256=payload['checkpoint_sha256']))
    print('CACHE_READY',dest,flush=True)


def train(a):
    import torch
    import torch.nn.functional as F
    reg_weight,rank_weight,corr_weight=loss_weights(a)
    batch_order=getattr(a,'batch_order','random')
    test_adapt_fraction=float(getattr(a,'test_adapt_fraction',0.))
    if not np.isfinite(test_adapt_fraction) or not 0<=test_adapt_fraction<1:
        raise ValueError('Test adaptation fraction must be in [0,1)')
    if test_adapt_fraction and not getattr(a,'eval_cache',None):
        raise ValueError('Partial test adaptation requires both original train and test caches')
    out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=torch.load(a.cache,map_location='cpu',weights_only=False)
    if data['contract']!='spatial_six_real_ccd_readout_inputs_v1' or sha(a.checkpoint)!=data['checkpoint_sha256']:
        raise ValueError('Cache/checkpoint identity mismatch')
    eval_cache=getattr(a,'eval_cache',None)
    official=bool(eval_cache)
    if official:
        evaluation=torch.load(eval_cache,map_location='cpu',weights_only=False)
        official_train,official_test=official_partitions(data,evaluation)
        data=dict(data)
        for key in ('vision','language','mask','targets','original_prediction'):
            data[key]=torch.cat([data[key],evaluation[key]])
        data['video_ids']=data['video_ids']+evaluation['video_ids']
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model,_=load_model('spatial',a.checkpoint,'cpu');model.requires_grad_(False)
    source_frozen=frozen_digest(source['state_dict'])
    # Standalone lab packages are not Git worktrees; their pinned release is canonical.
    release_path=Path(a.checkpoint).resolve().parent.parent/'release.json'
    commit=read(release_path)['source_commit'] if release_path.exists() else subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    write(out/'launch.json',dict(arguments=vars(a),command=sys.argv,commit=commit,adaptation_source_sha256=sha(__file__),
          cache_sha256=sha(a.cache),eval_cache_sha256=sha(eval_cache) if official else None,checkpoint_sha256=sha(a.checkpoint),
          torch=torch.__version__,device=a.device,gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
          training_scope='existing readout.* only; no hardware, no new modules',
          mean=data['target_mean'],std=data['target_std'],seed=a.seed))
    targets=data['targets']; n=len(targets); all_idx=np.arange(n)
    results={}
    fractions=(1.,) if official else ((a.train_fraction,) if getattr(a,'train_fraction',None) is not None else (1.,.8))
    for fraction in fractions:
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
        adapted_idx=np.empty(0,dtype=np.int64)
        if official:train_idx,hold_idx,adapted_idx=partial_test_indices(official_train,official_test,test_adapt_fraction,a.seed)
        else:train_idx,hold_idx=split_indices(n,fraction,a.seed)
        mixed=bool(len(adapted_idx))
        tag=(f'train2250_plus_test{len(adapted_idx)}_holdout{len(hold_idx)}' if mixed else
             ('original_train2250_test558' if official else ('full100' if fraction==1 else 'split80')))
        folder=out/tag;folder.mkdir()
        select_idx=hold_idx if official else (train_idx if fraction==1 else hold_idx)
        protocol=('partial original-test deployment adaptation; select on remaining holdout; full 558 is mixed seen/unseen' if mixed else
                  ('original 2250 train only; periodic 558 test selection, not untouched test' if official else
                   ('same-data adaptation' if fraction==1 else 'fixed holdout, used for checkpoint selection')))
        write(folder/'split.json',dict(seed=a.seed,train=[data['video_ids'][i] for i in train_idx],
              holdout=[data['video_ids'][i] for i in hold_idx],selection='remaining_test_holdout_srcc' if mixed else ('original_test_srcc' if official else ('train_srcc' if fraction==1 else 'holdout_srcc')),
              original_train=[data['video_ids'][i] for i in official_train] if official else None,
              original_test=[data['video_ids'][i] for i in official_test] if official else None,
              test_adaptation=[data['video_ids'][i] for i in adapted_idx],test_adapt_fraction=test_adapt_fraction,
              untouched_test=False,test_samples_in_gradient=len(adapted_idx) if official else None,description=protocol))
        head=copy.deepcopy(model.readout).to(a.device).requires_grad_(True)
        scope=getattr(a,'scope','head')
        allowed=trainable_readout_names(dict(head.named_parameters()),scope)
        for name,param in head.named_parameters():param.requires_grad_(name in allowed)
        initial={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        anchor_strength=float(getattr(a,'anchor',0.))
        anchor={k:v.detach().clone() for k,v in head.named_parameters() if v.requires_grad}
        ema_decay=float(getattr(a,'ema',0.))
        ema=copy.deepcopy(head).requires_grad_(False).eval() if ema_decay else None
        optimizer=torch.optim.AdamW([v for v in head.parameters() if v.requires_grad],lr=a.lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs,eta_min=a.lr*.1)
        def inputs(idx):return tuple(data[k][idx].to(a.device) for k in ('vision','language','mask'))
        def evaluate(candidate=None):
            candidate=head if candidate is None else candidate
            candidate.eval();scores=[]
            with torch.no_grad():
                for start in range(0,n,a.batch_size):
                    scores.append((candidate(*inputs(all_idx[start:start+a.batch_size]))*data['target_std']+data['target_mean']).cpu())
            p=torch.cat(scores).numpy()
            values=dict(all=metrics(p,targets.numpy()),train=metrics(p[train_idx],targets.numpy()[train_idx]),
                        selection=metrics(p[select_idx],targets.numpy()[select_idx]))
            if mixed:
                for label,idx in [('original_train',official_train),('test_adaptation',adapted_idx),
                                  ('test_holdout_selection',hold_idx),('original_test_all_mixed',official_test)]:
                    values[label]=metrics(p[idx],targets.numpy()[idx])
            return p,values
        baseline,base_metrics=evaluate()
        if np.max(np.abs(baseline-data['original_prediction'].numpy()))>.005:
            raise ValueError('Cached head replay differs from full measured forward')
        write(folder/'before.json',base_metrics)
        best_metrics=base_metrics;best_epoch=0;best_state=copy.deepcopy(initial);best_pred=baseline.copy();best_variant='initial'
        history=[]
        for epoch in range(1,a.epochs+1):
            head.train();order=training_order(train_idx,targets.numpy(),batch_order);losses=[]
            for start in range(0,len(order),a.batch_size):
                idx=order[start:start+a.batch_size]
                y=(targets[idx].to(a.device)-data['target_mean'])/data['target_std']
                p=head(*inputs(idx));reg=F.smooth_l1_loss(p,y)
                delta=y[:,None]-y[None,:];valid=delta.abs()>.1
                rank=F.softplus(-torch.sign(delta)*(p[:,None]-p[None,:]))[valid].mean() if valid.any() else p.sum()*0
                pc=p-p.mean();yc=y-y.mean()
                correlation=1-(pc*yc).sum()/(pc.square().sum().sqrt()*yc.square().sum().sqrt()).clamp_min(1e-6)
                loss=reg_weight*reg+rank_weight*rank+corr_weight*correlation
                if anchor_strength:
                    loss=loss+anchor_strength*sum((v-anchor[k]).square().sum() for k,v in head.named_parameters() if v.requires_grad)
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                if ema is not None:
                    with torch.no_grad():
                        for ep,hp in zip(ema.parameters(),head.parameters()):ep.lerp_(hp,1-ema_decay)
                        for eb,hb in zip(ema.buffers(),head.buffers()):eb.copy_(hb)
            scheduler.step();pred,m=evaluate();variants={'raw':m}
            candidates=[('raw',head,pred,m)]
            if ema is not None:
                ema_pred,ema_metrics=evaluate(ema);variants['ema']=ema_metrics
                candidates.append(('ema',ema,ema_pred,ema_metrics))
            for variant,candidate,cp,cm in candidates:
                if (cm['selection']['srcc'],-cm['selection']['rmse'])>(best_metrics['selection']['srcc'],-best_metrics['selection']['rmse']):
                    best_epoch=epoch;best_metrics=cm;best_state={k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()};best_pred=cp.copy();best_variant=variant
            row=dict(epoch=epoch,loss=float(np.mean(losses)),metrics=m,variants=variants,best_epoch=best_epoch,best_variant=best_variant)
            history.append(row);write(folder/'history.json',history)
            write(out/'status.json',dict(state='training',arm=tag,epoch=epoch,epochs=a.epochs,best_epoch=best_epoch,best_metrics=best_metrics,updated=time.strftime('%Y-%m-%dT%H:%M:%S')))
            if epoch==1 or epoch%10==0: print(tag,epoch,'select SRCC',m['selection']['srcc'],'best',best_metrics['selection']['srcc'],flush=True)
        last_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        for label,weights,ep in [('best',best_state,best_epoch),('last',last_state,a.epochs)]:
            if any(not torch.equal(v,initial[k]) for k,v in weights.items() if k not in allowed):
                raise ValueError('Frozen readout tensor changed')
            merged=dict(source);merged['state_dict']=dict(source['state_dict'])
            merged['state_dict'].update({'readout.'+k:v for k,v in weights.items()})
            if frozen_digest(merged['state_dict'])!=source_frozen:raise ValueError('Frozen parameters changed')
            merged['hardware_adaptation']=dict(arm=tag,epoch=ep,variant=best_variant if label=='best' else 'raw',source_checkpoint_sha256=sha(a.checkpoint),cache_sha256=sha(a.cache),eval_cache_sha256=sha(eval_cache) if official else None,selection_uses_holdout=official or fraction<1,test_samples_in_gradient=len(adapted_idx) if official else None,test_adapt_fraction=test_adapt_fraction,protocol=protocol,split_sha256=sha(folder/'split.json'),independent_test=False,commit=commit,adaptation_source_sha256=sha(__file__))
            torch.save(merged,folder/(label+'_checkpoint.pt'))
        changed=[k for k,v in best_state.items() if not torch.equal(v,initial[k])]
        result=dict(arm=tag,best_epoch=best_epoch,best_variant=best_variant,scope=scope,trainable_names=sorted(allowed),trainable_parameters=sum(v.numel() for v in head.parameters() if v.requires_grad),before=base_metrics,after=best_metrics,changed_readout_tensors=changed,
                    frozen_parameters_unchanged=True,frozen_sha256=source_frozen,
                    best_checkpoint_sha256=sha(folder/'best_checkpoint.pt'),independent_test=False,protocol=protocol,
                    test_samples_in_gradient=len(adapted_idx) if official else None,
                    rows=[dict(video=v,target=float(targets[i]),before=float(baseline[i]),after=float(best_pred[i]),partition=(('original_train' if i in official_train else ('test_adaptation' if i in adapted_idx else 'test_holdout')) if mixed else ('train' if i in train_idx else ('test' if official else 'holdout')))) for i,v in enumerate(data['video_ids'])])
        write(folder/'results.json',result);results[tag]={k:v for k,v in result.items() if k!='rows'}
        del head,optimizer;torch.cuda.empty_cache()
    write(out/'results.json',results);write(out/'status.json',dict(state='complete',results=results))
    print('COMPLETE',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    e=sub.add_parser('extract');e.add_argument('--project',required=True);e.add_argument('--session-dir',required=True)
    t=sub.add_parser('train');t.add_argument('--cache',required=True);t.add_argument('--checkpoint',required=True)
    t.add_argument('--eval-cache',help='Original 558-video test cache; no test backpropagation unless explicit --test-adapt-fraction is positive')
    t.add_argument('--test-adapt-fraction',type=float,default=0.,help='Explicitly repurpose a fixed random fraction of original test for deployment adaptation; select on remaining holdout and label full-test results as mixed')
    t.add_argument('--epochs',type=int,default=100);t.add_argument('--lr',type=float,default=1e-4)
    t.add_argument('--batch-size',type=int,default=32);t.add_argument('--seed',type=int,default=20260914)
    t.add_argument('--train-fraction',type=float,choices=[.8,1.],default=None)
    t.add_argument('--ema',type=float,default=0.,help='EMA parameter decay per optimizer step; zero disables')
    t.add_argument('--anchor',type=float,default=0.,help='L2-SP coefficient on sum squared deviation from original readout')
    t.add_argument('--scope',choices=['head','terminal'],default='head')
    t.add_argument('--reg-weight',type=float,default=1.,help='SmoothL1 regression loss weight')
    t.add_argument('--rank-weight',type=float,default=.2,help='Pairwise ranking loss weight')
    t.add_argument('--corr-weight',type=float,default=.1,help='Batch Pearson correlation loss weight')
    t.add_argument('--batch-order',choices=['random','mos_stratified'],default='random',help='Training-only ordering; each identity appears exactly once per epoch')
    for p in (e,t):p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    q=sub.add_parser('queue',help='Wait for the verified measurement archive, extract, and train both arms')
    q.add_argument('--archive',required=True);q.add_argument('--archive-sha256',required=True)
    q.add_argument('--archive-bytes',required=True,type=int);q.add_argument('--project',required=True)
    q.add_argument('--output',required=True);q.add_argument('--device',default='cuda')
    q.add_argument('--epochs',type=int,default=100);q.add_argument('--lr',type=float,default=1e-4)
    q.add_argument('--batch-size',type=int,default=32);q.add_argument('--seed',type=int,default=20260914)
    q.add_argument('--resume-verified',action='store_true',help='Reuse extracted data only after rechecking archive and every file SHA; no training overwrite')
    o=sub.add_parser('official_queue',help='Wait for six-stage training acquisition, extract and adapt original 2250 train only')
    for name in ('project','session-dir','output','eval-cache'):o.add_argument('--'+name,required=True)
    o.add_argument('--device',default='cuda');o.add_argument('--epochs',type=int,default=100)
    a=parser.parse_args()
    if a.action=='train' and (not 0<=a.ema<1 or a.anchor<0):parser.error('Require 0 <= EMA < 1 and anchor >= 0')
    globals()[a.action](a)


def official_queue(a):
    from types import SimpleNamespace
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    root,s=Path(a.project),Path(a.session_dir)
    def status(state,**extra):write(out/'queue_status.json',dict(state=state,updated=time.strftime('%Y-%m-%dT%H:%M:%S'),**extra))
    try:
        release=read(root/'release.json')
        if release.get('dataset_split')!='train' or release.get('train_videos_in_package')!=2250:raise ValueError('Not the original training handoff')
        # Verify immutable test cache before waiting; never download substituted labels.
        assets=read(root/'SHA256.json');relative=Path(a.eval_cache).resolve().relative_to(root.resolve()).as_posix()
        if sha(a.eval_cache)!=assets[relative]:raise ValueError('Test cache manifest mismatch')
        status('waiting_for_all_six_training_stages')
        deadline=time.monotonic()+12*3600
        while not (s/'results.json').exists():
            if (out/'STOP').exists():raise RuntimeError('STOP requested')
            if time.monotonic()>deadline:raise TimeoutError('Acquisition did not finish in 12 hours')
            time.sleep(10)
        results=read(s/'results.json')
        if results.get('dataset_split')!='train' or results.get('count')!=2250:raise ValueError('Wrong acquisition results')
        # Final evaluator may finish just before the supervisor's last SHA audit.
        while not all((s/'audits'/(stage+'.json')).exists() for stage in STAGES):
            if time.monotonic()>deadline:raise TimeoutError('Stage audits missing')
            time.sleep(5)
        for stage in STAGES:
            audit=read(s/'audits'/(stage+'.json'))
            if audit['status']!='passed' or audit['count']!=2250:raise ValueError('Incomplete stage audit')
        status('extracting_measured_training_readout_features')
        cache=out/'train_readout_cache.pt'
        extract(SimpleNamespace(project=str(root),session_dir=str(s),output=str(cache),device=a.device))
        status('finetuning_original_train2250',epochs=a.epochs,test_videos=558,test_samples_in_gradient=0)
        train(SimpleNamespace(cache=str(cache),eval_cache=a.eval_cache,checkpoint=str(root/'weights/best_checkpoint.pt'),
            output=str(out/'adaptation'),device=a.device,epochs=a.epochs,lr=1e-5,batch_size=64,seed=20260914,
            train_fraction=None,scope='head',anchor=.1,ema=.98))
        status('complete',results=str(out/'adaptation/results.json'))
    except BaseException as e:
        status('failed',error=repr(e));raise


def queue(a):
    import zipfile
    from types import SimpleNamespace
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    archive=Path(a.archive);started=time.monotonic()
    try:
        while not archive.exists() or archive.stat().st_size<a.archive_bytes:
            if time.monotonic()-started>7200: raise TimeoutError('Measurement archive transfer timeout')
            write(out/'queue_status.json',dict(state='waiting_for_data',bytes=archive.stat().st_size if archive.exists() else 0,total=a.archive_bytes))
            time.sleep(10)
        if archive.stat().st_size!=a.archive_bytes or sha(archive)!=a.archive_sha256:
            raise ValueError('Transferred archive does not match the acquisition SHA256')
        dest=out/'verified_measurement'
        if dest.exists():
            if not a.resume_verified:raise FileExistsError(dest)
        else:
            dest.mkdir()
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    if not (dest/name).resolve().is_relative_to(dest.resolve()):
                        raise ValueError('Unsafe archive member')
                z.extractall(dest)
        catalog=read(dest/'measurement_SHA256.json')
        for name,digest in catalog.items():
            if not (dest/name).resolve().is_relative_to(dest.resolve()) or sha(dest/name)!=digest:
                raise ValueError('Measurement manifest mismatch: '+name)
        write(out/'queue_status.json',dict(state='extracting_readout_features',archive_sha256=a.archive_sha256))
        extract(SimpleNamespace(project=a.project,session_dir=str(dest),output=str(out/'measured_readout_cache.pt'),device=a.device))
        write(out/'queue_status.json',dict(state='training_both_arms'))
        train(SimpleNamespace(cache=str(out/'measured_readout_cache.pt'),checkpoint=str(Path(a.project)/'weights/best_checkpoint.pt'),
              output=str(out/'adaptation'),device=a.device,epochs=a.epochs,lr=a.lr,batch_size=a.batch_size,seed=a.seed))
        write(out/'queue_status.json',dict(state='complete',results=str(out/'adaptation/results.json')))
    except Exception as e:
        write(out/'queue_status.json',dict(state='failed',error=repr(e)))
        raise


if __name__=='__main__':main()
