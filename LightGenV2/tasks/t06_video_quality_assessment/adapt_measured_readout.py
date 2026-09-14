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


def extract(a):
    import torch
    from .lab_bench import verified_ccd, identity
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
    if len(state['fields']) != 558:
        raise ValueError('Expected the entire 558-video measured dataset')
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
            if (i+1)%50==0: print('EXTRACT',i+1,'/558',flush=True)
    finally:
        handle.remove()
    if len(set(ids))!=len(ids): raise ValueError('Duplicate video identities')
    max_error=max(abs(p-reference[v]) for p,v in zip(predictions,ids))
    if max_error>.003:
        raise ValueError(f'Measured replay mismatch: {max_error}')
    payload=dict(contract='spatial_six_real_ccd_readout_inputs_v1',
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
    out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(a.cache,map_location='cpu',weights_only=False)
    if data['contract']!='spatial_six_real_ccd_readout_inputs_v1' or sha(a.checkpoint)!=data['checkpoint_sha256']:
        raise ValueError('Cache/checkpoint identity mismatch')
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model,_=load_model('spatial',a.checkpoint,'cpu');model.requires_grad_(False)
    source_frozen=frozen_digest(source['state_dict'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    write(out/'launch.json',dict(arguments=vars(a),command=sys.argv,commit=commit,
          cache_sha256=sha(a.cache),checkpoint_sha256=sha(a.checkpoint),
          torch=torch.__version__,device=a.device,gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
          training_scope='existing readout.* only; no hardware, no new modules',
          mean=data['target_mean'],std=data['target_std'],seed=a.seed))
    targets=data['targets']; n=len(targets); all_idx=np.arange(n)
    results={}
    for fraction in (1.,.8):
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
        tag='full100' if fraction==1 else 'split80';folder=out/tag;folder.mkdir()
        train_idx,hold_idx=split_indices(n,fraction,a.seed)
        select_idx=train_idx if fraction==1 else hold_idx
        write(folder/'split.json',dict(seed=a.seed,train=[data['video_ids'][i] for i in train_idx],
              holdout=[data['video_ids'][i] for i in hold_idx],selection='train_srcc' if fraction==1 else 'holdout_srcc',
              untouched_test=False,description='same-data adaptation' if fraction==1 else 'fixed holdout, used for checkpoint selection'))
        head=copy.deepcopy(model.readout).to(a.device).requires_grad_(True)
        initial={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs,eta_min=a.lr*.1)
        def inputs(idx):return tuple(data[k][idx].to(a.device) for k in ('vision','language','mask'))
        def evaluate():
            head.eval();scores=[]
            with torch.no_grad():
                for start in range(0,n,a.batch_size):
                    scores.append((head(*inputs(all_idx[start:start+a.batch_size]))*data['target_std']+data['target_mean']).cpu())
            p=torch.cat(scores).numpy()
            return p,dict(all=metrics(p,targets.numpy()),train=metrics(p[train_idx],targets.numpy()[train_idx]),
                          selection=metrics(p[select_idx],targets.numpy()[select_idx]))
        baseline,base_metrics=evaluate()
        if np.max(np.abs(baseline-data['original_prediction'].numpy()))>.005:
            raise ValueError('Cached head replay differs from full measured forward')
        write(folder/'before.json',base_metrics)
        best_metrics=base_metrics;best_epoch=0;best_state=copy.deepcopy(initial);best_pred=baseline.copy()
        history=[]
        for epoch in range(1,a.epochs+1):
            head.train();order=np.random.permutation(train_idx);losses=[]
            for start in range(0,len(order),a.batch_size):
                idx=order[start:start+a.batch_size]
                y=(targets[idx].to(a.device)-data['target_mean'])/data['target_std']
                p=head(*inputs(idx));reg=F.smooth_l1_loss(p,y)
                delta=y[:,None]-y[None,:];valid=delta.abs()>.1
                rank=F.softplus(-torch.sign(delta)*(p[:,None]-p[None,:]))[valid].mean() if valid.any() else p.sum()*0
                pc=p-p.mean();yc=y-y.mean()
                correlation=1-(pc*yc).sum()/(pc.square().sum().sqrt()*yc.square().sum().sqrt()).clamp_min(1e-6)
                loss=reg+.2*rank+.1*correlation
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
            scheduler.step();pred,m=evaluate()
            if (m['selection']['srcc'],-m['selection']['rmse'])>(best_metrics['selection']['srcc'],-best_metrics['selection']['rmse']):
                best_epoch=epoch;best_metrics=m;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};best_pred=pred.copy()
            row=dict(epoch=epoch,loss=float(np.mean(losses)),metrics=m,best_epoch=best_epoch)
            history.append(row);write(folder/'history.json',history)
            write(out/'status.json',dict(state='training',arm=tag,epoch=epoch,epochs=a.epochs,best_epoch=best_epoch,best_metrics=best_metrics,updated=time.strftime('%Y-%m-%dT%H:%M:%S')))
            if epoch==1 or epoch%10==0: print(tag,epoch,'select SRCC',m['selection']['srcc'],'best',best_metrics['selection']['srcc'],flush=True)
        last_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        for label,weights,ep in [('best',best_state,best_epoch),('last',last_state,a.epochs)]:
            merged=dict(source);merged['state_dict']=dict(source['state_dict'])
            merged['state_dict'].update({'readout.'+k:v for k,v in weights.items()})
            if frozen_digest(merged['state_dict'])!=source_frozen:raise ValueError('Frozen parameters changed')
            merged['hardware_adaptation']=dict(arm=tag,epoch=ep,source_checkpoint_sha256=sha(a.checkpoint),cache_sha256=sha(a.cache),selection_uses_holdout=fraction<1,independent_test=False,commit=commit)
            torch.save(merged,folder/(label+'_checkpoint.pt'))
        changed=[k for k,v in best_state.items() if not torch.equal(v,initial[k])]
        result=dict(arm=tag,best_epoch=best_epoch,before=base_metrics,after=best_metrics,changed_readout_tensors=changed,
                    frozen_parameters_unchanged=True,frozen_sha256=source_frozen,
                    best_checkpoint_sha256=sha(folder/'best_checkpoint.pt'),independent_test=False,
                    rows=[dict(video=v,target=float(targets[i]),before=float(baseline[i]),after=float(best_pred[i]),partition='train' if i in train_idx else 'holdout') for i,v in enumerate(data['video_ids'])])
        write(folder/'results.json',result);results[tag]={k:v for k,v in result.items() if k!='rows'}
        del head,optimizer;torch.cuda.empty_cache()
    write(out/'results.json',results);write(out/'status.json',dict(state='complete',results=results))
    print('COMPLETE',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    e=sub.add_parser('extract');e.add_argument('--project',required=True);e.add_argument('--session-dir',required=True)
    t=sub.add_parser('train');t.add_argument('--cache',required=True);t.add_argument('--checkpoint',required=True)
    t.add_argument('--epochs',type=int,default=100);t.add_argument('--lr',type=float,default=1e-4)
    t.add_argument('--batch-size',type=int,default=32);t.add_argument('--seed',type=int,default=20260914)
    for p in (e,t):p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    q=sub.add_parser('queue',help='Wait for the verified measurement archive, extract, and train both arms')
    q.add_argument('--archive',required=True);q.add_argument('--archive-sha256',required=True)
    q.add_argument('--archive-bytes',required=True,type=int);q.add_argument('--project',required=True)
    q.add_argument('--output',required=True);q.add_argument('--device',default='cuda')
    q.add_argument('--epochs',type=int,default=100);q.add_argument('--lr',type=float,default=1e-4)
    q.add_argument('--batch-size',type=int,default=32);q.add_argument('--seed',type=int,default=20260914)
    a=parser.parse_args();globals()[a.action](a)


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
        if dest.exists(): raise FileExistsError(dest)
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
