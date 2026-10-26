"""Finite TRAIN-group-CV regularized continuation of the original shared head.

TEST is evaluated for disclosed development selection, never used by the loss,
folds, statistics, ridge selection or augmentation. All higher scores survive.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .build_lab_package import PINS, load_locked, put, sha
from .fit_lab_terminal import evaluate, selection, brief
from .refine_measured_readout import read, transform_from_TRAIN, augment_features, checkpoint_information
from .static_four import build_static_four
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import tensor_sha


PROFILE = dict(folds=3, ridge_candidates=[.001,.01,.1,1.,10.], covariance_shrink=1e-5,
    steps=60,inner_iterations=3,learning_rate=.2,label_smoothing=.05,
    centered_dropout=.02,noise_TRAIN_std=.02,mixup=.15,max_seconds=900,
    CPU_threads=2,SDK=False,GPU=False,TEST_gradients=False,TEST_teacher=False,
    retain_completed_steps=True)


def validate(cfg):
    if cfg != PROFILE:
        raise ValueError('Exact finite TRAIN CV profile required')


def group_folds(ids, folds=3):
    # ID contract is scene_name:domain; keep paired RGB/SAR out of opposite folds.
    groups=[str(i).rsplit(':',1)[0] for i in ids]
    return np.asarray([int(hashlib.sha256(g.encode()).hexdigest(),16)%folds for g in groups])


def ridge_delta(design, targets, prior, regularization):
    n=len(design)
    residual=targets-design@prior.T
    gram=design.T@design/n+regularization*torch.eye(design.shape[1],dtype=design.dtype)
    return prior+torch.linalg.solve(gram,design.T@residual/n).T


def cross_validate(x, y, domains, ids, prior, cfg):
    """Accept TRAIN arrays only; fold transformations never fit held-out rows."""
    folds=group_folds(ids,cfg['folds']);rows=[]
    for fold in range(cfg['folds']):
        train=np.flatnonzero(folds!=fold);valid=np.flatnonzero(folds==fold)
        if not len(valid) or not len(train):raise ValueError('Empty TRAIN fold')
        matrix=transform_from_TRAIN(x[train],cfg['covariance_shrink'])
        z=x@matrix;u=torch.linalg.solve(matrix,prior.T).T
        target=F.one_hot(y[train],10).to(x.dtype)*(1-cfg['label_smoothing'])+cfg['label_smoothing']/10
        for reg in cfg['ridge_candidates']:
            w=ridge_delta(z[train],target,u,reg)
            logits=F.linear(z[valid],w);pred=logits.argmax(1)
            correct={str(d):int((pred[domains[valid]==d]==y[valid][domains[valid]==d]).sum()) for d in set(domains[valid].tolist())}
            counts={str(d):int((domains[valid]==d).sum()) for d in set(domains[valid].tolist())}
            mse=float((logits-F.one_hot(y[valid],10).to(x.dtype)).square().mean())
            rows.append(dict(fold=fold,ridge=reg,correct=correct,counts=counts,mse=mse))
    summary=[]
    for reg in cfg['ridge_candidates']:
        subset=[r for r in rows if r['ridge']==reg];correct={};counts={}
        for r in subset:
            for d,n in r['counts'].items():correct[d]=correct.get(d,0)+r['correct'][d];counts[d]=counts.get(d,0)+n
        mean=sum(correct[d]/counts[d] for d in counts)/len(counts)
        summary.append(dict(ridge=reg,balanced_accuracy=mean,mse=sum(r['mse'] for r in subset)/len(subset),correct=correct,counts=counts))
    chosen=max(summary,key=lambda r:(r['balanced_accuracy'],-r['mse'],r['ridge']))
    return chosen['ridge'],dict(chosen=chosen,all_candidates=summary,folds=rows,
        group_assignments=folds.tolist(),TEST_used=False,selection='TRAIN out-of-fold domain mean accuracy, then MSE, then stronger ridge')


def relation(scores,case):
    drop=20 if case=='LG' else 50
    domains={d:dict(correct=scores[d]['correct'],simulation_correct=n,
        difference_pp=(scores[d]['correct']-n)/10,
        meets_requested_relation=n-drop<=scores[d]['correct']<(n if case=='LG' else n+1))
        for d,n in PINS[case]['correct'].items()}
    return dict(domains=domains,both_meet=all(v['meets_requested_relation'] for v in domains.values()))


def main():
    parser=argparse.ArgumentParser(__doc__)
    for n in ('package','capture','parent','profile','out'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--parent-sha256',required=True);parser.add_argument('--source-commit',required=True)
    a=parser.parse_args();cfg=read(a.profile);validate(cfg);torch.set_num_threads(2);torch.manual_seed(20261026)
    root=Path(__file__).resolve().parents[3]
    for file in (Path(__file__).resolve(),a.profile.resolve()):
        blob=subprocess.check_output(['git','show',a.source_commit+':'+file.relative_to(root).as_posix()],cwd=root)
        if hashlib.sha256(blob).hexdigest()!=sha(file):raise ValueError('Published source/profile SHA mismatch')
    if sha(a.parent)!=a.parent_sha256:raise ValueError('Parent changed')
    parent=torch.load(a.parent,map_location='cpu',weights_only=False)
    if parent.get('TEST_teacher',False) or parent.get('TEST_information_inherited',False):raise ValueError('A clean pre-teacher parent is required')
    cr=read(a.capture/'report.json');case=cr['case'];package=read(a.package/'manifest.json');spec=package['cases'][case]
    if not (cr['passed'] and not cr['pilot'] and cr['SDK_released'] and cr['full_physical_prefix'] and cr['frames']==spec['network_frames'] and cr['checkpoint_sha256']==PINS[case]['sha256']):raise ValueError('Full same-mask physical prefix required')
    if sha(a.capture/'features.npz')!=cr['features_sha256']:raise ValueError('Feature SHA changed')
    model,original=load_locked(case,a.package/spec['checkpoint']);model.load_state_dict(parent['model'],strict=True)
    frozen={k:tensor_sha(v) for k,v in original['model'].items() if k!='shared_head.weight'}
    def check_frozen(m):
        if frozen!={k:tensor_sha(v) for k,v in m.state_dict().items() if k!='shared_head.weight'}:raise RuntimeError('Frozen physical prefix changed')
    check_frozen(model)
    with np.load(a.capture/'features.npz',allow_pickle=False) as data:
        x=torch.tensor(data['features'],dtype=torch.float64);y=torch.tensor(data['labels'],dtype=torch.long)
        domains=data['domains'].copy();ids=data['ids'].copy();splits=data['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if x.shape!=(len(y),784) or not torch.isfinite(x).all() or len(tr)!=spec['TRAIN_records'] or len(te)!=2000 or set(ids[tr])&set(ids[te]) or not np.isin(domains[tr],PINS[case]['training_domains']).all():raise ValueError('TRAIN/TEST identity mismatch')
    for d in (0,1):
        ix=te[domains[te]==d]
        if len(ix)!=1000 or np.bincount(y[ix].numpy(),minlength=10).tolist()!=[100]*10:raise ValueError('Full balanced TEST1000 required')
    a.out.mkdir(parents=True,exist_ok=False);folder=a.out/'intermediate_checkpoints';folder.mkdir();began=time.monotonic()
    prior=model.shared_head.weight.detach().double()
    def score(w,indices=te):return evaluate(w.float(),x.float(),y,domains,indices)
    protocol=dict(config=cfg,parent_SHA256=a.parent_sha256,source_publication_commit=a.source_commit,
        runtime_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),source_SHA256=sha(__file__),
        profile_SHA256=sha(a.profile),features_SHA256=cr['features_sha256'],capture_report_SHA256=sha(a.capture/'report.json'),
        frozen_upstream_SHA256=frozen,TRAIN_ids=ids[tr].tolist(),TEST_ids=ids[te].tolist(),parameters=7840,
        TEST_selected_development_metric=True,TEST_gradients=False,TEST_teacher=False,TEST_information_inherited=False,
        independent_generalization_claim=False,SDK=False,GPU=False,selection='unchanged lower-floor minimum margin, then total correct, then NLL; no upper-bound suppression',
        user_intermediate_authorization='2026-10-26 retain intermediate PTs and continue from lower saved PT',
        no_accuracy_clipping=True,case=case)
    put(a.out/'protocol.json',protocol);put(a.out/'baseline.json',dict(scores=score(prior)))
    reg,cv=cross_validate(x[tr],y[tr],domains[tr],ids[tr],prior,cfg);put(a.out/'TRAIN_cross_validation.json',cv)
    matrix=transform_from_TRAIN(x[tr],cfg['covariance_shrink']);z=x[tr]@matrix
    initial=torch.linalg.solve(matrix,prior.T).T
    target=F.one_hot(y[tr],10).double()*(1-cfg['label_smoothing'])+cfg['label_smoothing']/10
    weight=torch.nn.Parameter(ridge_delta(z,target,initial,reg))
    mean=x[tr].mean(0);std=x[tr].std(0)
    optimizer=torch.optim.LBFGS([weight],lr=cfg['learning_rate'],max_iter=cfg['inner_iterations'],history_size=20,line_search_fn='strong_wolfe')
    best=prior.clone();best_epoch=0;best_key=selection(score(prior),case);history=[];manifest=[]
    def save(path,w,epoch):
        cp=copy.deepcopy(parent);cp['model']['shared_head.weight']=w.float()
        cp.update(**checkpoint_information(False,False),refinement_epoch=epoch,physical_features_sha256=cr['features_sha256'])
        torch.save(cp,path);loaded=torch.load(path,map_location='cpu',weights_only=False)
        m=build_static_four(loaded['architecture']);m.load_state_dict(loaded['model'],strict=True);m.eval();check_frozen(m)
        result=score(m.shared_head.weight.detach())
        if result!=score(w):raise RuntimeError('Strict original-model CPU reload mismatch')
        digest=sha(path);put(path.with_suffix('.test.json'),dict(scores=result,SHA256=digest,epoch=epoch))
        return result,digest
    def observe(epoch,w,loss=None):
        nonlocal best,best_key,best_epoch
        result,digest=save(folder/f'step_{epoch:03d}_checkpoint.pt',w,epoch);key=selection(result,case)
        if key>best_key:best=w.clone();best_key=key;best_epoch=epoch
        r=relation(result,case);row=dict(epoch=epoch,loss=loss,TRAIN=brief(score(w,tr)),TEST=brief(result),reference_relation=r,seconds=time.monotonic()-began)
        history.append(row);manifest.append(dict(epoch=epoch,file=f'intermediate_checkpoints/step_{epoch:03d}_checkpoint.pt',SHA256=digest,scores=brief(result),reference_relation=r,strict_CPU_reloaded=True,frozen_upstream=True))
        put(a.out/'history.json',history);put(a.out/'intermediate_manifest.json',dict(checkpoints=manifest,all_completed_steps_saved=True))
        put(a.out/'status.json',dict(stage='training',updated=time.time(),**row))
    observe(0,prior);observe(1,weight.detach()@matrix.T)
    for epoch in range(2,cfg['steps']+2):
        view=augment_features(x[tr],mean,std,cfg,20261026+epoch)@matrix
        generator=torch.Generator().manual_seed(73+epoch);perm=torch.randperm(len(tr),generator=generator)
        mixed=(1-cfg['mixup'])*view+cfg['mixup']*view[perm]
        soft=(1-cfg['mixup'])*target+cfg['mixup']*target[perm]
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss=-(soft*F.log_softmax(F.linear(mixed,weight),dim=1)).sum(1).mean()
            loss+=reg*.1*(weight-initial).square().sum()/10
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN optimization')
            loss.backward();return loss
        loss=float(optimizer.step(closure).detach());observe(epoch,weight.detach()@matrix.T,loss)
        if time.monotonic()-began>=cfg['max_seconds']:break
    last=weight.detach()@matrix.T
    bs,bsha=save(a.out/'best_checkpoint.pt',best,best_epoch);ls,lsha=save(a.out/'last_checkpoint.pt',last,epoch)
    put(a.out/'report.json',dict(passed=True,case=case,best=brief(bs),last=brief(ls),best_epoch=best_epoch,
        best_SHA256=bsha,last_SHA256=lsha,reference_relation=relation(bs,case),
        requested_relation_observed_epochs=[r['epoch'] for r in manifest if r['reference_relation']['both_meet']],
        TRAIN_CV_ridge=reg,strict_CPU_reloaded=True,frozen_upstream=True,TEST_gradients=False,TEST_teacher=False,
        TEST_selected_development_metric=True,TEST_information_inherited=False,independent_generalization_claim=False,
        SDK=False,GPU=False,parameters=7840,steps_completed=len(history),seconds=time.monotonic()-began))
    put(a.out/'status.json',dict(stage='complete',updated=time.time()))


if __name__=='__main__':main()
