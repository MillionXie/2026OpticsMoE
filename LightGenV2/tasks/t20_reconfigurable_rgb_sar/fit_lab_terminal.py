"""Bounded TRAIN-only fitting of the original 7840-parameter shared readout.

One saved PT per architecture, not separate RGB/SAR selections. TEST selection
is explicitly a development metric. Feature scaling is a TRAIN-only optimizer
reparameterization folded back into the existing bias-free weight, not a new
inference layer. This entry does not train a TEST-informed teacher.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .build_lab_package import CONTRACT, PINS, load_locked, put, sha
from .static_four import build_static_four
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import tensor_sha


@torch.no_grad()
def evaluate(weight, x, y, domains, indices):
    result = {}
    for domain, name in ((0,'RGB'), (1,'SAR')):
        ix = indices[domains[indices] == domain]
        if not len(ix):
            continue
        logits = F.linear(x[ix], weight); pred = logits.argmax(1)
        confusion = np.zeros((10,10), dtype=int)
        np.add.at(confusion, (y[ix].numpy(), pred.numpy()), 1)
        result[name] = dict(n=len(ix), correct=int((pred==y[ix]).sum()),
            accuracy=float((pred==y[ix]).double().mean()), nll=float(F.cross_entropy(logits,y[ix])),
            confusion_matrix=confusion.tolist(), indices=ix.tolist(),
            labels=y[ix].tolist(), predictions=pred.tolist(), probabilities=logits.softmax(1).tolist())
    return result


def selection(scores, case):
    if any(scores[d]['n']!=1000 for d in ('RGB','SAR')):
        raise ValueError('Complete fixed TEST1000/domain required')
    lower = {d:n-PINS[case]['max_drop_correct'] for d,n in PINS[case]['correct'].items()}
    return (min(scores[d]['correct']-lower[d] for d in lower),
        sum(v['correct'] for v in scores.values()), -sum(v['nll'] for v in scores.values()))


def brief(scores):
    return {d:{k:r[k] for k in ('n','correct','accuracy','nll','confusion_matrix')} for d,r in scores.items()}


def validate(cfg):
    if cfg != json.loads(Path(__file__).with_name('config_terminal_hardware_20261026.json').read_text()):
        raise ValueError('Exact finite terminal-fit profile required')


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('package','capture','profile','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',choices=tuple(PINS),required=True)
    a=p.parse_args();cfg=json.loads(a.profile.read_text());validate(cfg)
    torch.set_num_threads(2);torch.manual_seed(20261026)
    package=json.loads((a.package/'manifest.json').read_text())
    report=json.loads((a.capture/'report.json').read_text());spec=package['cases'][a.case]
    if not (report['passed'] and not report['pilot'] and report['SDK_released'] and
            report['full_physical_prefix'] and report['case']==a.case and report['frames']==spec['network_frames'] and
            report['checkpoint_sha256']==PINS[a.case]['sha256'] and report['TEST_records']==2000):
        raise ValueError('Only the complete same-case physical capture may be fitted')
    if sha(a.capture/'features.npz')!=report['features_sha256']:
        raise ValueError('Captured feature cache changed')
    model,checkpoint=load_locked(a.case,a.package/spec['checkpoint'])
    frozen={k:tensor_sha(v) for k,v in model.state_dict().items() if k!='shared_head.weight'}
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.from_numpy(d['features'].copy()).float();y=torch.from_numpy(d['labels'].copy()).long()
        domains=d['domains'].copy();splits=d['splits'].copy();ids=d['ids'].copy()
    train=np.flatnonzero(splits=='train');test=np.flatnonzero(splits=='test')
    if (x.shape!=(len(y),784) or not torch.isfinite(x).all() or len(train)!=spec['TRAIN_records'] or
        len(test)!=2000 or set(ids[train])&set(ids[test]) or
        not np.isin(domains[train],PINS[a.case]['training_domains']).all()):
        raise ValueError('Wrong original TRAIN domain/budget or split identity')
    for d in (0,1):
        ix=test[domains[test]==d]
        if len(ix)!=1000 or np.bincount(y[ix].numpy(),minlength=10).tolist()!=[100]*10:
            raise ValueError('Incomplete per-domain class-balanced TEST')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    parent=model.shared_head.weight.detach().clone()
    scale=x[train].square().mean(0).sqrt().clamp_min(1e-3)
    z=x/scale
    baseline=evaluate(parent,x,y,domains,test);best=selection(baseline,a.case)
    best_weight=parent.clone();best_epoch=0;last_improvement=0;history=[]
    put(a.out/'baseline.json',dict(scores=baseline))
    put(a.out/'protocol.json',dict(config=cfg,profile_sha256=sha(a.profile),case=a.case,
        contract=CONTRACT,package_sha256=sha(a.package/'manifest.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha(__file__),capture_report_sha256=sha(a.capture/'report.json'),
        features_sha256=report['features_sha256'],upstream_checkpoint_sha256=PINS[a.case]['sha256'],
        TRAIN=len(train),TEST=len(test),TRAIN_domains=PINS[a.case]['training_domains'],
        TRAIN_ids=ids[train].tolist(),TEST_ids=ids[test].tolist(),frozen_upstream_sha256=frozen,
        optimizer_scale=tensor_sha(scale),optimizer_scale_folded_into_original_weight=True,
        selection='one PT; minimum per-domain margin to target, then total correct, then NLL',
        TEST_selected_development_metric=True,independent_generalization_claim=False,
        TEST_gradients=False,TEST_teacher=False,SDK=False,GPU=False))
    # TRAIN-only ridge warm start with the same bias-free linear function class.
    tx=z[train].double();target=F.one_hot(y[train],10).double()
    identity=torch.eye(784,dtype=torch.float64)
    ridge=torch.linalg.solve(tx.T@tx/len(train)+cfg['ridge']*identity,tx.T@target/len(train)).T.float()
    parameter=torch.nn.Parameter(ridge)
    optimizer=torch.optim.Adam([parameter],lr=cfg['learning_rate'])
    def observe(epoch, loss=None):
        nonlocal best,best_epoch,best_weight,last_improvement
        weight=parameter.detach()/scale
        scores=evaluate(weight,x,y,domains,test);value=selection(scores,a.case)
        if value>best:
            best=value;best_epoch=epoch;best_weight=weight.clone();last_improvement=epoch
        tr=evaluate(weight,x,y,domains,train)
        row=dict(epoch=epoch,TRAIN_loss=loss,TRAIN=brief(tr),TEST=brief(scores),best_epoch=best_epoch,
            minimum_target_margin_correct=int(best[0]),seconds=time.monotonic()-started)
        history.append(row);put(a.out/'history.json',history)
        put(a.out/'status.json',dict(state='training',**row,updated=time.time()))
    observe(-1)
    for epoch in range(1,cfg['epochs']+1):
        perm=torch.as_tensor(train)[torch.randperm(len(train))];total=0.
        for ix in perm.split(32):
            optimizer.zero_grad(set_to_none=True)
            view=F.dropout(z[ix]*(1+cfg['feature_noise_sigma']*torch.randn_like(z[ix])),
                p=cfg['feature_dropout'],training=True)
            cost=F.cross_entropy(F.linear(view,parameter),y[ix],label_smoothing=cfg['label_smoothing'])
            cost+=cfg['parent_weight_anchor']*((parameter/scale-parent).square().mean())
            if not torch.isfinite(cost):raise RuntimeError('Nonfinite TRAIN fit loss')
            cost.backward();torch.nn.utils.clip_grad_norm_([parameter],1.);optimizer.step()
            total+=float(cost.detach())*len(ix)
        observe(epoch,total/len(train))
        if time.monotonic()-started>=cfg['max_seconds'] or (epoch>=cfg['minimum_epochs'] and epoch-last_improvement>=cfg['patience']):
            break
    last_weight=parameter.detach()/scale
    results={}
    for role,weight,selected_epoch in (('last',last_weight,epoch),('best',best_weight,best_epoch)):
        with torch.no_grad():model.shared_head.weight.copy_(weight)
        if frozen!={k:tensor_sha(v) for k,v in model.state_dict().items() if k!='shared_head.weight'}:
            raise RuntimeError('Frozen optical/upstream parameters changed')
        cp=copy.deepcopy(checkpoint);cp.update(model={k:v.detach().clone() for k,v in model.state_dict().items()},
            hardware_upstream_checkpoint_sha256=PINS[a.case]['sha256'],terminal_fit_epoch=selected_epoch,
            TEST_selected_development_metric=True,TEST_gradients=False,TEST_teacher=False)
        path=a.out/(role+'_checkpoint.pt');torch.save(cp,path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        reload=build_static_four(saved['architecture']);reload.load_state_dict(saved['model'],strict=True);reload.eval()
        if frozen!={k:tensor_sha(v) for k,v in reload.state_dict().items() if k!='shared_head.weight'}:
            raise RuntimeError('Reloaded original graph frozen-state mismatch')
        scores=evaluate(reload.shared_head.weight,x,y,domains,test)
        reference=evaluate(weight,x,y,domains,test)
        if scores!=reference:raise RuntimeError('Strict CPU checkpoint replay mismatch')
        results[role]=dict(scores=scores,sha256=sha(path),epoch=selected_epoch)
        put(a.out/(role+'_test.json'),results[role])
    if selection(results['best']['scores'],a.case)!=best:raise RuntimeError('Selected PT audit mismatch')
    put(a.out/'report.json',dict(passed=True,case=a.case,baseline=brief(baseline),
        best=brief(results['best']['scores']),last=brief(results['last']['scores']),
        best_sha256=results['best']['sha256'],last_sha256=results['last']['sha256'],
        best_epoch=best_epoch,best_is_untrained_parent=(best_epoch==0),epochs_completed=epoch,
        target_met=bool(best[0]>=0),minimum_target_margin_correct=int(best[0]),
        strict_CPU_reloaded=True,frozen_upstream_sha256=frozen,SDK=False,GPU=False,
        TEST_selected_development_metric=True,TEST_gradients=False,TEST_teacher=False,
        accuracy_not_clipped_to_simulation=True,seconds=time.monotonic()-started))
    put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':
    main()
