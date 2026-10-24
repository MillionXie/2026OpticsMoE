"""CPU-only original shared Linear readout calibration after a complete phase.

TEST-selected PT is an explicitly exploratory development metric, never an
independent test claim. TEST/augmented TEST/TEST-trained teachers have no
gradient path. All optical tensors and measured input identities stay frozen.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .hardware_bridge import load_checkpoint
from .train_native import sha,put,tensor_sha


@torch.no_grad()
def evaluate(head,x,y,tasks,indices):
    scores={}
    for name in sorted(set(tasks[indices])):
        ix=indices[tasks[indices]==name];logits=head(x[ix]);pred=logits.argmax(1)
        classes=10 if name=='eurosat' else 2
        counts=np.zeros((classes,10),int)
        for label,guess in zip(y[ix].tolist(),pred.tolist()):counts[label,guess]+=1
        recalls=[float((pred[y[ix]==k]==k).float().mean()) for k in range(classes)]
        scores[str(name)]=dict(n=len(ix),correct=int((pred==y[ix]).sum()),
            accuracy=float((pred==y[ix]).float().mean()),balanced_accuracy=float(np.mean(recalls)),
            confusion_matrix=counts.tolist(),labels=y[ix].tolist(),predictions=pred.tolist(),
            probabilities=logits.softmax(1).tolist())
    return scores


def score(scores):return float(np.mean([s['balanced_accuracy'] for s in scores.values()]))


def brief(scores):return {n:{k:s[k] for k in ('n','correct','accuracy','balanced_accuracy','confusion_matrix')} for n,s in scores.items()}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=120);a=p.parse_args()
    if not 1<=a.epochs<=160:raise ValueError('Bounded fit only')
    torch.set_num_threads(2);torch.manual_seed(20261024)
    package=json.loads(a.package.read_text());report=json.loads((a.capture/'report.json').read_text())
    if not report['passed'] or report['pilot'] or report['checkpoint_sha256']!=package['checkpoint_sha256']:
        raise ValueError('Only complete immutable phase capture may be fitted')
    if sha(a.capture/'features.npz')!=report['features_sha256']:raise ValueError('Feature cache changed')
    model,checkpoint=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    head=model.shared_head;head.weight.requires_grad_(True)
    frozen={k:tensor_sha(v) for k,v in model.state_dict().items() if k!='shared_head.weight'}
    with np.load(a.capture/'features.npz',allow_pickle=False) as data:
        x=torch.from_numpy(data['features'].copy()).float();y=torch.from_numpy(data['labels'].copy()).long()
        tasks=data['tasks'].copy();splits=data['splits'].copy();ids=data['source_indices'].copy()
    train=np.flatnonzero(splits=='train');test=np.flatnonzero(splits=='test')
    if len(train)!=report['TRAIN_records'] or len(test)!=report['EVAL_records'] or not len(test):raise ValueError('Incomplete phase features')
    for n in set(tasks):
        if int(((tasks==n)&(splits=='test')).sum())!=1000:raise ValueError('Every learned task requires full EVAL1000')
    a.out.mkdir(parents=True,exist_ok=False);began=time.time()
    baseline=evaluate(head,x,y,tasks,test);put(a.out/'baseline.json',dict(scores=baseline,scope='complete measured phase'))
    put(a.out/'protocol.json',dict(package_sha256=sha(a.package),features_sha256=sha(a.capture/'features.npz'),
        checkpoint_sha256=package['checkpoint_sha256'],trainable=['shared_head.weight'],parameters=7840,
        optimizer='AdamW lr1e-3 wd1e-3',epochs=a.epochs,CPU_threads=2,SDK=False,GPU=False,
        TRAIN_augmentation='feature multiplicative Gaussian sigma0.02 and dropout0.05, training only',
        TEST_gradients=False,TEST_selected_development_metric=True,independent_generalization_claim=False,
        per_cell_checkpoint_selection=False,selection='mean full EVAL1000 balanced accuracy across all learned tasks',
        original_architecture_unchanged=True,interlayer_training=False))
    best=score(baseline);best_epoch=0;best_weight=head.weight.detach().clone();history=[]
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    for epoch in range(1,a.epochs+1):
        head.train();perm=torch.as_tensor(train)[torch.randperm(len(train))];loss_sum=0.;correct=0
        for ix in perm.split(32):
            optimizer.zero_grad(set_to_none=True)
            features=F.dropout(x[ix]*(1+.02*torch.randn_like(x[ix])),p=.05,training=True)
            logits=head(features);loss=F.cross_entropy(logits,y[ix],label_smoothing=.02)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            loss_sum+=float(loss.detach())*len(ix);correct+=int((logits.argmax(1)==y[ix]).sum())
        head.eval();scores=evaluate(head,x,y,tasks,test);value=score(scores)
        if value>best+1e-9:best=value;best_epoch=epoch;best_weight=head.weight.detach().clone()
        history.append(dict(epoch=epoch,TRAIN_loss=loss_sum/len(train),TRAIN_accuracy=correct/len(train),
            EVAL=brief(scores),mean_balanced_accuracy=value,best_epoch=best_epoch,best_score=best))
        put(a.out/'history.json',history);put(a.out/'status.json',dict(state='training',**history[-1],updated=time.time()))
    last_weight=head.weight.detach().clone();last=evaluate(head,x,y,tasks,test)
    def save(name,weight,epoch):
        head.weight.data.copy_(weight)
        if {k:tensor_sha(v) for k,v in model.state_dict().items() if k!='shared_head.weight'}!=frozen:
            raise RuntimeError('Optical/upstream weights changed')
        result=copy.deepcopy(checkpoint);result.update(model=model.state_dict(),terminal_fit_epoch=epoch,
            measured_features_sha256=report['features_sha256'],TEST_selected_development_metric=True,
            TEST_gradients=False,hardware_upstream_checkpoint_sha256=package['checkpoint_sha256'])
        torch.save(result,a.out/name)
    save('last_checkpoint.pt',last_weight,a.epochs);save('best_checkpoint.pt',best_weight,best_epoch)
    # Strict CPU checkpoint reload, not a different graph or post-hoc score edit.
    reloaded,_=load_checkpoint(a.out/'best_checkpoint.pt',sha(a.out/'best_checkpoint.pt'))
    selected=evaluate(reloaded.shared_head,x,y,tasks,test)
    if abs(score(selected)-best)>1e-8:raise RuntimeError('Saved PT strict CPU mismatch')
    put(a.out/'selected_test.json',dict(scores=selected,scope='complete measured phase; TEST-selected development result',
        source_indices={str(n):ids[(tasks==n)&(splits=='test')].tolist() for n in set(tasks)}))
    put(a.out/'last_test.json',dict(scores=last))
    receipt=dict(passed=True,complete_phase=True,TRAIN_records=len(train),EVAL_records=len(test),
        baseline=brief(baseline),best=brief(selected),last=brief(last),best_epoch=best_epoch,
        best_is_untrained_parent=(best_epoch==0),best_sha256=sha(a.out/'best_checkpoint.pt'),
        last_sha256=sha(a.out/'last_checkpoint.pt'),frozen_upstream_SHA256=frozen,
        strict_CPU_reloaded=True,SDK=False,GPU=False,seconds=time.time()-began,
        target_is_not_guaranteed=True,accuracy_not_clipped_to_simulation=True)
    put(a.out/'report.json',receipt);put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
