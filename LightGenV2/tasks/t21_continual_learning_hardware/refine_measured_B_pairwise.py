"""One TRAIN-selected convex shared readout with paired current-CLEVR ranking.

Same original7840 Linear, all10 logits, oldEuro300/currentCLEVR1000 only.
Pairs reconstructed by source identity, never physical capture order. Only
TRAIN chooses regularization. Full2000 EVAL compares one candidate to parent.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .hardware_bridge import load_checkpoint
from .fit_measured_stage_TRAIN import join_TRAIN
from .refine_measured_ce import contrast_transform
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief,score
from .train_native import sha,put,clevr_pairwise_loss


def pair_indices(ids,labels):
    ids=np.asarray(ids);labels=np.asarray(labels)
    if len(ids)%2 or len(np.unique(ids))!=len(ids):raise ValueError('Unique complete TRAIN pairs required')
    result=[]
    for group in sorted(set((ids//2).tolist())):
        ix=np.flatnonzero(ids//2==group)
        if len(ix)!=2 or sorted(labels[ix].tolist())!=[0,1] or set(ids[ix]%2)!={0,1}:raise ValueError('Unpaired TRAIN source')
        pos=int(ix[labels[ix]==1][0]);neg=int(ix[labels[ix]==0][0])
        if ids[pos]%2!=0 or ids[neg]!=ids[pos]+1:raise ValueError('Original positive/negative source identity changed')
        result.extend((pos,neg))
    return torch.tensor(result)


def objective(logits,y,target,tasks,pairs):
    ce=F.cross_entropy(logits,y,label_smoothing=.02,reduction='none')
    kd=F.kl_div(F.log_softmax(logits/2,1),F.softmax(target.detach()/2,1),reduction='none').sum(1)*4
    loss=sum((.2*ce[torch.tensor(tasks==n)]+4*kd[torch.tensor(tasks==n)]).mean()/2 for n in ('eurosat','clevr'))
    # Four times current-task ranking, in the same equal-task average.
    return loss+2*clevr_pairwise_loss(logits[pairs],y[pairs])


def solve(z,y,target,tasks,pairs,strength,started):
    weight=torch.zeros((10,z.shape[1]),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([weight],lr=1,max_iter=240,max_eval=480,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        if time.monotonic()-started>180:raise RuntimeError('Fixed180-second convex fit limit reached')
        optimizer.zero_grad(set_to_none=True)
        loss=objective(z@weight.T,y,target,tasks,pairs)+.5*strength*weight[:,:-1].square().sum()/10
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite paired TRAIN objective')
        loss.backward();return loss
    optimizer.step(closure);return weight.detach()


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('capture','parent','teacher','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);a=p.parse_args();torch.set_num_threads(2)
    r=json.loads((a.capture/'report.json').read_text());tm=json.loads(a.teacher.with_name('protocol.json').read_text())
    if not r['passed'] or r['pilot'] or not r['SDK_released'] or sha(a.capture/'features.npz')!=r['features_sha256']:raise ValueError('Complete immutable B required')
    if not tm['passed'] or tm['TEST_used'] or tm['TEST_augmented'] or tm['checkpoint_SHA256']!=r['checkpoint_sha256'] or sha(a.teacher)!=tm['teacher_cache_SHA256']:
        raise ValueError('Matching S1 TRAIN teacher required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp['stage']!=2 or cp.get('measured_features_sha256')!=r['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Original measured B parent only')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float64);y=torch.tensor(d['labels'],dtype=torch.long)
        tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');tt=tasks[tr]
    if x.shape!=(3300,784) or len(tr)!=1300 or len(te)!=2000 or set(tt)!={'eurosat','clevr'}:raise ValueError('B counts changed')
    with np.load(a.teacher,allow_pickle=False) as d:
        order=join_TRAIN(tt,ids[tr],d['tasks'],d['source_indices'])
        if not np.array_equal(y[tr].numpy(),d['labels'][order]):raise ValueError('Teacher labels differ')
        target=torch.tensor(d['logits'][order],dtype=torch.float64)
    ci=np.flatnonzero(tt=='clevr');ei=np.flatnonzero(tt=='eurosat')
    if len(ci)!=1000 or len(ei)!=300:raise ValueError('Old300/current1000 required')
    pairs=ci[pair_indices(ids[tr][ci],y[tr][ci].numpy()).numpy()].reshape(-1,2)
    rng=np.random.default_rng(20261025);hold=[]
    for k in range(10):
        pool=ei[y[tr][ei].numpy()==k]
        if len(pool)!=30:raise ValueError('Old balanced TRAIN30/class required')
        hold.extend(rng.permutation(pool)[:6].tolist())
    hold.extend(pairs[rng.permutation(500)[:100]].ravel().tolist());hold=np.array(sorted(hold));fit=np.setdiff1d(np.arange(1300),hold)
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();rows=[]
    for strength in (.01,.1,1.):
        transform=contrast_transform(x[tr][fit]);local=np.flatnonzero(tt[fit]=='clevr')
        pair=local[pair_indices(ids[tr][fit][local],y[tr][fit][local].numpy()).numpy()]
        w=solve(x[tr][fit]@transform.T,y[tr][fit],target[fit],tt[fit],torch.tensor(pair),strength,started)
        predictions=((x[tr][hold]@transform.T)@w.T).argmax(1);values={}
        for task in ('eurosat','clevr'):
            mask=torch.tensor(tt[hold]==task);values[task]=float((predictions[mask]==y[tr][hold][mask]).double().mean())
        rows.append(dict(strength=strength,TRAIN_holdout_accuracy=values,TRAIN_holdout_mean=float(np.mean(list(values.values())))))
    strength=max(rows,key=lambda row:(row['TRAIN_holdout_mean'],row['strength']))['strength']
    transform=contrast_transform(x[tr]);local=np.flatnonzero(tt=='clevr');pair=local[pair_indices(ids[tr][local],y[tr][local].numpy()).numpy()]
    w=solve(x[tr]@transform.T,y[tr],target,tt,torch.tensor(pair),strength,started);folded=(w@transform).float()
    put(a.out/'protocol.json',dict(parent_SHA256=a.parent_sha256,features_SHA256=r['features_sha256'],teacher_metadata=tm,
        method='Equal-task0.2 CE+4 KL(T2), plus current CLEVR pairwise4 within equal task mean; convex LBFGS240',
        TRAIN_internal_rows=rows,selected_strength=strength,TRAIN_internal_fit_source_indices=ids[tr][fit].tolist(),TRAIN_internal_holdout_source_indices=ids[tr][hold].tolist(),
        full_current_TRAIN_pairs=500,old_TRAIN300=True,TEST_gradients=False,teacher_TEST_inputs=False,original_graph=True,trainable_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),TEST_selected_development_metric=True,independent_generalization_claim=False))
    baseline=evaluate(model.shared_head,x.float(),y,tasks,te);parent=score(baseline)
    with torch.no_grad():model.shared_head.weight.copy_(folded)
    candidate=evaluate(model.shared_head,x.float(),y,tasks,te);new=score(candidate)
    def save(role):
        if upstream_state(model)!=before:raise RuntimeError('Upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,readout_refinement='TRAIN-only convex shared readout with source-ID paired current CLEVR ranking',TEST_gradients=False)
        torch.save(saved,a.out/(role+'_checkpoint.pt'))
    save('last')
    if new<=parent:model.load_state_dict(cp['model'],strict=True)
    save('best');scores={}
    for role in ('best','last'):
        loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before:raise RuntimeError('Saved upstream changed')
        value=evaluate(loaded.shared_head,x.float(),y,tasks,te);scores[role]=brief(value)
        put(a.out/(role+'_test.json'),dict(scores=value,source_indices=ids[te].tolist(),TEST_selected_development_metric=True))
    put(a.out/'report.json',dict(passed=True,best=scores['best'],last=scores['last'],baseline=brief(baseline),strict_CPU_reloaded=True,frozen_upstream=True,original_graph=True,trainable_parameters=7840,
        best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),retained_parent=new<=parent,TRAIN_holdout_rows=rows,selected_strength=strength,
        TEST_gradients=False,teacher_TEST_inputs=False,SDK=False,GPU=False,TEST_selected_development_metric=True,independent_generalization_claim=False,seconds=time.monotonic()-started))
    put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
