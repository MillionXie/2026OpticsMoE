"""Finite TRAIN-constrained last-phase refinement, followed by one locked development TEST."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import boundary_refine as base
from .phase_repair_probe import select_groups, rgb_views, grouped_metrics, sha_tensor
from .expert_tail_schedule import save,sha


def project_direction(direction, constraints, passes=3):
    """Bounded cyclic removal of first-order harmful components, not an exact QP."""
    import torch
    d=direction.clone()
    for _ in range(passes):
        for g in constraints:
            dot=(d*g).sum();norm2=g.square().sum()
            if float(dot)<0 and float(norm2)>0:d.add_(g,alpha=-float(dot/norm2))
    rms=d.square().mean().sqrt()
    if not torch.isfinite(rms) or float(rms)<1e-12:raise ValueError('Projection leaves no finite direction')
    return d/rms


def acceptable(current, candidate, baseline):
    a=candidate['repair_clean'];c=candidate['complex_clean'];r=candidate['replay_clean']
    total=sum(candidate[n]['correct'] for n in ['repair_clean','complex_clean','replay_clean'])
    previous=sum(current[n]['correct'] for n in ['repair_clean','complex_clean','replay_clean'])
    nll=sum(candidate[n]['nll'] for n in ['repair_clean','complex_clean','replay_clean'])
    old_nll=sum(current[n]['nll'] for n in ['repair_clean','complex_clean','replay_clean'])
    return (a['corrected']>=2 and c['new_wrong']==0 and candidate['complex_d4']['new_wrong']==0 and r['new_wrong']<=1
        and total>sum(baseline[n]['correct'] for n in ['repair_clean','complex_clean','replay_clean'])
        and candidate['repair_d4']['mean_log_margin']>current['repair_d4']['mean_log_margin']+1e-5
        and (total>previous or (total==previous and nll<old_nll-1e-4)))


def run(c,out,profile):
    import numpy as np
    import torch
    from .train import data,evaluate
    assert c['max_updates']==8 and c['trial_rms']==[.0015,.00075,.000375] and c['radius']==.02
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'
    commit=os.environ['T10_PROJECTED_SOURCE_COMMIT'];root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    paths=[Path(__file__),profile,Path(c['boundary_profile']),*[Path(__file__).with_name(n) for n in ['phase_repair_probe.py','boundary_refine.py','model.py','train.py','plan.py','diagnose_topk.py','expert_tail_schedule.py']],Path(__file__).parent/'configs/study.json',root/'LightGenV2/common/optical_contract.py'];hashes={}
    for p in paths:
        n=p.resolve().relative_to(root).as_posix();assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+n]);hashes[n]=sha(p)
    out.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4);torch.manual_seed(17)
    cfg=json.loads(Path(c['boundary_profile']).read_text());model,m,epoch=base.load_model(cfg);model.eval()
    for n,p in model.named_parameters():p.requires_grad_(n=='global_phases.1')
    phase=model.global_phases[-1];anchor=phase.detach().clone();frozen={n:sha_tensor(p) for n,p in model.named_parameters() if n!='global_phases.1'}
    assert sha(Path(m['arguments']['data']))==m['data_sha256'];splits,_=data(Path(m['arguments']['data']));train=splits['train']
    assert sha(Path(c['reference_train_scores']))==c['reference_train_scores_sha256']
    with np.load(c['reference_train_scores'],allow_pickle=False) as z:
        assert np.array_equal(z['ids'],train[2]) and np.array_equal(z['labels'],train[1].numpy());groups=select_groups(z['labels'],z['scores'])
    assert len(groups['repair'])>0
    ix=np.array(sum(groups.values(),[]));rows=np.repeat(ix,2);tags=np.repeat(np.array(sum(([n]*len(v) for n,v in groups.items()),[])),2)
    codes=np.zeros(len(rows),np.int64);codes[1::2]=np.random.default_rng(17).integers(1,8,size=len(ix));y=train[1].numpy()[rows]
    with np.load(m['arguments']['data'],allow_pickle=False) as z:assert set(train[2][ix]).isdisjoint(z['test_ids'].astype(str))
    assert set(train[2][ix]).isdisjoint(splits['val'][2])
    save(out/'metadata.json',dict(pid=os.getpid(),command=sys.argv,source_commit=commit,source_files=hashes,configuration=c,source_selected_epoch=epoch,
        source_checkpoint_sha256=cfg['reference_sha256'],data_sha256=m['data_sha256'],groups={n:train[2][v].tolist() for n,v in groups.items()},
        role='TRAIN-only constrained selection followed by one locked user-authorized development TEST; no independent generalization claim'))
    fields=np.empty((len(rows),model.c,model.c),np.float32);rp=[];parent=[]
    for start in range(0,len(rows),2):
        rgb=rgb_views(train,rows[start:start+2],codes[start:start+2])
        with torch.no_grad():
            f,r=base.prefix(model,rgb);p=base.suffix(model,f,r)['probabilities']
            if start==0:assert torch.equal(p,model(rgb)['probabilities'])
        fields[start:start+2]=f.cpu().numpy();rp.append(r.cpu().numpy());parent.append(p.cpu().numpy())
        save(out/'status.json',dict(state='train_fields',pid=os.getpid(),completed=min(start+2,len(rows)),total=len(rows),timestamp=time.time()))
    rp=np.concatenate(rp);parent=np.concatenate(parent);baseline,_=grouped_metrics(parent,y,tags,parent)
    np.savez_compressed(out/'train_identity.npz',ids=train[2][rows],labels=y,indices=rows,groups=tags,view_codes=codes,parent=parent)
    def forward(order):
        return base.suffix(model,torch.from_numpy(fields[order]).cuda(),torch.from_numpy(rp[order]).cuda())
    def score(name):
        pp=[]
        with torch.no_grad():
            for start in range(0,len(rows),2):pp.append(forward(np.arange(start,min(start+2,len(rows))))['probabilities'].cpu().numpy())
        pp=np.concatenate(pp);assert np.isfinite(pp).all() and np.max(abs(pp.sum(1)-1))<1e-5
        stats,margin=grouped_metrics(pp,y,tags,parent)
        np.savez_compressed(out/(name+'_train_scores.npz'),scores=pp,margin=margin)
        return stats
    training=np.flatnonzero(tags!='observe');repair_count=int((tags[training]=='repair').sum());complex_count=int((tags[training]=='complex').sum())
    protected=np.flatnonzero((tags=='complex')&(parent.argmax(1)==y));history=[];current=baseline
    optimizer=torch.optim.AdamW([phase],lr=1e-4,weight_decay=0.)
    for step in range(1,9):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0,len(training),2):
            order=training[start:start+2];o=forward(order);p=o['probabilities'];logp=p.clamp_min(1e-12).log();q=torch.from_numpy(parent[order]).cuda();labels=torch.from_numpy(y[order]).long().cuda()
            v,kl,_=base.margin_and_protection(p,labels,q,cfg);is_repair=torch.as_tensor(tags[order]=='repair',device='cuda');is_complex=torch.as_tensor(tags[order]=='complex',device='cuda')
            logq=q.clamp_min(1e-12).log();pm=logq[:,2]-logq[:,[1,4]].max(1).values;cm=logp[:,2]-logp[:,[1,4]].max(1).values
            supervised=v[is_repair].square().sum()/repair_count
            protect=(pm[is_complex]-cm[is_complex]-.02).clamp_min(0).square().sum()/complex_count
            physics=.005*(-o['capture'].clamp_min(1e-12).log()).mean()+.02*model.regularizer()+.01*(49*o['route_probabilities'].mean(0).square().sum()-1)
            loss=supervised+protect+(kl+physics+100*((phase-anchor)/2).sin().square().mean())*len(order)/len(training)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite projected objective')
            loss.backward()
        assert phase.grad is not None and torch.isfinite(phase.grad).all();norm=float(phase.grad.norm());assert norm>0
        assert all(p.grad is None for n,p in model.named_parameters() if n!='global_phases.1')
        old=phase.detach().clone();state=copy.deepcopy(optimizer.state_dict());optimizer.step();direction=phase.detach()-old
        with torch.no_grad():phase.copy_(old)
        constraints=[]
        # Targeted TRAIN linearized margin constraints, recomputed at the current accepted phase.
        for row in protected:
            p=forward(np.array([row]))['probabilities'];logp=p.clamp_min(1e-12).log();margin=logp[0,2]-logp[0,[1,4]].max();g,=torch.autograd.grad(margin,phase);constraints.append(g.detach())
        direction=project_direction(direction,constraints);del constraints
        trials=[];accepted=False
        for attempt,amp in enumerate(c['trial_rms'],1):
            with torch.no_grad():phase.copy_(old+amp*direction)
            rms=base.project_phase(phase,anchor,.02);stats=score(f'update{step:02}_trial{attempt}')
            accepted=acceptable(current,stats,baseline);trials.append(dict(trial=attempt,amplitude=amp,circular_rms=rms,accepted=accepted,groups=stats))
            if accepted:current=stats;break
        if not accepted:
            with torch.no_grad():phase.copy_(old)
            optimizer.load_state_dict(state)
        history.append(dict(step=step,gradient_norm=norm,trials=trials,accepted=accepted,timestamp=time.time()))
        save(out/'history.json',history);save(out/'status.json',dict(state='projected_updates',pid=os.getpid(),step=step,accepted_updates=sum(x['accepted'] for x in history),timestamp=time.time()))
        if not accepted:break
    accepted_updates=sum(x['accepted'] for x in history);reference=Path(cfg['reference_run']);source_test=json.loads((reference/'test_result.json').read_text())['test']
    if accepted_updates:
        checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=accepted_updates,power_fraction=1.,source_commit=commit,data_sha256=m['data_sha256'],metadata=json.loads((out/'metadata.json').read_text()))
        torch.save(checkpoint,out/'last_checkpoint.pt');save(out/'candidate_lock.json',dict(checkpoint_sha256=sha(out/'last_checkpoint.pt'),selection='TRAIN constraints only; locked before single full development TEST',accepted_updates=accepted_updates))
        full_train,_=evaluate(model,splits['train'],4);val,_=evaluate(model,splits['val'],4)
        with np.load(m['arguments']['data'],allow_pickle=False) as z:test=(torch.from_numpy(z['test_images'].transpose(0,3,1,2).copy()),torch.from_numpy(z['test_labels'].astype(np.int64)),z['test_ids'].astype(str))
        dev,pp=evaluate(model,test,4);np.savez_compressed(out/'candidate_test_predictions.npz',ids=test[2],labels=test[1].numpy(),scores=pp.numpy(),predictions=pp.argmax(1).numpy())
        candidate_correct=round(dev['accuracy']*752);selected_new=candidate_correct>600 or (candidate_correct==600 and dev['macro_nll']<source_test['macro_nll']-1e-4)
        import shutil
        shutil.copy2(out/'last_checkpoint.pt' if selected_new else reference/'best_checkpoint.pt',out/'best_checkpoint.pt')
    else:
        import shutil
        shutil.copy2(reference/'best_checkpoint.pt',out/'best_checkpoint.pt');shutil.copy2(reference/'best_checkpoint.pt',out/'last_checkpoint.pt')
        candidate_correct=None;selected_new=False;full_train=val=dev=None
    assert all(sha_tensor(p)==frozen[n] for n,p in model.named_parameters() if n in frozen)
    result=dict(source_commit=commit,accepted_updates=accepted_updates,max_updates=8,trial_count=sum(len(x['trials']) for x in history),train_baseline=baseline,train_selected=current,
        full_train=full_train,val=val,candidate_development_test=dev,candidate_correct=candidate_correct,selected_new=selected_new,selected_correct=candidate_correct if selected_new else 600,
        pilot_target_met=selected_new and candidate_correct>=616,best_sha256=sha(out/'best_checkpoint.pt'),last_sha256=sha(out/'last_checkpoint.pt'),frozen_parameter_sha256=frozen,
        elapsed_seconds=time.time()-started,source_reused_without_test_forward=not accepted_updates)
    save(out/'result.json',result);save(out/'status.json',dict(state='complete',pid=os.getpid(),timestamp=time.time(),selected_correct=result['selected_correct']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(json.loads(a.profile.read_text()),a.out,a.profile)
    except Exception as e:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),pid=os.getpid(),timestamp=time.time()))
        raise
