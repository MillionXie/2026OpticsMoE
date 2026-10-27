"""Finite TRAIN-only last-phase direction diagnosis; no VAL/TEST forwards or fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import boundary_refine as base
from .expert_tail_schedule import save, sha


def select_groups(labels, scores, seed=17):
    import numpy as np
    labels=np.asarray(labels);pred=scores.argmax(1);logs=np.log(np.maximum(scores,1e-12))
    wrong=logs.copy();wrong[np.arange(len(labels)),labels]=-np.inf
    margin=logs[np.arange(len(labels)),labels]-wrong.max(1)
    repair=[]
    for a,b in [(1,4),(4,1)]:
        ix=np.flatnonzero((labels==a)&(pred==b)&(margin>=-.5)&(margin<0))
        repair.extend(ix[np.argsort(-margin[ix],kind='stable')][:16].tolist())
    complex_ix=np.flatnonzero((labels==2)&(pred==2))
    cm=logs[complex_ix,2]-logs[complex_ix][:,[1,4]].max(1)
    protect=complex_ix[np.argsort(cm,kind='stable')][:32].tolist()
    groups={'repair':repair,'complex':protect};used=set(repair+protect)
    for name in ['replay','observe']:
        chosen=[]
        for k in range(8):
            ix=np.asarray([i for i in np.flatnonzero(labels==k) if i not in used])
            np.random.default_rng(seed+k*100003+(name=='observe')).shuffle(ix)
            chosen.extend(ix[:4].tolist())
        groups[name]=chosen;used.update(chosen)
    flat=sum(groups.values(),[])
    assert len(flat)==len(set(flat)) and len(protect)==32
    return groups


def phase_direction(gradient, adam=False):
    import torch
    direction=-gradient/(gradient.abs()+1e-8) if adam else -gradient
    rms=direction.square().mean().sqrt()
    if not torch.isfinite(rms) or float(rms)==0:raise ValueError('Zero/nonfinite repair direction')
    return direction/rms


def grouped_metrics(scores, labels, groups, source):
    import numpy as np
    pred=scores.argmax(1);old=source.argmax(1);result={}
    logs=np.log(np.maximum(scores,1e-12));other=logs.copy();other[np.arange(len(labels)),labels]=-np.inf
    margin=logs[np.arange(len(labels)),labels]-other.max(1)
    for name in ['repair','complex','replay','observe']:
        for view in [0,1]:
            ix=np.flatnonzero((groups==name)&(np.arange(len(labels))%2==view))
            if len(ix)==0:continue
            result[name+('_clean' if view==0 else '_d4')]=dict(samples=len(ix),correct=int((pred[ix]==labels[ix]).sum()),
                corrected=int(((old[ix]!=labels[ix])&(pred[ix]==labels[ix])).sum()),new_wrong=int(((old[ix]==labels[ix])&(pred[ix]!=labels[ix])).sum()),
                mean_log_margin=float(margin[ix].mean()),nll=float(-logs[ix,labels[ix]].mean()))
    return result,margin


def run(c,out,profile):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .train import data, images
    assert c['gpu_uuid']=='GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd' and os.environ['CUDA_VISIBLE_DEVICES']==c['gpu_uuid']
    assert c['amplitudes']==[.0015,.005,.01,.02] and c['seed']==17
    commit=os.environ['T10_PHASE_PROBE_SOURCE_COMMIT'];root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    paths=[Path(__file__),profile,Path(c['boundary_profile']),Path(__file__).with_name('boundary_refine.py'),*[Path(__file__).with_name(n) for n in ['model.py','train.py','plan.py','diagnose_topk.py','expert_tail_schedule.py']],Path(__file__).parent/'configs/study.json',root/'LightGenV2/common/optical_contract.py']
    hashes={}
    for p in paths:
        n=p.resolve().relative_to(root).as_posix();assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+n]);hashes[n]=sha(p)
    out.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4);torch.manual_seed(17)
    cfg=json.loads(Path(c['boundary_profile']).read_text());model,m,source_epoch=base.load_model(cfg);model.eval()
    assert sha(Path(m['arguments']['data']))==m['data_sha256']
    splits,_=data(Path(m['arguments']['data']));train=splits['train'];ids=train[2]
    scorepath=Path(c['reference_train_scores']);assert sha(scorepath)==c['reference_train_scores_sha256']
    with np.load(scorepath,allow_pickle=False) as z:
        assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],train[1].numpy());groups=select_groups(z['labels'],z['scores'])
    if not groups['repair']:raise ValueError('No eligible near-boundary TRAIN confusion; do not replace with extreme errors')
    ix=np.array(sum(groups.values(),[]),dtype=np.int64);tags=np.array(sum(([n]*len(v) for n,v in groups.items()),[]))
    assert set(ids[ix]).isdisjoint(splits['val'][2])
    # data() loads TRAIN/VAL only. TEST IDs are read solely for split identity, never images or labels in a gradient.
    with np.load(Path(m['arguments']['data']),allow_pickle=False) as z:assert set(ids[ix]).isdisjoint(z['test_ids'].astype(str))
    rows=np.repeat(ix,2);tags=np.repeat(tags,2);codes=np.zeros(len(rows),dtype=np.int64);codes[1::2]=np.random.default_rng(17).integers(1,8,size=len(ix));y=train[1].numpy()[rows]
    save(out/'metadata.json',dict(pid=os.getpid(),command=sys.argv,source_commit=commit,source_files=hashes,configuration=c,source_epoch=source_epoch,
        data_sha256=m['data_sha256'],source_checkpoint_sha256=cfg['reference_sha256'],gpu_uuid=c['gpu_uuid'],groups={n:ids[v].tolist() for n,v in groups.items()},
        role='TRAIN-only direction diagnosis, no VAL/TEST prediction or optimization; eight source-reset trials, not eight training runs'))
    for n,p in model.named_parameters():p.requires_grad_(n=='global_phases.1')
    phase=model.global_phases[-1];anchor=phase.detach().clone();frozen={n:sha_tensor(p) for n,p in model.named_parameters() if n!='global_phases.1'}
    fields=np.empty((len(rows),model.c,model.c),np.float32);rp=[];parent=[]
    for start in range(0,len(rows),2):
        rgb=images(train,torch.as_tensor(rows[start:start+2]),'cuda',0,17,False)
        for j,code in enumerate(codes[start:start+2]):
            x=rgb[j].clone();rgb[j]=torch.rot90(x.flip(-1) if code>=4 else x,int(code%4),(-2,-1))
        with torch.no_grad():
            f,r=base.prefix(model,rgb);p=base.suffix(model,f,r)['probabilities']
            if start==0:assert torch.equal(p,model(rgb)['probabilities'])
        fields[start:start+2]=f.cpu().numpy();rp.append(r.cpu().numpy());parent.append(p.cpu().numpy())
        save(out/'status.json',dict(state='building_train_fields',pid=os.getpid(),completed=min(start+2,len(rows)),total=len(rows),timestamp=time.time()))
    rp=np.concatenate(rp);parent=np.concatenate(parent)
    np.savez_compressed(out/'train_identity.npz',ids=ids[rows],indices=rows,labels=y,groups=tags,view_codes=codes,parent=parent,route_probabilities=rp)
    def batch(order):
        return (torch.from_numpy(fields[order]).cuda(),torch.from_numpy(rp[order]).cuda(),torch.from_numpy(y[order]).long().cuda(),torch.from_numpy(parent[order]).cuda())
    def losses(order):
        f,r,labels,q=batch(order);o=base.suffix(model,f,r);p=o['probabilities'];logs=p.clamp_min(1e-12).log();logq=q.clamp_min(1e-12).log()
        gap=logs.gather(1,labels[:,None]).squeeze(1)-logs.scatter(1,labels[:,None],float('-inf')).max(1).values
        repair=torch.as_tensor(tags[order]=='repair',device='cuda');protect=torch.as_tensor(tags[order]=='complex',device='cuda');zero=p.sum()*0
        repair_loss=(.2-gap[repair]).clamp_min(0).square().mean() if repair.any() else zero
        pm=logq[:,2]-logq[:,[1,4]].max(1).values;now=logs[:,2]-logs[:,[1,4]].max(1).values
        protection=(pm[protect]-now[protect]-.02).clamp_min(0).square().mean() if protect.any() else zero
        _,kl,_=base.margin_and_protection(p,labels,q,cfg)
        return dict(repair=repair_loss,complex=protection,kl=kl,capture=.005*(-o['capture'].clamp_min(1e-12).log()).mean(),
            tv=.02*model.regularizer(),anchor=100*((phase-anchor)/2).sin().square().mean())
    repair_rows=np.flatnonzero(tags=='repair');gradient=torch.zeros_like(phase)
    for start in range(0,len(repair_rows),2):
        order=repair_rows[start:start+2];loss=losses(order)['repair'];g,=torch.autograd.grad(loss,phase);gradient.add_(g,alpha=len(order)/len(repair_rows))
    assert torch.isfinite(gradient).all() and float(gradient.norm())>0
    fixed=np.array([*repair_rows[:8],*np.flatnonzero(tags=='complex')[:4],*np.flatnonzero(tags=='replay')[:4]])
    def decomposition():
        accum={n:torch.zeros_like(phase) for n in ['repair','complex','kl','capture','tv','anchor']};values={n:0. for n in accum}
        # Fixed small batch diagnostics only, not full TRAIN gradient statistics.
        for start in range(0,len(fixed),2):
            order=fixed[start:start+2];parts=losses(order)
            for n,l in parts.items():
                g,=torch.autograd.grad(l,phase,retain_graph=True);accum[n].add_(g,alpha=len(order)/len(fixed));values[n]+=float(l.detach())*len(order)/len(fixed)
        result={}
        for n,g in accum.items():
            denom=float(g.norm()*gradient.norm());result[n]=dict(loss=values[n],norm=float(g.norm()),cosine_to_full_repair_gradient=float((g*gradient).sum())/denom if denom else None)
        return result
    diagnostics={'source':decomposition(),'repair_gradient_norm':float(gradient.norm()),'fixed_diagnostic_rows':fixed.tolist()}
    # Measure the actual first Adam update, then restore the source exactly.
    optimizer=torch.optim.AdamW([phase],lr=1e-4,weight_decay=0.);phase.grad=gradient.clone();optimizer.step();actual=phase.detach()-anchor
    diagnostics['actual_first_adam']={'phase_rms':float(actual.square().mean().sqrt()),'dot_repair_gradient':float((actual*gradient).sum()),'cosine_to_negative_repair':float(-(actual*gradient).sum()/(actual.norm()*gradient.norm()))}
    with torch.no_grad():phase.copy_(anchor)
    del optimizer;phase.grad=None
    source_masks=[];count=[0];changed=[0];mask_mode=['source']
    def hook(module,args,output):
        if count[0]%2==0:
            with torch.no_grad():
                z=output[:,model.margin:model.margin+model.g,model.margin:model.margin+model.g].abs().square();z=z/z.mean((-2,-1),keepdim=True).clamp_min(1e-12)
                mask=(F.layer_norm(z,(model.g,model.g),eps=1e-6)>0).cpu().numpy()
                if mask_mode[0]=='source':source_masks.append(mask)
                else:changed[0]+=int(np.count_nonzero(mask!=source_masks[count[0]//2]))
        count[0]+=1
    handle=model.prop.register_forward_hook(hook)
    def evaluate(label):
        count[0]=0;changed[0]=0;scores=[]
        with torch.no_grad():
            for start in range(0,len(rows),2):
                f,r,_,_=batch(np.arange(start,min(start+2,len(rows))));scores.append(base.suffix(model,f,r)['probabilities'].cpu().numpy())
        scores=np.concatenate(scores);stats,margin=grouped_metrics(scores,y,tags,parent)
        np.savez_compressed(out/(label+'_predictions.npz'),ids=ids[rows],labels=y,groups=tags,view_codes=codes,scores=scores,log_margin=margin)
        return dict(groups=stats,relu_mask_changed_fraction=changed[0]/(len(rows)*model.g**2),max_probability_change=float(abs(scores-parent).max()))
    baseline=evaluate('source');assert baseline['max_probability_change']<1e-6;mask_mode[0]='trial';trials=[]
    for name,direction in [('negative_gradient',phase_direction(gradient)),('adam',phase_direction(gradient,True))]:
        for amp in c['amplitudes']:
            with torch.no_grad():phase.copy_(anchor+amp*direction)
            rms=base.project_phase(phase,anchor,.02);label=name+'_'+str(amp);stats=evaluate(label)
            trials.append(dict(direction=name,amplitude=amp,actual_circular_rms=rms,**stats))
            save(out/'progress.json',dict(pid=os.getpid(),completed=len(trials),total=8,timestamp=time.time(),latest=trials[-1]))
            with torch.no_grad():phase.copy_(anchor)
            assert torch.equal(phase,anchor)
    handle.remove()
    # Probe losses at the most repaired clean-view point, breaking ties using no TEST/VAL information.
    chosen=max(trials,key=lambda x:(x['groups']['repair_clean']['correct'],-x['groups']['complex_clean']['new_wrong'],-x['groups']['repair_clean']['nll']))
    direction=phase_direction(gradient,chosen['direction']=='adam')
    with torch.no_grad():phase.copy_(anchor+chosen['amplitude']*direction)
    base.project_phase(phase,anchor,.02);diagnostics['most_repaired_train_trial']=decomposition()
    with torch.no_grad():phase.copy_(anchor)
    assert all(sha_tensor(p)==frozen[n] for n,p in model.named_parameters() if n in frozen)
    assert torch.equal(phase,anchor)
    result=dict(baseline=baseline,trials=trials,gradient_diagnostics=diagnostics,source_restored_exactly=True,prefix_parameter_sha256=frozen,
        train_unique_images=len(ix),train_views=len(rows),trial_count=8,selected_for_diagnostic_decomposition={'direction':chosen['direction'],'amplitude':chosen['amplitude']},
        elapsed_seconds=time.time()-started,no_val_test_forward=True,source_commit=commit)
    save(out/'result.json',result);save(out/'status.json',dict(state='complete',pid=os.getpid(),timestamp=time.time()))


def sha_tensor(p):return hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(json.loads(a.profile.read_text()),a.out,a.profile)
    except Exception as e:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),pid=os.getpid(),timestamp=time.time()))
        raise
