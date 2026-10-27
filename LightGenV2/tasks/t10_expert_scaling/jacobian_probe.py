"""One fixed TRAIN optical-Jacobian ridge direction; three source-reset probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import boundary_refine as base
from .boundary_transfer_probe import matched_groups,cached_prefix,cached_suffix
from .phase_repair_probe import rgb_views,sha_tensor
from .expert_tail_schedule import save,sha


def competitors(parent,labels):
    import numpy as np
    other=parent.copy();other[np.arange(len(labels)),labels]=-np.inf
    comp=other.argmax(1);comp[labels==1]=4;comp[labels==4]=1
    return comp


def solve_gram(gram,target):
    import numpy as np
    gram=np.asarray(gram,dtype=np.float64);target=np.asarray(target,dtype=np.float64);gram=(gram+gram.T)/2
    lam=.01*float(np.trace(gram))/len(target)
    if not np.isfinite(gram).all() or not np.isfinite(target).all() or not lam>0:raise ValueError('Degenerate/nonfinite Gram')
    eigen,vectors=np.linalg.eigh(gram);energy=(vectors.T@target)**2
    alpha=np.linalg.solve(gram+lam*np.eye(len(target)),target)
    return alpha,dict(ridge_lambda=lam,eigenvalues=eigen.tolist(),target_energy_below_lambda_fraction=float(energy[eigen<lam].sum()/energy.sum()) if energy.sum() else 0.,
        target_energy_by_eigenvector=energy.tolist(),weighted_linear_residual_norm=float(np.linalg.norm(gram@alpha-target)))


def stats(scores,labels,tags,parent,fixed):
    import numpy as np
    pred=scores.argmax(1);old=parent.argmax(1);logs=np.log(np.maximum(scores,1e-12));source=np.log(np.maximum(parent,1e-12))
    fm=logs[np.arange(len(labels)),labels]-logs[np.arange(len(labels)),fixed];sm=source[np.arange(len(labels)),labels]-source[np.arange(len(labels)),fixed]
    other=logs.copy();other[np.arange(len(labels)),labels]=-np.inf;global_margin=logs[np.arange(len(labels)),labels]-other.max(1)
    source_other=source.copy();source_other[np.arange(len(labels)),labels]=-np.inf
    result={}
    for group in dict.fromkeys(tags.tolist()):
        for view in [0,1]:
            ix=np.flatnonzero((tags==group)&(np.arange(len(tags))%2==view));y=labels[ix]
            result[group+('_clean' if view==0 else '_d4')]=dict(samples=len(ix),correct=int((pred[ix]==y).sum()),corrected=int(((old[ix]!=y)&(pred[ix]==y)).sum()),
                new_wrong=int(((old[ix]==y)&(pred[ix]!=y)).sum()),fixed_margin_mean=float(fm[ix].mean()),fixed_margin_change=float((fm[ix]-sm[ix]).mean()),
                global_margin_mean=float(global_margin[ix].mean()),competitor_switches=int((other[ix].argmax(1)!=source_other[ix].argmax(1)).sum()))
    return result,fm-sm


def run(c,out,profile):
    import numpy as np
    import torch
    from .train import data
    assert c['amplitudes']==[.0015,.005,.01] and c['ridge_factor']==.01 and c['target_margin']==.05
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'
    commit=os.environ['T10_JACOBIAN_SOURCE_COMMIT'];root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip());hashes={}
    paths=[Path(__file__),profile,Path(c['boundary_profile']),*[Path(__file__).with_name(n) for n in ['boundary_transfer_probe.py','phase_repair_probe.py','boundary_refine.py','model.py','train.py','plan.py','diagnose_topk.py','expert_tail_schedule.py']],Path(__file__).parent/'configs/study.json',root/'LightGenV2/common/optical_contract.py']
    for p in paths:
        n=p.resolve().relative_to(root).as_posix();assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+n]);hashes[n]=sha(p)
    out.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4);torch.manual_seed(17)
    cfg=json.loads(Path(c['boundary_profile']).read_text());model,m,epoch=base.load_model(cfg);model.eval();assert sha(Path(m['arguments']['data']))==m['data_sha256']
    splits,_=data(Path(m['arguments']['data']));train=splits['train'];assert sha(Path(c['reference_train_scores']))==c['reference_train_scores_sha256']
    with np.load(c['reference_train_scores'],allow_pickle=False) as z:
        assert np.array_equal(z['ids'],train[2]) and np.array_equal(z['labels'],train[1].numpy());groups,pairs=matched_groups(z['labels'],z['scores']);reference=z['scores'].copy()
    ix=np.array(sum(groups.values(),[]));rows=np.repeat(ix,2);tags=np.repeat(np.array(sum(([n]*len(v) for n,v in groups.items()),[])),2);y=train[1].numpy()[rows]
    with np.load(m['arguments']['data'],allow_pickle=False) as z:assert set(train[2][ix]).isdisjoint(z['test_ids'].astype(str))
    assert set(train[2][ix]).isdisjoint(splits['val'][2]);codes=np.zeros(len(rows),np.int64);codes[1::2]=np.random.default_rng(17).integers(1,8,size=len(ix))
    for n,p in model.named_parameters():p.requires_grad_(n=='global_phases.1')
    phase=model.global_phases[-1];anchor=phase.detach().clone();frozen={n:sha_tensor(p) for n,p in model.named_parameters() if n!='global_phases.1'}
    save(out/'metadata.json',dict(pid=os.getpid(),command=sys.argv,source_commit=commit,source_files=hashes,configuration=c,source_selected_epoch=epoch,
        data_sha256=m['data_sha256'],source_checkpoint_sha256=cfg['reference_sha256'],groups={n:train[2][v].tolist() for n,v in groups.items()},
        role='TRAIN optical margin Jacobian/ridge diagnosis only; observations excluded from Jacobian and target; no VAL/TEST forward or continuous optimizer'))
    fields=np.empty((len(rows),model.c,model.c),np.float32);rp=[];parent=[]
    for start in range(0,len(rows),2):
        rgb=rgb_views(train,rows[start:start+2],codes[start:start+2])
        with torch.no_grad():
            f,r=cached_prefix(model,rgb,'global_phases.1');p=cached_suffix(model,f,r,'global_phases.1')['probabilities']
            if start==0:assert torch.equal(p,model(rgb)['probabilities'])
        fields[start:start+2]=f.cpu().numpy();rp.append(r.cpu().numpy());parent.append(p.cpu().numpy())
        save(out/'status.json',dict(state='train_fields',pid=os.getpid(),completed=min(start+2,len(rows)),total=len(rows),timestamp=time.time()))
    rp=np.concatenate(rp);parent=np.concatenate(parent);assert abs(parent[::2]-reference[ix]).max()<1e-6;fixed=competitors(parent,y)
    selected=np.flatnonzero(~np.char.startswith(tags,'observe'));assert len(selected)==128
    source_logs=np.log(np.maximum(parent,1e-12));source_margins=source_logs[np.arange(len(y)),y]-source_logs[np.arange(len(y)),fixed]
    weights=np.array([1/int((tags[selected]==tags[row]).sum()) for row in selected],np.float32)
    target=np.array([max(0,.05-source_margins[row]) if tags[row].startswith('fit') else 0. for row in selected],np.float64)
    np.savez_compressed(out/'train_identity.npz',ids=train[2][rows],indices=rows,labels=y,groups=tags,view_codes=codes,parent=parent,fixed_competitors=fixed,jacobian_rows=selected)
    def forward(order):return cached_suffix(model,torch.from_numpy(fields[order]).cuda(),torch.from_numpy(rp[order]).cuda(),'global_phases.1')['probabilities']
    jac=np.empty((len(selected),phase.numel()),np.float32)
    for j,row in enumerate(selected):
        p=forward(np.array([row]));logp=p.clamp_min(1e-12).log();margin=logp[0,int(y[row])]-logp[0,int(fixed[row])]
        g,=torch.autograd.grad(margin,phase);assert torch.isfinite(g).all();jac[j]=g.detach().cpu().numpy().reshape(-1)
        if j%8==0:save(out/'status.json',dict(state='jacobian_rows',pid=os.getpid(),completed=j+1,total=len(selected),timestamp=time.time()))
    save(out/'status.json',dict(state='gram_and_ridge',pid=os.getpid(),timestamp=time.time()))
    J=torch.from_numpy(jac);A=J*torch.from_numpy(np.sqrt(weights))[:,None];gram=(A@A.T).numpy().astype(np.float64);weighted_target=np.sqrt(weights.astype(np.float64))*target
    alpha,spectral=solve_gram(gram,weighted_target);delta=torch.mv(A.T,torch.from_numpy(alpha.astype(np.float32)));prediction=torch.mv(J,delta).numpy();solution_rms=float(delta.square().mean().sqrt())
    assert solution_rms>0 and np.isfinite(prediction).all()
    # Explain class-mean components; do not delete fit means and make targets unreachable.
    means=[J[torch.from_numpy(np.flatnonzero(tags[selected]==name))].mean(0) for name in ['fit_s','fit_d']]
    b0=means[0]/means[0].norm();b1=means[1]-b0*torch.dot(b0,means[1]);basis=[b0]
    if float(b1.norm())>1e-12:basis.append(b1/b1.norm())
    B=torch.stack(basis);projected_energy=(J@B.T).square().sum(1).numpy();row_energy=J.square().sum(1).numpy()
    explanation={'raw_J_energy_in_fit_mean_span':float(projected_energy.sum()/row_energy.sum()),'solution_energy_in_fit_mean_span':float((B@delta).square().sum()/delta.square().sum()),
        'note':'Descriptive fit-mean span only; not removed from solve. Small-eigenvalue target energy is a stability diagnostic, not proof of transfer.'}
    predicted={}
    for name in dict.fromkeys(tags[selected].tolist()):
        sub=tags[selected]==name;predicted[name]=dict(target_mean=float(target[sub].mean()),predicted_mean=float(prediction[sub].mean()),predicted_min=float(prediction[sub].min()),predicted_max=float(prediction[sub].max()),predicted_rms=float(np.sqrt(np.mean(prediction[sub]**2))))
    np.savez_compressed(out/'ridge_diagnostics.npz',weighted_gram=gram,target=target,weights=weights,coefficients=alpha,source_margins=source_margins[selected],linear_solution_change=prediction)
    # Preserve large Jacobian on server only, never include it in artifact download/Git.
    save(out/'status.json',dict(state='saving_jacobian',pid=os.getpid(),timestamp=time.time()));np.save(out/'jacobian.npy',jac)
    jac_sha=sha(out/'jacobian.npy');baseline,_=stats(parent,y,tags,parent,fixed);direction=delta.reshape(phase.shape).cuda()/solution_rms;trials=[]
    del A,delta
    for amplitude in c['amplitudes']:
        with torch.no_grad():phase.copy_(anchor+amplitude*direction)
        rms=base.project_phase(phase,anchor,.02);actual_phase=torch.atan2((phase-anchor).sin(),(phase-anchor).cos()).detach().cpu().reshape(-1);linear=torch.mv(J,actual_phase).numpy();scores=[]
        with torch.no_grad():
            for start in range(0,len(rows),2):scores.append(forward(np.arange(start,min(start+2,len(rows)))).cpu().numpy())
        scores=np.concatenate(scores);assert np.isfinite(scores).all() and abs(scores.sum(1)-1).max()<1e-5
        metrics,changes=stats(scores,y,tags,parent,fixed);err=changes[selected]-linear
        np.savez_compressed(out/('ridge_'+str(amplitude)+'_scores.npz'),scores=scores,actual_margin_change=changes,predicted_jacobian_row_change=linear)
        trials.append(dict(amplitude=amplitude,circular_rms=rms,max_circular_phase_change=float(actual_phase.abs().max()),groups=metrics,linear_actual_mean_abs_error=float(abs(err).mean()),
            linear_actual_correlation=float(np.corrcoef(linear,changes[selected])[0,1]) if np.std(linear)>0 and np.std(changes[selected])>0 else None))
        save(out/'progress.json',dict(pid=os.getpid(),completed=len(trials),total=3,latest=trials[-1],timestamp=time.time()))
        with torch.no_grad():phase.copy_(anchor)
        assert torch.equal(phase,anchor)
    assert all(sha_tensor(p)==frozen[n] for n,p in model.named_parameters() if n in frozen)
    result=dict(source_commit=commit,baseline=baseline,trials=trials,spectral=spectral,class_mean_explanation=explanation,predicted_full_solution_groups=predicted,
        raw_solution_rms=solution_rms,jacobian_shape=list(jac.shape),jacobian_sha256=jac_sha,source_restored_exactly=True,frozen_parameter_sha256=frozen,
        no_val_test_forward=True,elapsed_seconds=time.time()-started)
    save(out/'result.json',result);save(out/'status.json',dict(state='complete',pid=os.getpid(),timestamp=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(json.loads(a.profile.read_text()),a.out,a.profile)
    except Exception as e:
        if a.out.exists():save(a.out/'failure.json',dict(error=repr(e),pid=os.getpid(),timestamp=time.time()))
        raise
