"""Independent cross-fold, convex-KKT, folded-head and saved TEST mathematics."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_richsplit_crossbias as study
from audit_caltech20_alt246_richsplit import verify_predictions,metrics,agree
def prob(z):
    q=z-z.max(1,keepdims=True);q=np.exp(q);return q/q.sum(1,keepdims=True)
def value_grad(theta,z,y):
    q=z*theta[0]+theta[1:];q-=q.max(1,keepdims=True);logp=q-np.log(np.exp(q).sum(1,keepdims=True));p=np.exp(logp);p[np.arange(len(y)),y]-=1;p/=len(y);penalty=theta-np.r_[1.,np.zeros(20)];lam=study.config()['ridge'];value=-float(np.mean(logp[np.arange(len(y)),y]))+.5*lam*float(np.square(penalty).sum());grad=np.r_[float(np.sum(p*z)),p.sum(0)]+lam*penalty;return value,grad
def check_solver(solved,z,y):
    assert solved['success'] and len(solved['trace'])==solved['function_evaluations'];theta=np.array(solved['parameters'],np.float64);assert theta.shape==(21,) and np.isfinite(theta).all() and 1/3<=theta[0]<=2;value,grad=value_grad(theta,z,y);assert abs(value-solved['objective'])<1e-9 and np.allclose(grad,solved['gradient'],atol=1e-8,rtol=1e-6);projected=grad.copy()
    if theta[0]<=1/3+1e-7 and grad[0]>0:projected[0]=0
    if theta[0]>=2-1e-7 and grad[0]<0:projected[0]=0
    assert np.abs(projected).max()<1e-5,float(np.abs(projected).max())
    for row in solved['trace']:
        x=np.asarray(row['parameters']);assert 1/3<=x[0]<=2 and abs(value_grad(x,z,y)[0]-row['objective'])<1e-9
    return theta,float(np.abs(projected).max())
def native(state,theta):
    W=(state['linear.weight'].numpy().astype(np.float64)*theta[0]).astype(np.float32);b=(state['linear.bias'].numpy().astype(np.float64)*theta[0]+theta[1:]).astype(np.float32);return W,b
def audit(base,out,phase):
    s=study.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists() and base.name==study.RUN;parent=study.parent();meta=s.read(base/'metadata.json');assert meta['sources']==study.sources() and meta['config']==study.config() and meta['parent']==parent;study.engine.verify_sources(meta['sources'],meta['git_commit']);ex=s.read(base/(phase+'_exit.json'));p=Path('/proc',str(ex['pid']),'cmdline');assert ex['returncode']==0 and ex['actual_exit'] and (not p.exists() or str(base/phase).encode() not in p.read_bytes());original=study.temp.parent_state(parent);manifest=study.b.dataset.manifest();data=study.load('val',parent);W=original['linear.weight'].numpy().astype(np.float64);bias=original['linear.bias'].numpy().astype(np.float64);z=data['features'].astype(np.float64)@W.T+bias;checks={}
    if phase=='cv':
        result=s.read(base/'cv/result.json');assert result['sources']==meta['sources'] and result['parent']==parent and result['OOF_sha256']==s.sha(base/'cv/out_of_fold.npz') and result['new_TEST_evaluations']==0
        with np.load(base/'cv/out_of_fold.npz',allow_pickle=False) as q:arrays={k:q[k].copy() for k in q.files}
        labels=data['labels'];assignment=np.empty(len(labels),np.int64)
        for k in range(20):
            chosen=np.flatnonzero(labels==k);generator=np.random.default_rng(np.random.SeedSequence([study.config()['seed'],k]));chosen=generator.permutation(chosen)
            for i,index in enumerate(chosen):assignment[index]=i%5
        assert np.array_equal(assignment,arrays['fold']) and arrays['ids'].tolist()==manifest['split_ids']['val'] and np.array_equal(arrays['labels'],labels);assert len(result['folds'])==5
        for i,row in enumerate(result['folds']):
            fit=np.flatnonzero(assignment!=i);hold=np.flatnonzero(assignment==i);assert row['fold']==i and row['train_indices_sha256']==s.digest(fit) and row['heldout_indices_sha256']==s.digest(hold) and row['TRAIN_n']==len(fit) and row['heldout_n']==len(hold) and len(np.intersect1d(fit,hold))==0;theta,residual=check_solver(row['fit'],z[fit],labels[fit]);w,b=native(original,theta);p=prob(data['features'][hold].astype(np.float64)@w.astype(np.float64).T+b.astype(np.float64));assert np.abs(p-arrays['probabilities'][hold]).max()<=3e-6;checks['fold'+str(i)]=dict(TRAIN_n=len(fit),heldout_n=len(hold),KKT_maxabs=residual,native_probability_maxabs=float(np.abs(p-arrays['probabilities'][hold]).max()))
        baseline=prob(z);assert np.abs(baseline-arrays['baseline_probabilities']).max()<=3e-6;actual=metrics(labels,arrays['probabilities']);before=metrics(labels,arrays['baseline_probabilities']);agree(actual,result['OOF']);agree(before,result['baseline']);eligible=actual['accuracy']>before['accuracy'] and actual['nll']<=before['nll']-.001;assert eligible==result['eligible'];checks.update(eligible=eligible,OOF=actual,baseline=before)
    else:
        study.receipt(base,'cv');assert s.read(base/'cv/result.json')['eligible'];f=s.read(base/'fit/fit_result.json');assert f['best_sha256']==s.sha(f['best_path']) and f['last_sha256']==s.sha(base/'fit/last.pt');best=torch.load(f['best_path'],map_location='cpu',weights_only=False);state=best['model'];theta=np.asarray(f['parameters']);nw,nb=native(original,theta);assert set(state)==set(original) and study.temp.head.frozen_digest(state)==study.temp.head.frozen_digest(original) and np.array_equal(state['linear.weight'].numpy(),nw) and np.array_equal(state['linear.bias'].numpy(),nb) and best['sources']==study.sources() and best['config']==study.config() and best['contract']==parent['contract'] and f['actual_epochs']==f['selected_epoch']==best['epoch']==120 and f['calibration_degrees_of_freedom']==21 and f['inference_parameters_added']==f['additional_optical_epochs']==0 and f['calibration_uses_VAL_labels'] is True
        for name,value in original.items():
            if not name.startswith('linear.'):assert torch.equal(value,state[name]),name
        if phase=='fit':
            last=torch.load(base/'fit/last.pt',map_location='cpu',weights_only=False);actual_theta,residual=check_solver(last['solver'],z,data['labels']);assert np.array_equal(actual_theta,theta) and f['optimizer_iterations']==last['solver']['iterations'];checks['full_calibration_KKT_maxabs']=residual
            for role in ('train','val'):
                path=Path(f['selected_'+role+'_path']);actual,checks[role]=verify_predictions(path,state,role,manifest);agree(actual,f['metrics'][role]);agree(actual,best['metrics'][role]);raw=study.load(role,parent)
                with np.load(path,allow_pickle=False) as q:
                    for key in ('features','probe_amplitudes','labels','ids'):assert np.array_equal(q[key],raw[key])
        else:
            study.receipt(base,'fit');t=s.read(base/'test/result.json');started=s.read(base/'test/started.json');assert t['best_sha256']==started['best_sha256']==f['best_sha256'] and t['model_digest']==started['model_digest']==s.digest(state) and t['selected_epoch']==started['selected_epoch']==120 and t['test_used_for_selection'] is False and t['VAL_used_for_calibration'] is True and t['predictions_sha256']==s.sha(t['predictions_path']);actual,checks['test']=verify_predictions(Path(t['predictions_path']),state,'test',manifest);agree(actual,t['metrics'])
            if t['reused']:assert t['new_test_evaluations']==0 and t['model_digest'] in parent['test_registry'] and t['predictions_path']==parent['test_registry'][t['model_digest']]['predictions_path']
            else:assert t['new_test_evaluations']==1 and t['model_digest'] not in parent['test_registry']
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,sources=study.sources(),training_commit=meta['git_commit'],checks=checks,VAL_used_for_calibration=True,inference_parameters_added=0,real_model_forwards=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['cv','fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
