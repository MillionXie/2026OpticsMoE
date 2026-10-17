"""Independent NumPy linear correction/calibration, no optical model or images."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_same_linear_teacher_ridge as m
from audit_caltech20_original_equal60 import metrics,agree

def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def probabilities(z):
    e=np.exp(z-z.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
def compute(state,x):return probabilities(x.astype(np.float64)@state['linear.weight'].numpy().astype(np.float64).T+state['linear.bias'].numpy().astype(np.float64))
def scale_fit(z,y,cfg):
    lo,hi=cfg['calibration_inverse_temperature_bounds'];w=len(y)/(20*np.bincount(y,minlength=20)[y])
    def g(a):return float((w*((probabilities(a*z)*z).sum(1)-z[np.arange(len(y)),y])).mean())
    if g(lo)>=0:return lo
    if g(hi)<=0:return hi
    for _ in range(60):
        x=(lo+hi)*.5
        if g(x)>0:hi=x
        else:lo=x
    return (lo+hi)*.5
def audit(key,phase):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';root=m.folder(key);parent,old_fit=m.admit(key);meta=m.s.read(root/'metadata.json');assert meta['sources']==m.sources() and meta['config']==m.config();m.engine.verify_sources(meta['sources'],meta['git_commit']);cfg=m.config();out=root/('science' if phase=='fit' else 'saved_test_audit');assert not out.exists();best=torch.load(root/'fit/selected.pt',map_location='cpu',weights_only=False);parent_pt=torch.load(parent/key/'best.pt',map_location='cpu',weights_only=False);state=parent_pt['model'];selected=best['model'];f=m.s.read(root/'fit/fit_result.json');assert f['selected_sha256']==m.s.sha(root/'fit/selected.pt') and f['candidate_sha256']==m.s.sha(root/'fit/candidate.pt') and f['parent_pt_sha256']==m.s.sha(parent/key/'best.pt')
    assert set(state)==set(selected) and selected['linear.weight'].shape==(20,256) and selected['linear.bias'].shape==(20,) and parent_pt['contract']['terminal_linear_parameters']==5140
    for name in state:
        if not name.startswith('linear.'):assert torch.equal(state[name],selected[name]),name
    assert f['parent_core_digest']==f['selected_core_digest']==m.core_digest(state)==m.core_digest(selected);assert f['new_inference_parameters']==f['new_optical_optimizer_steps']==f['TEST_arrays_read']==0 and f['analytical_head_steps']==1 and best['sources']==m.sources() and best['config']==cfg and best['epoch']==parent_pt['epoch']==f['optical_selected_epoch'];checks={}
    if phase=='fit':
        ex=m.s.read(root/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid'])).exists();a={role:arrays(parent/key/('selected_'+role+'.npz')) for role in ('train','val')};teacher=torch.load(m.old.teacher.OUT/'selected_head.pt',map_location='cpu',weights_only=False);tf=arrays(m.old.teacher.OUT/'train_features.npz');assert np.array_equal(tf['ids'],a['train']['ids']) and np.array_equal(tf['labels'],a['train']['labels'])
        x=((tf['features']-teacher['mean'])/teacher['std']).astype(np.float32).astype(np.float64);tz=x@teacher['model']['weight'].numpy().astype(np.float64).T+teacher['model']['bias'].numpy().astype(np.float64);valid=(tz.argmax(1)==tf['labels'])&(probabilities(tz).max(1)>=cfg['teacher_correct_confidence_gate']);X=np.c_[a['train']['features'].astype(np.float64),np.ones(len(tf['labels']))];old_theta=np.r_[state['linear.weight'].numpy().astype(np.float64).T,state['linear.bias'].numpy().astype(np.float64)[None]];target=tz-X@old_theta;target-=target.mean(1,keepdims=True);target*=valid[:,None];A=X.T@X;lam=cfg['ridge_relative_regularization']*np.trace(A)/A.shape[0];correction=np.linalg.solve(A+lam*np.eye(A.shape[0]),X.T@target);residual=float(np.abs((A+lam*np.eye(A.shape[0]))@correction-X.T@target).max());assert residual<1e-7
        theta=old_theta+cfg['head_weight_trust_fraction']*correction;v=np.c_[a['val']['features'].astype(np.float64),np.ones(len(a['val']['labels']))]@theta;scale=scale_fit(v,a['val']['labels'],cfg);assert abs(scale-f['calibration_inverse_temperature'])<1e-10 and abs(lam-f['ridge_lambda'])<1e-10 and f['teacher_valid_samples']==int(valid.sum());candidate=torch.load(root/'fit/candidate.pt',map_location='cpu',weights_only=False)['model'];error=max(float(np.abs(candidate['linear.weight'].numpy()-(theta[:-1].T*scale).astype(np.float32)).max()),float(np.abs(candidate['linear.bias'].numpy()-(theta[-1]*scale).astype(np.float32)).max()));assert error<2e-6,error
        cv=metrics(a['val']['labels'],compute(candidate,a['val']['features']));pv=old_fit['metrics']['val'];accepted=cv['accuracy']>pv['accuracy']+1e-12 or (abs(cv['accuracy']-pv['accuracy'])<1e-12 and cv['balanced_nll']<pv['balanced_nll']-1e-6);assert f['candidate_accepted']==accepted and m.s.digest(selected)==m.s.digest(candidate if accepted else state)
        for role,z in a.items():
            assert f['parent_feature_sha256'][role]==m.s.sha(parent/key/('selected_'+role+'.npz'));saved=arrays(root/'fit'/('selected_'+role+'.npz'));assert np.array_equal(saved['features'],z['features']) and np.array_equal(saved['labels'],z['labels']) and np.array_equal(saved['ids'],z['ids']);assert saved['ids'].tolist()==m.old.base.manifest()['split_ids'][role];err=float(np.abs(compute(selected,z['features'])-saved['probabilities']).max());assert err<3e-6;actual=metrics(saved['labels'],saved['probabilities']);agree(actual,f['metrics'][role]);agree(actual,best['metrics'][role]);checks[role]=dict(probability_maxabs=err,prediction_sha256=m.s.sha(root/'fit'/('selected_'+role+'.npz')))
        checks['ridge']=dict(normal_equation_maxabs=residual,weight_solution_maxabs=error,teacher_valid_TRAIN_samples=int(valid.sum()),VAL_scalar_calibrated=True)
    else:
        rec=m.s.read(root/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==m.s.sha(root/'science/audit.json');ex=m.s.read(root/'test_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid'])).exists();res=m.s.read(root/'once_test/result.json');saved=arrays(root/'once_test/predictions.npz');old=arrays(parent/'once_tests'/key/'predictions.npz');assert np.array_equal(saved['features'],old['features']) and np.array_equal(saved['labels'],old['labels']) and np.array_equal(saved['ids'],old['ids']);assert saved['ids'].tolist()==m.old.base.manifest()['split_ids']['test'];assert res['model_digest']==m.s.digest(selected) and res['selected_sha256']==f['selected_sha256'] and res['parent_TEST_prediction_sha256']==m.s.sha(parent/'once_tests'/key/'predictions.npz') and res['new_optical_TEST_forwards']==0 and res['test_used_for_selection'] is False
        if res['test_reused']:assert m.s.digest(state)==m.s.digest(selected) and res['production_head_TEST_evaluations']==0 and m.s.sha(root/'once_test/predictions.npz')==m.s.sha(parent/'once_tests'/key/'predictions.npz')
        else:assert res['production_head_TEST_evaluations']==1
        error=float(np.abs(compute(selected,saved['features'])-saved['probabilities']).max());assert error<3e-6;agree(metrics(saved['labels'],saved['probabilities']),res['metrics']);checks[key]=dict(probability_maxabs=error,prediction_sha256=m.s.sha(root/'once_test/predictions.npz'),independent_arithmetic_replays=1)
    out.mkdir();m.s.save(out/'audit.json',dict(m.identity(),passed=True,case=key,phase=phase,checks=checks,optical_models=0,optical_forward_calls=0,teacher_backbone_forward_calls=0,TEST_images_read=0,cached_linear_arithmetic_only=True,not_data_independence_proof=True),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.case,a.phase)
    except BaseException as e:
        out=m.folder(a.case);mark=out/(a.phase+'_audit_failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
