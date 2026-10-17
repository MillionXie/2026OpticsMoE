"""Independent NumPy saved-objective/gradient and cached prediction audit."""
import argparse,os
import numpy as np
import torch
import caltech20_same_linear_convex as m
from audit_caltech20_original_equal60 import metrics,agree

def softmax(z):
    e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)

def logprob(z):
    z=z-z.max(1,keepdims=True); return z-np.log(np.exp(z).sum(1,keepdims=True))

def independent(theta,X,y,q,valid,anchor,cfg):
    n=len(y); z=X@theta; soft=np.eye(20)[y]*(1-cfg['label_smoothing'])+cfg['label_smoothing']/20
    counts=np.bincount(y,minlength=20).astype(float); w=counts**-.5; w/=w[y].mean(); weighted=soft*w[None]
    ce=float(-(weighted*logprob(z)).sum(1).mean()); p=softmax(z); gc=p*weighted.sum(1,keepdims=True)-weighted
    T=cfg['teacher_temperature']; kd=float(((q*(np.log(np.maximum(q,1e-300))-logprob(z/T))).sum(1)*valid).mean()*T*T)
    gk=T*(softmax(z/T)-q)*valid[:,None]; penalty=float(np.square(theta-anchor).sum()/(2*n))
    loss=cfg['CE_weight']*ce+cfg['teacher_weight']*kd+penalty
    gradient=X.T@(cfg['CE_weight']*gc+cfg['teacher_weight']*gk)/n+(theta-anchor)/n
    return loss,np.array([ce,kd,penalty]),gradient

def predictions(state,features): return softmax(features.astype(np.float64)@state['linear.weight'].double().numpy().T+state['linear.bias'].double().numpy())

def audit(key,phase):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''; root=m.folder(key); parent,pf=m.admit(key); cfg=m.config(); meta=m.s.read(root/'metadata.json')
    assert meta['sources']==m.sources() and meta['config']==cfg and meta['parent']==str(parent); m.engine.verify_sources(meta['sources'],meta['git_commit'])
    out=root/('science' if phase=='fit' else 'saved_test_audit'); assert not out.exists()
    f=m.s.read(root/'fit/fit_result.json'); pt=torch.load(parent/'fit/selected.pt',map_location='cpu',weights_only=False); parent_state=pt['model']; selected_pt=torch.load(root/'fit/selected.pt',map_location='cpu',weights_only=False); selected=selected_pt['model']
    assert f['selected_sha256']==m.s.sha(root/'fit/selected.pt') and f['candidate_sha256']==m.s.sha(root/'fit/candidate.pt') and f['parent_pt_sha256']==m.s.sha(parent/'fit/selected.pt')
    assert set(parent_state)==set(selected) and selected['linear.weight'].shape==(20,256) and selected['linear.bias'].shape==(20,) and pt['contract']['terminal_linear_parameters']==5140
    for name in parent_state:
        if not name.startswith('linear.'): assert torch.equal(parent_state[name],selected[name]),name
    assert f['parent_core_digest']==f['selected_core_digest']==m.old.core_digest(selected)==m.old.core_digest(parent_state)
    assert selected_pt['sources']==m.sources() and selected_pt['config']==cfg and selected_pt['epoch']==pt['epoch']==f['optical_selected_epoch'] and f['optical_actual_epochs']==80
    assert f['new_optical_optimizer_steps']==f['new_inference_parameters']==f['TEST_arrays_read']==0 and f['selection_before_TEST']
    checks={}
    if phase=='fit':
        ex=m.s.read(root/'producer_exit.json'); assert ex['returncode']==0 and ex['actual_exit'] and m.gone(ex['pid'])
        arrays={r:m.old.load_np(parent/'fit'/('selected_'+r+'.npz')) for r in ('train','val')}
        teacher=torch.load(m.old.old.teacher.OUT/'selected_head.pt',map_location='cpu',weights_only=False); tf=m.old.load_np(m.old.old.teacher.OUT/'train_features.npz')
        assert np.array_equal(tf['ids'],arrays['train']['ids']) and np.array_equal(tf['labels'],arrays['train']['labels'])
        tx=((tf['features']-teacher['mean'])/teacher['std']).astype(np.float32).astype(np.float64)
        tz=tx@teacher['model']['weight'].double().numpy().T+teacher['model']['bias'].double().numpy()
        q=softmax(tz/cfg['teacher_temperature']); valid=(tz.argmax(1)==tf['labels'])&(softmax(tz).max(1)>=cfg['teacher_correct_confidence_gate'])
        assert int(valid.sum())==f['teacher_valid_samples']; X=np.c_[arrays['train']['features'].astype(np.float64),np.ones(len(q))]; y=tf['labels']
        anchor=np.r_[parent_state['linear.weight'].double().numpy().T,parent_state['linear.bias'].double().numpy()[None]]
        assert f['trace_sha256']==m.s.sha(root/'fit/closure_trace.npz') and f['optimization_sha256']==m.s.sha(root/'fit/optimization.pt')
        trace=m.old.load_np(root/'fit/closure_trace.npz'); opt=torch.load(root/'fit/optimization.pt',map_location='cpu',weights_only=False); theta=opt['theta'].numpy(); assert np.array_equal(opt['anchor'].numpy(),anchor) and opt['sources']==m.sources()
        stats=next(iter(opt['optimizer']['state'].values())); group=opt['optimizer']['param_groups'][0]
        assert f['head_LBFGS_iterations']==stats['n_iter']<=cfg['max_iter'] and f['closure_evaluations']==stats['func_evals']==len(trace['loss'])
        assert group['max_iter']==cfg['max_iter'] and group['max_eval']==cfg['max_eval'] and group['history_size']==cfg['history_size'] and group['line_search_fn']=='strong_wolfe'
        max_loss=max_parts=max_grad=0.
        for t,l,parts,gi in zip(trace['theta'],trace['loss'],trace['parts'],trace['grad_inf']):
            el,ep,eg=independent(t,X,y,q,valid,anchor,cfg); max_loss=max(max_loss,abs(el-l)); max_parts=max(max_parts,float(np.abs(ep-parts).max())); max_grad=max(max_grad,abs(float(np.abs(eg).max())-gi))
        loss,parts,gradient=independent(theta,X,y,q,valid,anchor,cfg)
        assert max_loss<1e-9 and max_parts<1e-9 and max_grad<1e-9
        assert abs(loss-f['final_objective'])<1e-9 and np.max(np.abs(gradient-opt['final_gradient'].numpy()))<1e-9 and abs(float(np.abs(gradient).max())-f['final_gradient_inf'])<1e-9
        assert f['stationary_gradient_1e5']==bool(np.abs(gradient).max()<=1e-5) and loss<=trace['loss'][0]+1e-9
        va=arrays['val']; z=np.c_[va['features'].astype(np.float64),np.ones(len(va['labels']))]@theta
        # Recompute convex positive scalar calibration independently.
        lo,hi=cfg['calibration_inverse_temperature_bounds']; weights=len(va['labels'])/(20*np.bincount(va['labels'],minlength=20)[va['labels']])
        def derivative(a): return float((weights*((softmax(a*z)*z).sum(1)-z[np.arange(len(z)),va['labels']])).mean())
        if derivative(lo)>=0: scale=lo
        elif derivative(hi)<=0: scale=hi
        else:
            for _ in range(cfg['calibration_iterations']):
                mid=(lo+hi)/2
                if derivative(mid)>0: hi=mid
                else: lo=mid
            scale=(lo+hi)/2
        assert abs(scale-f['calibration_inverse_temperature'])<1e-10
        candidate=torch.load(root/'fit/candidate.pt',map_location='cpu',weights_only=False)['model']
        assert np.max(np.abs(candidate['linear.weight'].numpy()-(theta[:-1].T*scale).astype(np.float32)))<2e-6
        assert np.max(np.abs(candidate['linear.bias'].numpy()-(theta[-1]*scale).astype(np.float32)))<2e-6
        assert np.array_equal(z.argmax(1),predictions(candidate,va['features']).argmax(1))
        cv=metrics(va['labels'],predictions(candidate,va['features'])); pv=pf['metrics']['val']
        accepted=cv['accuracy']>pv['accuracy']+1e-12 or (abs(cv['accuracy']-pv['accuracy'])<1e-12 and cv['balanced_nll']<pv['balanced_nll']-1e-6)
        assert accepted==f['candidate_accepted'] and m.s.digest(selected)==m.s.digest(candidate if accepted else parent_state)==f['selected_model_digest']
        for role,a in arrays.items():
            assert f['parent_feature_sha256'][role]==m.s.sha(parent/'fit'/('selected_'+role+'.npz')); saved=m.old.load_np(root/'fit'/('selected_'+role+'.npz'))
            for field in ('features','ids','labels'): assert np.array_equal(saved[field],a[field])
            assert saved['ids'].tolist()==m.old.old.base.manifest()['split_ids'][role]
            err=float(np.abs(predictions(selected,a['features'])-saved['probabilities']).max()); assert err<3e-6
            agree(metrics(a['labels'],saved['probabilities']),f['metrics'][role]); agree(metrics(a['labels'],predictions(candidate,a['features'])),f['candidate_metrics'][role]); agree(f['metrics'][role],selected_pt['metrics'][role]); checks[role]=dict(probability_maxabs=err)
        checks['optimization']=dict(closure_objective_maxabs=max_loss,closure_components_maxabs=max_parts,closure_gradient_inf_maxabs=max_grad,final_gradient_maxabs=float(np.abs(gradient-opt['final_gradient'].numpy()).max()),iterations=stats['n_iter'],closure_evaluations=stats['func_evals'],full_LBFGS_trajectory_replayed=False,stationarity_verified=f['stationary_gradient_1e5'])
    else:
        r=m.s.read(root/'audit_completed.json'); assert r['actual_audit_returncode']==0 and r['audit_sha256']==m.s.sha(root/'science/audit.json')
        ex=m.s.read(root/'test_exit.json'); assert ex['returncode']==0 and ex['actual_exit'] and m.gone(ex['pid'])
        result=m.s.read(root/'once_test/result.json'); saved=m.old.load_np(root/'once_test/predictions.npz'); original=m.old.load_np(parent/'once_test/predictions.npz')
        for field in ('features','ids','labels'): assert np.array_equal(saved[field],original[field])
        assert saved['ids'].tolist()==m.old.old.base.manifest()['split_ids']['test'] and result['selected_sha256']==f['selected_sha256'] and result['model_digest']==m.s.digest(selected) and result['parent_TEST_prediction_sha256']==m.s.sha(parent/'once_test/predictions.npz')
        assert result['new_optical_TEST_forwards']==0 and result['test_used_for_selection'] is False
        if result['test_reused']: assert result['production_head_TEST_evaluations']==0 and m.s.digest(selected)==m.s.digest(parent_state) and m.s.sha(root/'once_test/predictions.npz')==m.s.sha(parent/'once_test/predictions.npz')
        else: assert result['production_head_TEST_evaluations']==1
        err=float(np.abs(predictions(selected,saved['features'])-saved['probabilities']).max()); assert err<3e-6; agree(metrics(saved['labels'],saved['probabilities']),result['metrics']); checks[key]=dict(probability_maxabs=err,independent_arithmetic_replays=1)
    out.mkdir(); m.s.save(out/'audit.json',dict(m.identity(),passed=True,case=key,phase=phase,checks=checks,optical_models=0,optical_forward_calls=0,teacher_backbone_forward_calls=0,TEST_images_read=0,not_data_independence_proof=True),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--case',required=True); p.add_argument('--phase',choices=['fit','test'],required=True); a=p.parse_args()
    try: audit(a.case,a.phase)
    except BaseException as e:
        mark=m.folder(a.case)/(a.phase+'_audit_failure.json')
        if not mark.exists(): m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
