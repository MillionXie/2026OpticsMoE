"""Independent frozen-core, objective gradient and cached prediction audit."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_robust_head as m
from check_caltech20_rich_robust_head import numpy_objective

def audit(depth,phase):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(4);out=m.OUT/f'L{depth}_rho03';dest=out/('science' if phase=='fit' else 'test_science');assert not dest.exists()
    meta=m.s.read(out/'metadata.json');assert meta['sources']==m.sources() and meta['config']==m.config();m.engine.verify_sources(meta['sources'],meta['git_commit']);choice,mod,parent=m.get_parent(depth);assert meta['parent']==choice and meta['parent_model_digest']==m.s.digest(parent['ema'])
    ex=m.s.read(out/('producer_exit.json' if phase=='fit' else 'test_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
    pt=torch.load(out/'candidate.pt',map_location='cpu',weights_only=False);f=m.s.read(out/'fit_result.json');assert f['candidate_sha256']==m.s.sha(out/'candidate.pt') and pt['sources']==m.sources() and m.core_digest(pt['model'])==m.core_digest(parent['ema'])==meta['parent_core_digest']
    checks={}
    if phase=='fit':
        proof=m.s.read(out/'extract_result.json');assert proof['passed'] and proof['model_unchanged'] and meta['optimizer_steps']==meta['TEST_images_read']==meta['new_optical_TEST_forwards']==0
        for role in ('train','val'):
            assert proof['features_sha256'][role]==m.s.sha(out/(role+'_features.npz'))
            actual,receipt=m.mathcheck.verify_predictions(out/('candidate_'+role+'.npz'),pt['model'],role,m.robust.base.manifest());m.mathcheck.agree(actual,f['metrics'][role]);checks[role]=receipt
        opt=torch.load(out/'optimization.pt',map_location='cpu',weights_only=False);train=m.load_np(out/'train_features.npz');X=np.c_[train['features'].astype(float),np.ones(len(train['labels']))];q,valid=m.targets();anchor=m.theta(parent['ema']);lam=m.config()['anchor_by_depth'][str(depth)]
        assert np.array_equal(opt['anchor'].numpy(),anchor) and m.s.digest(opt['sources'])==m.s.digest(meta['sources'])
        value,gradient=numpy_objective(opt['theta'].numpy(),X,train['labels'],q.astype(float),valid.astype(float),anchor,lam)
        initial,ig=numpy_objective(anchor,X,train['labels'],q.astype(float),valid.astype(float),anchor,lam)
        error=float(np.max(np.abs(gradient-opt['final_gradient'].numpy())));assert error<=1e-9 and abs(value-f['final_objective'])<=1e-8 and abs(initial-f['initial_objective'])<=1e-8 and abs(np.abs(ig).max()-f['initial_gradient_inf'])<=1e-9
        assert value<=initial+1e-9 and f['fit_needed']==bool(np.abs(ig).max()>1e-5) and 0<=f['actual_LBFGS_iterations']<=200
        expected=m.theta(pt['model']);assert np.allclose(expected,opt['theta'].numpy(),atol=5e-7,rtol=1e-7)
        for n,v in pt['model'].items():
            if not n.startswith('linear.'):assert torch.equal(v,parent['ema'][n]),n
        checks['objective']=dict(independent_gradient_maxabs=error,initial_gradient_inf=f['initial_gradient_inf'],initial_objective=initial,final_objective=value,LBFGS_iterations=f['actual_LBFGS_iterations'],all_nonhead_parameters_frozen=True)
    else:
        rec=m.s.read(out/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==m.s.sha(out/'science/audit.json')
        r=m.s.read(out/'test_result.json');assert r['test_used_for_selection'] and not r['independent_generalization'] and r['new_optical_TEST_forwards']==r['TEST_images_read']==0 and r['model_digest']==m.s.digest(pt['model'])
        actual,receipt=m.mathcheck.verify_predictions(out/'candidate_test.npz',pt['model'],'test',m.robust.base.manifest());m.mathcheck.agree(actual,r['metrics']);checks['test']=receipt
    dest.mkdir();m.s.save(dest/'audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],checks=checks,phase=phase,model_forward_calls=0,TEST_images_read=0,test_used_for_selection=True,frozen_controls=m.controls()),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--depth',required=True,type=int);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.depth,a.phase)
    except BaseException as e:
        out=m.OUT/f'L{a.depth}_rho03';mark=out/(a.phase+'_audit_failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
