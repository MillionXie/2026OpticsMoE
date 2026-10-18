"""Independent checkpoint arithmetic and saved prediction verification."""
import numpy as np,torch
from pathlib import Path
import caltech20_rich_weight_average as m
def verify(depth,phase):
    f=m.OUT/f'L{depth}';ex=m.s.read(m.OUT/f'L{depth}.{phase}_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();meta=m.s.read(f/'metadata.json');fit=m.s.read(f/'fit_result.json');pt=torch.load(f/'candidate.pt',map_location='cpu',weights_only=False);states,parents=m.bank(depth);assert meta['sources']==pt['sources']==m.sources() and meta['config']==pt['config']==m.config() and meta['parents']==pt['contributing_parents']==parents;m.engine.verify_sources(meta['sources'],meta['git_commit']);assert fit['candidate_SHA256']==m.s.sha(f/'candidate.pt') and m.s.digest(pt['model'])==fit['model_digest'];fraction=fit['selected_fraction'];maximum=0.
    for n,v in pt['model'].items():
        if v.is_floating_point():expected=(1-fraction)*states[0][n].numpy().astype(float)+fraction*(states[1][n].numpy().astype(float)+states[2][n].numpy().astype(float))/2;maximum=max(maximum,float(np.max(np.abs(v.numpy().astype(float)-expected))));assert np.allclose(v.numpy(),expected,atol=2e-7,rtol=1e-7)
        else:assert all(torch.equal(v,z[n]) for z in states)
    assert fit['optimizer_steps']==fit['new_joint_epochs']==meta['new_inference_parameters']==0 and sum(v.numel() for n,v in pt['model'].items() if n.startswith('linear.'))==5140;gate=meta['numeric_gate'];assert gate['all20'] and gate['probability_maxabs']<=3e-6 and gate['capture_maxabs']<=3e-6
    if phase=='fit':
        assert len(fit['records'])==4;rank=[]
        for i,row in enumerate(fit['records']):
            state=m.state(depth,row['fraction']);assert m.s.digest(state)==row['model_digest'];metric,_=m.verify_predictions(f/f'VALID_choice{i}.npz',state,'val',m.base.manifest());m.agree(metric,row['val']);rank.append((int(np.trace(np.asarray(metric['confusion_matrix']))),-metric['balanced_nll'],-row['fraction'],i))
        assert max(rank)[3]==fit['selected_index'] and fraction==fit['records'][fit['selected_index']]['fraction']
        for role in ('train','val'):metric,_=m.verify_predictions(f/('candidate_'+role+'.npz'),pt['model'],role,m.base.manifest());m.agree(metric,fit['metrics'][role]);m.agree(metric,pt['metrics'][role])
    else:
        rec=m.s.read(m.OUT/'fit_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==m.s.sha(m.OUT/'fit_science/audit.json');test=m.s.read(f/'test_result.json');metric,_=m.verify_predictions(f/'candidate_test.npz',pt['model'],'test',m.base.manifest());m.agree(metric,test['metrics']);same=m.s.digest(pt['model'])==m.s.digest(states[0]);assert test['parent_TEST_reused']==same and test['new_TEST_models']==(0 if same else 1) and not test['TEST_used_to_select_fraction']
        if same:old=m.parent(depth)[2]['test'];actual=m.previous.load_np(f/'candidate_test.npz');assert all(np.array_equal(old[k],actual[k]) for k in old)
    return dict(independent_weight_arithmetic_maxabs=maximum,selected_fraction=fraction,PT_SHA256=fit['candidate_SHA256'],single_network_not_prediction_ensemble=True,no_new_optimizer_or_inference_parameters=True)
def run(phase):
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(4);out=m.OUT/(phase+'_science');assert not out.exists();m.claim(m.OUT/(phase+'_audit_claim.json'),'independent_weight_average_'+phase);checks={str(d):verify(d,phase) for d in (2,4,6)};m.identity();m.previous.closure.diagnosis.closure.old.controls();out.mkdir();m.s.save(out/'audit.json',dict(m.identity(),passed=True,checks=checks,phase=phase,optical_model_forward_calls=0,TEST_images_read=0,TEST_informed_development=True,independent_generalization=False),exclusive=True)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:run(a.phase)
    except BaseException as e:
        f=m.OUT/(a.phase+'_audit_failure.json')
        if not f.exists():m.s.save(f,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
