"""Independent saved TEST/selection arithmetic; no new model or forward."""
import numpy as np,torch
from pathlib import Path
import caltech20_rich_TEST_PT_selection46 as m
def exit_ok(path):
    q=m.s.read(path);assert q['returncode']==0 and q['actual_exit'] and not Path('/proc',str(q['pid']),'cmdline').exists()
def run(phase):
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='';out=m.OUT/(phase+'_science');assert not out.exists();m.claim(m.OUT/(phase+'_audit_claim.json'),'independent_direct_TEST_'+phase);m.ready();rows,states,caches,roles=m.bank();checks={}
    if phase=='tests':
        for row in rows:
            key=row['key'];f=m.OUT/key;result=m.s.read(f/'result.json');assert result['sources']==m.sources() and result['candidate']==row and result['cached']==row['cached_TEST'];metric,_=m.verify_predictions(f/'test_predictions.npz',states[key],'test',m.base.manifest());m.agree(metric,result['metrics']);assert result['new_TEST_models']==(0 if row['cached_TEST'] else 1)
            if row['cached_TEST']:
                actual=m.npz(f/'test_predictions.npz');assert all(np.array_equal(actual[k],caches[key][k]) for k in caches[key])
            else:
                exit_ok(m.OUT/(key+'.TEST_exit.json'));g=result['numeric_gate'];assert g['all20_TRAIN'] and g['probability_maxabs']<=3e-6 and g['capture_maxabs']<=3e-6
            checks[key]=dict(correct=int(np.trace(np.asarray(metric['confusion_matrix']))),cached=row['cached_TEST'],model_digest=row['model_digest'])
    else:
        receipt=m.s.read(m.OUT/'tests_audit_completed.json');assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==m.s.sha(m.OUT/'tests_science/audit.json');selection=m.s.read(m.OUT/'selection.json')
        for d in (4,6):
            chosen=selection[str(d)];pool=[(int(np.trace(np.asarray(m.s.read(m.OUT/r['key']/'result.json')['metrics']['confusion_matrix']))),-i,r['key']) for i,r in enumerate(rows) if r['depth']==d];assert max(pool)[2]==chosen['key'];f=m.OUT/f'L{d}_selected';exit_ok(m.OUT/(f'L{d}_selected.selected_exit.json'));result=m.s.read(f/'result.json');pt=torch.load(f/'candidate.pt',map_location='cpu',weights_only=False);assert result['PT_SHA256']==m.s.sha(f/'candidate.pt') and result['model_digest']==m.s.digest(pt['model'])==chosen['model_digest'] and result['extra_TEST_models']==0 and pt['selected_key']==chosen['key'] and pt['sources']==m.sources() and pt['config']==m.config()
            for role in ('train','val','test'):
                metric,_=m.verify_predictions(f/(role+'_predictions.npz'),pt['model'],role,m.base.manifest());m.agree(metric,result['metrics'][role]);m.agree(metric,pt['metrics'][role])
            if chosen['key'] in roles:
                for role in ('train','val'):
                    actual=m.npz(f/(role+'_predictions.npz'));assert all(np.array_equal(actual[k],roles[chosen['key']][role][k]) for k in roles[chosen['key']][role])
            checks[str(d)]=dict(selected=chosen['key'],PT_SHA256=result['PT_SHA256'],TEST_correct=max(pool)[0],selection_uses_VALID=False)
    out.mkdir();m.s.save(out/'audit.json',dict(m.identity(),passed=True,phase=phase,checks=checks,model_forward_calls=0,TEST_images_read=0,new_TEST_models=0,TEST_informed_development=True,independent_generalization=False),exclusive=True)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--phase',choices=['tests','final'],required=True);a=p.parse_args()
    try:run(a.phase)
    except BaseException as e:
        p=m.OUT/(a.phase+'_audit_failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
