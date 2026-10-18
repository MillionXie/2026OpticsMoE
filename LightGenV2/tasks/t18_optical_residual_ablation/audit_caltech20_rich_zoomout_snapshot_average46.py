"""Independent saved prediction science and interpolation from actual raw PTs."""
import types,inspect,numpy as np,torch
import caltech20_rich_zoomout_snapshot_average46 as m
import audit_caltech20_rich_weight_average as previous
text=inspect.getsource(previous.verify);site="test['new_TEST_models']==(0 if same else 1)";assert text.count(site)==1;text=text.replace(site,"test['new_TEST_models']==(0 if same or test.get('known_endpoint_TEST_reused',False) else 1)");ns=dict(previous.verify.__globals__);ns['m']=m;exec(compile(text,__file__+':existing_saved_science_with_explicit_known_endpoint_cache','exec'),ns);original_verify=ns['verify']
def verify(depth,phase):
    result=original_verify(depth,phase);states,parents=m.bank(depth);raw=torch.load(parents[1]['path'],map_location='cpu',weights_only=False)[parents[1]['state_key']];pt=torch.load(m.OUT/f'L{depth}/candidate.pt',map_location='cpu',weights_only=False);f=pt['selected_fraction']
    for n,v in pt['model'].items():
        if v.is_floating_point():assert np.allclose(v.numpy(),(1-f)*states[0][n].numpy().astype(float)+f*raw[n].numpy().astype(float),atol=2e-7,rtol=1e-7)
        else:assert torch.equal(v,raw[n])
    if phase=='test':
        source=m.train.OUT/f'L{depth}_rare_zoomout';last=torch.load(source/'last.pt',map_location='cpu',weights_only=False);known=m.s.digest(pt['model'])==m.s.digest(last['ema']);same=m.s.digest(pt['model'])==m.s.digest(states[0]);test=m.s.read(m.OUT/f'L{depth}/test_result.json');assert test['known_endpoint_TEST_reused']==(known and not same)
        if known and not same:
            old=m.retained.load_predictions(source/'test_epoch05.npz');actual=m.retained.load_predictions(m.OUT/f'L{depth}/candidate_test.npz');assert all(np.array_equal(old[k],actual[k]) for k in old)
    result.update(independent_raw_preserved_state_interpolation=True,original_zoomout_own_science_verified=True);return result
text=inspect.getsource(previous.run);assert text.count('for d in (2,4,6)')==1;text=text.replace('for d in (2,4,6)','for d in (4,6)');ns=dict(previous.run.__globals__);ns.update(m=m,verify=verify);exec(compile(text,__file__+':existing_independent_science_only46','exec'),ns);run=ns['run']
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:run(a.phase)
    except BaseException as e:
        p=m.OUT/(a.phase+'_audit_failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
