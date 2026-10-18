"""All original saved-state math plus exact TRAIN class center reconstruction."""
import types,numpy as np
import caltech20_rich_prototype_consistency as m
import audit_caltech20_rich_semantic_alignment as previous
ks=dict(previous.kernel.__globals__);ks['m']=m;kernel=types.FunctionType(previous.kernel.__code__,ks,'saved_state_math',previous.kernel.__defaults__,previous.kernel.__closure__)
ns=dict(previous.inherited.__globals__);ns.update(m=m,kernel=kernel);inherited=types.FunctionType(previous.inherited.__code__,ns,'same_semantic_saved_accounting',previous.inherited.__defaults__,previous.inherited.__closure__)
ns=dict(vars(previous));ns.update(m=m,inherited=inherited);base_run=types.FunctionType(previous.run.__code__,ns,'same_semantic_audit',previous.run.__defaults__,previous.run.__closure__)
def run(key):
    base_run(key);d=m.config()['cases'][key]['depth'];features,scale,sha,info=m.prototype_data(d)
    with np.load(m.alignment.OUT/f'L{d}_train_targets.npz',allow_pickle=False) as z:a,y=z['features'],z['labels']
    centers=[]
    for c in range(20):v=a[y==c].astype(float).mean(0);centers.append((v-v.mean())/np.sqrt(v.var()+1e-5))
    assert np.max(np.abs(features-np.asarray(centers)[y]))<3e-6
    out=m.OUT/key;m.s.save(out/'science/prototype_checks.json',dict(passed=True,TRAIN3141_only=True,prototype_count=20,parent_SHA256=m.incumbent(d)[0]['sha256'],target_SHA256=sha,scale=scale,new_inference_parameters=0,new_model_forwards=0,TEST_calls=0),exclusive=True);q=m.s.read(out/'science/audit.json');q['prototype_checks_sha256']=m.s.sha(out/'science/prototype_checks.json');m.s.save(out/'science/audit.json',q)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
