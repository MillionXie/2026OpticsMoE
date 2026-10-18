"""Original strict saved state checks and both TRAIN-only semantic constraints."""
import ast,inspect,types,numpy as np
import caltech20_rich_L6_shared_supervision as m
import audit_caltech20_rich_targeted_dualview as original
import audit_caltech20_rich_dualview_hosthash_repair as saved
tree=ast.parse(inspect.getsource(original.run));changed=0
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in node.targets):node.value=ast.BinOp(left=node.value,op=ast.Add(),right=ast.parse("v['alignment_scale']*v['feature_alignment']+cfg['auxiliary_weight']*v['shared_head_auxiliary']",mode='eval').body);changed+=1
assert changed==1;ks=dict(saved.kernel.__globals__);ks['m']=m;kernel=types.FunctionType(saved.kernel.__code__,ks,'independent_saved_state',saved.kernel.__defaults__,saved.kernel.__closure__);ns=dict(vars(original));ns.update(m=m,kernel=kernel);exec(compile(ast.fix_missing_locations(tree),__file__+':same_loss_accounting_plus_training_only_shared_head','exec'),ns);inherited=ns['run']
def run(key):
    m.select(key);inherited(key);out=m.OUT/key;features,scale,sha,info=m.prototype_data(6)
    for row in m.s.read(out/'method_trace.json'):
        v=row['view_info'];assert v['auxiliary_count']==2 and v['shared_readout_parameters']==5140 and v['shared_head_auxiliary']>=0 and v['alignment_target_sha256']==sha and abs(v['alignment_scale']-scale)<1e-15
    with np.load(m.alignment.OUT/'L6_train_targets.npz',allow_pickle=False) as z:a,y=z['features'],z['labels']
    centers=[]
    for c in range(20):v=a[y==c].astype(float).mean(0);centers.append((v-v.mean())/np.sqrt(v.var()+1e-5))
    assert np.max(np.abs(features-np.asarray(centers)[y]))<3e-6
    m.s.save(out/'science/shared_head_checks.json',dict(passed=True,TRAIN_only_shared_head_calls=2,additional_inference_heads=0,shared_parameters=5140,prototype_target_SHA256=sha,model_forward_calls=0,TEST_images_read=0),exclusive=True);q=m.s.read(out/'science/audit.json');q['shared_head_checks_sha256']=m.s.sha(out/'science/shared_head_checks.json');m.s.save(out/'science/audit.json',q)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
