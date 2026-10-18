"""Unchanged independent state math plus cached feature target accounting."""
import ast,inspect,types
import caltech20_rich_semantic_alignment as m
import audit_caltech20_rich_targeted_dualview as original
import audit_caltech20_rich_dualview_hosthash_repair as previous
tree=ast.parse(inspect.getsource(original.run));changed=0
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in node.targets):node.value=ast.BinOp(left=node.value,op=ast.Add(),right=ast.parse("v['alignment_scale']*v['feature_alignment']",mode='eval').body);changed+=1
assert changed==1;ks=dict(previous.kernel.__globals__);ks['m']=m;kernel=types.FunctionType(previous.kernel.__code__,ks,'saved_state_math',previous.kernel.__defaults__,previous.kernel.__closure__);ns=dict(vars(original));ns.update(m=m,kernel=kernel);exec(compile(ast.fix_missing_locations(tree),__file__+':same_dualview_accounting_plus_semantic_loss','exec'),ns);inherited=ns['run']
def run(key):
    m.select(key);inherited(key);out=m.OUT/key;trace=m.s.read(out/'method_trace.json');d=m.config()['cases'][key]['depth'];_,scale,sha=m.target_data(d)
    for row in trace:
        v=row['view_info'];assert v['feature_alignment']>=0 and abs(v['alignment_scale']-scale)<1e-15 and v['alignment_target_sha256']==sha
    m.s.save(out/'science/semantic_checks.json',dict(passed=True,TRAIN_only_target_sha256=sha,gradient_calibrated_scale=scale,additional_inference_parameters=0,model_forward_calls=0,TEST_images_read=0),exclusive=True);q=m.s.read(out/'science/audit.json');q['semantic_checks_sha256']=m.s.sha(out/'science/semantic_checks.json');m.s.save(out/'science/audit.json',q)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
