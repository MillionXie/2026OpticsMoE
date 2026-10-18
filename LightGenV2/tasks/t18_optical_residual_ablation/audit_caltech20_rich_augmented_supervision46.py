"""Independent weighted-view scalar math and TEST-only earliest selection."""
import types,inspect,ast
import caltech20_rich_augmented_supervision46 as m
import audit_caltech20_rich_view_target_consistency46 as previous
import audit_caltech20_rich_targeted_dualview as view_source
f=previous.kernel;ns=dict(f.__globals__);ns['m']=m;kernel=types.FunctionType(f.__code__,ns,'independent_TEST_only_local_epoch_math',f.__defaults__,f.__closure__)
tree=ast.parse(inspect.getsource(view_source.run));changes=0
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in node.targets):node.value=ast.parse("cfg['clean_CE_coefficient']*v['clean']+cfg['augmented_CE_coefficient']*v['augmented']+cfg['KD_weight']*v['KD_clean']+cfg['consistency_weight']*v['JS']+cfg['phase_anchor_weight']*v['phase_anchor']+v['alignment_scale']*v['feature_alignment']",mode='eval').body;changes+=1
assert changes==1;ns=dict(view_source.run.__globals__);ns.update(m=m,kernel=kernel);exec(compile(ast.fix_missing_locations(tree),__file__+':independent_augmented_supervision_math','exec'),ns);inherited=ns['run']
for name in ('base_run','prototype_run','phase_run','retained_run','original_run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,kernel=kernel,inherited=inherited)
    for k in ('base_run','prototype_run','phase_run','retained_run','original_run'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
f=previous.inherited_run;ns=dict(f.__globals__);ns.update(m=m,original_run=original_run);zoom_audit=types.FunctionType(f.__code__,ns,'independent_unchanged_all_zoomout_plans',f.__defaults__,f.__closure__)
def run(key):
    zoom_audit(key);out=m.OUT/key;cfg=m.config();trace=m.s.read(out/'method_trace.json')
    for row in trace:
        v=row['view_info'];assert v['clean_CE_coefficient']==.2 and v['augmented_CE_coefficient']==.8 and v['teacher_and_prototype_student_view']=='clean_only' and v['consistency_target']=='clean_targets';assert abs(v['KD_clean']-v['KD_clean_actual'])<1e-7 and abs(v['feature_alignment']-v['prototype_clean_actual'])<1e-7
    q=m.s.read(out/'science/audit.json');q.update(independent_supervised_view_weights=[.2,.8],total_supervised_coefficient_unchanged=True,microbatch_size=4,production_micro1vs4_all20_tail5=True,TEST_integer_only_earliest_tie=True);m.s.save(out/'science/audit.json',q)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
