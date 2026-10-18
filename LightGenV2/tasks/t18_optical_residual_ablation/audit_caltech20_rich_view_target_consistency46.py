"""Independent old science plus raw two-view scalar allocation and TEST-only rank."""
import types,numpy as np,json,hashlib,inspect,ast
import caltech20_rich_view_target_consistency46 as m
import audit_caltech20_rich_rare_zoomout46 as previous
import audit_caltech20_rich_alpha40_rebalance as static
text=inspect.getsource(static.run)
assert text.count("epoch=tune if case['fresh'] else q['joint_budget_epochs']+tune")==2;text=text.replace("epoch=tune if case['fresh'] else q['joint_budget_epochs']+tune",'epoch=tune');assert text.count('key=lambda a:(a[1],a[2])')==1;text=text.replace('key=lambda a:(a[1],a[2])','key=lambda a:a[1]')
tree=ast.parse(text);changes=0
class Projection(ast.NodeTransformer):
    def visit_Assign(self,node):
        global changes
        if any(isinstance(t,ast.Name) and t.id=='raw' for t in node.targets):changes+=1;return [node,ast.parse("if n.startswith('fusion_logits.'):raw=raw.clamp(m.ALPHA_MIN,.95)").body[0]]
        return self.generic_visit(node)
tree=Projection().visit(tree);assert changes==1;ns=dict(static.run.__globals__);ns['m']=m;exec(compile(ast.fix_missing_locations(tree),__file__+':independent_direct_TEST_local_epochs','exec'),ns);kernel=ns['run']
chain=('inherited','base_run','prototype_run','phase_run','retained_run','original_run','run')
for name in chain:
    f=getattr(previous,name);ns=dict(f.__globals__);ns['m']=m;ns['kernel']=kernel
    for k in chain:
        if k in globals():ns[k]=globals()[k]
    # Correct the inherited saved report to reflect student supervision on both views.
    if 'teacher_clean_only' in f.__code__.co_consts:
        code=inspect.getsource(f).replace('teacher_clean_only=True',"teacher_clean_only=m.ACTIVE_VARIANT!='KD_both'");exec(compile(code,__file__+':accurate_student_views','exec'),ns);globals()[name]=ns[f.__name__]
    else:globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
inherited_run=run
def run(key):
    inherited_run(key);f=m.OUT/key;cfg=m.config();mode=cfg['cases'][key]['consistency_target'];trace=m.s.read(f/'method_trace.json')
    for row in trace:
        v=row['view_info'];kc,ka,pc,pa=[v[k] for k in ('KD_clean_actual','KD_aug_actual','prototype_clean_actual','prototype_aug_actual')];kd=kc if mode=='prototype_both' else .5*(kc+ka);proto=.5*(pc+pa) if mode=='prototype_both' else pc
        assert abs(v['KD_clean']-kd)<2e-6*max(1,abs(kd)) and abs(v['feature_alignment']-proto)<2e-6*max(1,abs(proto));assert v['consistency_target']==mode and v['same_total_auxiliary_coefficients'] and v['legacy_KD_clean_field_is_allocated_KD_term']
    viewpath=f/'science/view_and_derivative_checks.json';view=m.s.read(viewpath);view.update(teacher_clean_only=mode!='KD_both',teacher_reference_input_clean=True,student_target_allocation=mode);m.s.save(viewpath,view)
    proof=m.s.read(f/'science/audit.json');proof.update(independent_actual_two_view_allocation=True,student_target_allocation=mode,view_checks_sha256=m.s.sha(viewpath),direct_TEST_integer_count_no_VALID_tie_break=True,extra_forward_calls=0,unchanged_inference_graph=True);m.s.save(f/'science/audit.json',proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
