"""Independent saved math, projection, view accounting and retained-source audit."""
import ast,inspect
import caltech20_rich_targeted_dualview as m
import audit_caltech20_rich_alpha40_rebalance as original
tree=ast.parse(inspect.getsource(original.run));changes=0
class Projection(ast.NodeTransformer):
    def visit_Assign(self,node):
        global changes
        if any(isinstance(t,ast.Name) and t.id=='raw' for t in node.targets):changes+=1;return [node,ast.parse("if n.startswith('fusion_logits.'):raw=raw.clamp(m.ALPHA_MIN,.95)").body[0]]
        return self.generic_visit(node)
tree=Projection().visit(tree);assert changes==1;namespace=dict(vars(original));namespace['m']=m;exec(compile(ast.fix_missing_locations(tree),__file__+':same_saved_checks_with_physical_projection','exec'),namespace);kernel=namespace['run']
def run(key):
    m.select(key);kernel(key);out=m.OUT/key;cfg=m.config();case=cfg['cases'][key];trace=m.s.read(out/'method_trace.json');checks=[]
    for row in trace:
        v=row['view_info'];expected=.5*v['clean']+.5*v['augmented']+cfg['KD_weight']*v['KD_clean']+cfg['consistency_weight']*v['JS']+cfg['phase_anchor_weight']*v['phase_anchor']
        assert abs(expected-v['weighted_sum'])<=2e-6*max(1,abs(expected)) and abs(v['weighted_sum']-row['loss'])<=1e-7 and v['JS']>=-1e-6 and v['phase_anchor']>=0
        assert v['full_rms_gradient']==case['full_rms_gradient'] and v['teacher_view']=='unaugmented_MixUp_same_ids_permutation_mass' and len(v['clean_macro_sha256'])==64 and v['new_inference_parameters']==0
    gate=m.s.read(out/'numeric_gate.json');assert gate['parent_clean_probability_maxabs']<=3e-6 and gate['parent_clean_capture_maxabs']<=3e-6 and gate['full_rms_gradient']==case['full_rms_gradient']
    md=m.s.read(out/'metadata.json');assert md['optimizer_reset'] and md['alpha_restart'] is None and md['config']['reset_trained_weights'] is False
    assert all(.4001<a<.95 for row in m.s.read(out/'history.json') for a in row['alpha'])
    m.s.save(out/'science/view_and_derivative_checks.json',dict(passed=True,macros=len(trace),full_rms_gradient=case['full_rms_gradient'],teacher_clean_only=True,identical_inference_parameter_count=True,model_forward_calls=0,TEST_images_read=0),exclusive=True)
    proof=m.s.read(out/'science/audit.json');proof['view_checks_sha256']=m.s.sha(out/'science/view_and_derivative_checks.json');m.s.save(out/'science/audit.json',proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
