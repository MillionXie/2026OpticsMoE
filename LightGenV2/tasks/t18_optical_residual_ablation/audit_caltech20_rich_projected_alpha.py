"""Independent projected-Adam boundary, curriculum freezes and saved selection."""
import ast,inspect,types
import caltech20_rich_projected_alpha as m
import audit_caltech20_rich_alpha40_rebalance as kernel
tree=ast.parse(inspect.getsource(kernel.run));counts=dict(step=0,projection=0)
class ProjectedBoundary(ast.NodeTransformer):
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='expected_step' for t in node.targets):
            node.value=ast.parse("197*(case['epochs']-cfg['alpha_curriculum_epochs']) if n.startswith('fusion_logits.') else 197*case['epochs']",mode='eval').body;counts['step']+=1
        if any(isinstance(t,ast.Name) and t.id=='raw' for t in node.targets):
            counts['projection']+=1;extra=ast.parse("if n.startswith('fusion_logits.'):raw=raw.clamp(.4001,.95)").body[0];return [node,extra]
        return node
tree=ProjectedBoundary().visit(tree);assert counts==dict(step=1,projection=1)
namespace=dict(vars(kernel));namespace['m']=m;exec(compile(ast.fix_missing_locations(tree),__file__+':independent_projected_boundary','exec'),namespace);run=namespace['run']
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:
        run(a.key);out=m.OUT/a.key;h=m.s.read(out/'history.json');trace=m.s.read(out/'method_trace.json');cfg=m.config()
        for row in h[:cfg['alpha_curriculum_epochs']]:assert all(abs(v-m.alpha_schedule(row['tune_epoch']))<1e-7 for v in row['alpha'])
        for row in trace:assert row['alpha_trainable']==(row['tune']>cfg['alpha_curriculum_epochs'])
        assert all(.4<v<=.95 for row in h for v in row['alpha'])
        m.s.save(out/'science/curriculum_checks.json',dict(passed=True,alpha_frozen_epochs=cfg['alpha_curriculum_epochs'],direct_projection_after_each_Adam=True,model_forward_calls=0),exclusive=True)
        proof=m.s.read(out/'science/audit.json');proof['curriculum_checks_sha256']=m.s.sha(out/'science/curriculum_checks.json');m.s.save(out/'science/audit.json',proof)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
