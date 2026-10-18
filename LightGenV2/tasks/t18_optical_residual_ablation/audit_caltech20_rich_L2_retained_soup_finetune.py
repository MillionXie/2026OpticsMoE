"""Full independent retained math with explicitly local adaptation epoch labels."""
import ast,inspect,types
import caltech20_rich_L2_retained_soup_finetune as m
import audit_caltech20_rich_alpha40_rebalance as static
import audit_caltech20_rich_phasecode_gate_contract as previous
text=inspect.getsource(static.run);site="epoch=tune if case['fresh'] else q['joint_budget_epochs']+tune";assert text.count(site)==2;text=text.replace(site,'epoch=tune');tree=ast.parse(text);changed=0
class Projection(ast.NodeTransformer):
    def visit_Assign(self,node):
        global changed
        if any(isinstance(t,ast.Name) and t.id=='raw' for t in node.targets):changed+=1;return [node,ast.parse("if n.startswith('fusion_logits.'):raw=raw.clamp(m.ALPHA_MIN,.95)").body[0]]
        return self.generic_visit(node)
tree=Projection().visit(tree);assert changed==1;ns=dict(static.run.__globals__);ns['m']=m;exec(compile(ast.fix_missing_locations(tree),__file__+':independent_original_math_local_epochs','exec'),ns);kernel=ns['run']
ns=dict(previous.inherited.__globals__);ns.update(m=m,kernel=kernel);inherited=types.FunctionType(previous.inherited.__code__,ns,'same_semantic_loss_math',previous.inherited.__defaults__,previous.inherited.__closure__)
ns=dict(previous.base_run.__globals__);ns.update(m=m,inherited=inherited);base_run=types.FunctionType(previous.base_run.__code__,ns,'same_loss_accounting',previous.base_run.__defaults__,previous.base_run.__closure__)
ns=dict(previous.prototype_run.__globals__);ns.update(m=m,base_run=base_run);prototype_run=types.FunctionType(previous.prototype_run.__code__,ns,'same_independent_prototype',previous.prototype_run.__defaults__,previous.prototype_run.__closure__)
ns=dict(previous.phase_run.__globals__);ns.update(m=m,prototype_run=prototype_run);phase_run=types.FunctionType(previous.phase_run.__code__,ns,'same_phase_math',previous.phase_run.__defaults__,previous.phase_run.__closure__)
ns=dict(previous.run.__globals__);ns.update(m=m,phase_run=phase_run);retained_run=types.FunctionType(previous.run.__code__,ns,'same_retained_science',previous.run.__defaults__,previous.run.__closure__)
def run(key):
    retained_run(key);out=m.OUT/key;fit=m.s.read(out/'fit_result.json');meta=m.s.read(out/'metadata.json');assert fit['not_fresh_initialization'] and fit['local_added_epochs']==10 and fit['previous_budget_epochs'] is None and fit['previous_training_has_no_single_epoch'] and meta['parent']['parameter_average'] and not meta['case']['fresh'];proof=m.s.read(out/'science/audit.json');proof.update(local_adaptation_epoch_labels=True,not_from_scratch=True,canonical_parent_SHA256=meta['parent']['sha256']);m.s.save(out/'science/audit.json',proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
