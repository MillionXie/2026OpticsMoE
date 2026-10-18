"""Complete inherited namespaces, unchanged science plus gate field equality."""
import types
import caltech20_rich_phasecode_gate_contract as m
import audit_caltech20_rich_phasecode_consistency as previous
ks=dict(previous.kernel.__globals__);ks['m']=m;kernel=types.FunctionType(previous.kernel.__code__,ks,'same_saved_state_math',previous.kernel.__defaults__,previous.kernel.__closure__)
ns=dict(previous.inherited.__globals__);ns.update(m=m,kernel=kernel);inherited=types.FunctionType(previous.inherited.__code__,ns,'same_semantic_loss_math',previous.inherited.__defaults__,previous.inherited.__closure__)
ns=dict(previous.base_run.__globals__);ns.update(m=m,inherited=inherited);base_run=types.FunctionType(previous.base_run.__code__,ns,'same_loss_accounting',previous.base_run.__defaults__,previous.base_run.__closure__)
ns=dict(previous.prototype_run.__globals__);ns.update(m=m,base_run=base_run);prototype_run=types.FunctionType(previous.prototype_run.__code__,ns,'same_independent_prototype_centers',previous.prototype_run.__defaults__,previous.prototype_run.__closure__)
ns=dict(previous.run.__globals__);ns.update(m=m,prototype_run=prototype_run);phase_run=types.FunctionType(previous.run.__code__,ns,'same_phasecode_math',previous.run.__defaults__,previous.run.__closure__)
def run(key):
    phase_run(key);p=m.OUT/key;gate=m.s.read(p/'numeric_gate.json');assert gate['parent_clean_capture_maxabs']==gate['parent_capture_maxabs']<=3e-6;proof=m.s.read(p/'science/audit.json');proof['gate_capture_key_contract_exactly_equal']=True;m.s.save(p/'science/audit.json',proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
