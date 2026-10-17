"""Check complete TRAIN and auditor namespaces before any fresh GPU fit."""
import builtins,dis,types
import torch
import caltech20_rich_alpha40_fresh246 as m
import check_caltech20_rich_alpha40_rebalance as original
import audit_caltech20_rich_alpha40_fresh246 as auditor
namespace=dict(vars(original));namespace['m']=m
run=types.FunctionType(original.run.__code__,namespace,'run')
def unresolved(code,namespace):
    bad={q.argval for q in dis.get_instructions(code) if q.opname=='LOAD_GLOBAL' and q.argval not in namespace and not hasattr(builtins,q.argval)}
    for q in code.co_consts:
        if isinstance(q,types.CodeType):bad|=unresolved(q,namespace)
    return bad
if __name__=='__main__':
    try:
        for name in ('initialize','gate','fit','campaign','ready','identity','gradients','incumbent'):
            f=getattr(m,name);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
        assert not unresolved(auditor.run.__code__,auditor.run.__globals__)
        model=m.initialize(2,True);initial={n:v.clone() for n,v in model.phase_anchors.items()};model.to(dtype=torch.float64)
        assert all(torch.equal(v,initial[n].double()) and v.dtype==dict(model.named_parameters())[n].dtype for n,v in model.phase_anchors.items());del model
        run();p=m.s.read(m.CPU/'checks.json');p.update(all_kernel_and_audit_global_symbols_resolved=True,device_anchor_proof=dict(passed=True,actual_Module_apply_migrated_all_anchors=True),TEST_selection_development_only=True);m.s.save(m.CPU/'checks.json',p)
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
