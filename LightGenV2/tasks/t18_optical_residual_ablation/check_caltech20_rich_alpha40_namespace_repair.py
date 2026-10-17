"""Resolved-symbol and actual anchor lifecycle plus unchanged CPU contract."""
import ast,builtins,dis,inspect,os,types
import numpy as np
import torch
import caltech20_rich_alpha40_namespace_repair as m
import check_caltech20_rich_alpha40_rebalance as original
exec(compile(ast.parse(inspect.getsource(original.run)),__file__+':same_CPU_contract','exec'),globals())
def unresolved(code,namespace):
    bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in namespace and not hasattr(builtins,i.argval)}
    for value in code.co_consts:
        if isinstance(value,types.CodeType):bad|=unresolved(value,namespace)
    return bad
if __name__=='__main__':
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='';evidence=m.failed_preflight()
        for name in ('initialize','gate','fit','campaign','gradients','ready','identity','incumbent'):
            f=getattr(m,name);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
        model=m.initialize(2,False);initial={n:v.clone() for n,v in model.phase_anchors.items()};assert initial
        keys=set(model.state_dict());model.to(dtype=torch.float64)
        assert set(model.state_dict())==keys and all(v.dtype==dict(model.named_parameters())[n].dtype==torch.float64 for n,v in model.phase_anchors.items())
        assert all(torch.equal(v,initial[n].double()) for n,v in model.phase_anchors.items())
        name=next(iter(model.phase_anchors));model.phase_anchors[name]=model.phase_anchors[name].to('meta')
        try:model.context(1,1,m.config())
        except AssertionError:pass
        else:raise AssertionError('stale device anchor accepted')
        del model
        run();p=m.s.read(m.CPU/'checks.json');p.update(original_failed_preflight=evidence,device_anchor_proof=dict(passed=True,actual_Module_apply_migrated_all_anchors=True,parameter_and_state_keys_unchanged=True,rejected_stale_meta_anchor=True,real_forward_calls=0),all_kernel_global_symbols_resolved=True);m.s.save(m.CPU/'checks.json',p)
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
