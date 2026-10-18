"""Complete callable namespaces, unchanged independent prototype audit."""
import types,numpy as np
import caltech20_rich_L4_prototype_continue as m
import audit_caltech20_rich_prototype_consistency as original
ks=dict(original.kernel.__globals__);ks['m']=m;kernel=types.FunctionType(original.kernel.__code__,ks,'saved_state_math',original.kernel.__defaults__,original.kernel.__closure__)
ns=dict(original.inherited.__globals__);ns.update(m=m,kernel=kernel);inherited=types.FunctionType(original.inherited.__code__,ns,'prototype_loss_accounting',original.inherited.__defaults__,original.inherited.__closure__)
ns=dict(original.base_run.__globals__);ns.update(m=m,inherited=inherited);base_run=types.FunctionType(original.base_run.__code__,ns,'semantic_and_prototype_saved_checks',original.base_run.__defaults__,original.base_run.__closure__)
ns=dict(original.run.__globals__);ns.update(m=m,base_run=base_run,np=np);run=types.FunctionType(original.run.__code__,ns,'run',original.run.__defaults__,original.run.__closure__)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
