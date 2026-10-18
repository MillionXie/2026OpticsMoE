"""Unchanged independent calibrated targets and derivative admission."""
import types
import caltech20_rich_currentKD_strength46 as m
import check_caltech20_rich_currentKD46 as previous
import audit_caltech20_rich_currentKD_strength46 as audit
for name in ('independent_teacher','original_run','run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,audit=audit)
    for k in ('independent_teacher','original_run'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
if __name__=='__main__':
    try:
        assert m.config()['KD_weight']==.5 and list(m.config()['cases'])==['L4_strongerKD','L6_strongerKD'];run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
