"""Unchanged independent weighted-state arithmetic for one retained L2 bank."""
import inspect
import caltech20_rich_L2_local_weight_average as m
import check_caltech20_rich_weight_average as previous
import audit_caltech20_rich_L2_local_weight_average as audit
text=inspect.getsource(previous.run);assert text.count('for d in (2,4,6):')==1;text=text.replace('for d in (2,4,6):','for d in (2,):');ns=dict(previous.run.__globals__);ns.update(m=m,audit=audit);exec(compile(text,__file__+':independent_L2_local_state_arithmetic','exec'),ns);run=ns['run']
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
