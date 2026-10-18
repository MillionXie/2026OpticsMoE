"""Independent float64 weighted-state arithmetic; no new model/forward."""
import inspect
import caltech20_rich_phasecode_weight_average46 as m
import check_caltech20_rich_weight_average as previous
import audit_caltech20_rich_phasecode_weight_average46 as audit
text=inspect.getsource(previous.run);assert text.count('for d in (2,4,6):')==1;text=text.replace('for d in (2,4,6):','for d in (4,6):');ns=dict(previous.run.__globals__);ns.update(m=m,audit=audit);exec(compile(text,__file__+':unchanged_independent_weight_math_depth46','exec'),ns);run=ns['run']
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
