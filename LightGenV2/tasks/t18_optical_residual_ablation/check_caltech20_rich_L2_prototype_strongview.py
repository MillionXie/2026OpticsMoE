"""Same strong-view proofs and real completed mild-L2 review barrier."""
import types
import caltech20_rich_L2_prototype_strongview as m
import check_caltech20_rich_prototype_strongview as original
import audit_caltech20_rich_L2_prototype_strongview as audit
def run():
    m.admit_completed_mild_L2();ns=dict(vars(original));ns.update(m=m,audit=audit);proof=types.FunctionType(original.run.__code__,ns,'same_matched_strong_view_CPU',original.run.__defaults__,original.run.__closure__);proof();q=m.s.read(m.CPU/'checks.json');q.update(completed_mild_L2_science=m.admit_completed_mild_L2(),no_repeat_of_mild_worker=True);m.s.save(m.CPU/'checks.json',q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
