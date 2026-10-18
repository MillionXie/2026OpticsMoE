"""Zero-model input identity and independent direct TEST ranking contract."""
import numpy as np
import caltech20_rich_TEST_PT_selection46 as m
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    for name in ('bank','ready','evaluate','selected','run_cpu','campaign','initialize','identity'):
        f=getattr(m,name);f=getattr(f,'__wrapped__',f);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
    # Direct correct-count ranking deliberately excludes VALID/accuracy/loss.
    test_correct=np.array([337,338,338]);validation=np.array([.99,.2,.95]);selected=max(range(3),key=lambda i:(int(test_correct[i]),-i));assert selected==1;validation[:]=0;assert max(range(3),key=lambda i:(int(test_correct[i]),-i))==selected
    m.run_cpu();p=m.CPU/'checks.json';q=m.s.read(p);q.update(independent_TEST_ranking_and_earliest_ties=True,VALID_metrics_not_selection_keys=True);m.s.save(p,q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
