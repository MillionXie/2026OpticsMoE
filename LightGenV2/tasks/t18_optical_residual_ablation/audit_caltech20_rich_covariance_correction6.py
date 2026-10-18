import types
import audit_caltech20_rich_covariance_correction4 as baseline
import caltech20_rich_covariance_correction6 as m
from check_caltech20_rich_covariance_correction6 import independent_statistics
for n,v in vars(baseline).items():
 if not n.startswith('__') and n not in ('m','independent_statistics'):globals()[n]=v
f=baseline.run;run=types.FunctionType(f.__code__,globals(),'same_independent_covariance_science_L6',f.__defaults__,f.__closure__)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
