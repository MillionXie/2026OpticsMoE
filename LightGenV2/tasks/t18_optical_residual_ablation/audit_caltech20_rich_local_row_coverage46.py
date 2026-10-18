"""Reuse independently replayed single-row algebra; expand explicit class coverage only."""
import types,inspect
import audit_caltech20_rich_local_row_scalar_reference46 as previous
import caltech20_rich_local_row_coverage46 as m
for n,v in vars(previous).items():
 if not n.startswith('__') and n not in ('m','previous'):globals()[n]=v
for name,original in (('independent_plan',previous.independent_plan),('cpu',previous.cpu_kernel)):
 globals()[name]=types.FunctionType(original.__code__,globals(),name,original.__defaults__,original.__closure__)
text=inspect.getsource(previous.audit);assert text.count('assert len(stages)<=6')==1
text=text.replace('assert len(stages)<=6','assert len(stages)<=m.config()["maximum_classes"]')
exec(compile(text,__file__+':same_independent_math_explicit20class_budget','exec'),globals())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
