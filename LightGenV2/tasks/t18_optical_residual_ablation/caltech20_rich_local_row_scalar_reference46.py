"""Exact scalar-reference repair; original failed CPU source remains frozen."""
import types
import caltech20_rich_local_row_repair46 as previous
for n,v in vars(previous).items():
 if not n.startswith('__') and n!='previous':globals()[n]=v
FILES=('caltech20_rich_local_row_scalar_reference46.py',previous.FILES[1],'audit_caltech20_rich_local_row_scalar_reference46.py','CALTECH20_RICH_LOCAL_ROW_SCALAR_REFERENCE46_PROTOCOL_20261018.md')
def config():
 c=previous.config();c['run']+='_F64_scalar_reference';return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=previous.CPU.with_name(previous.CPU.name+'_F64_scalar_reference')
def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
for name in ('identity','claim','incumbent','stage_plan','prepared','gradient','update','ready','fit'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
if __name__=='__main__':
 try:fit()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
