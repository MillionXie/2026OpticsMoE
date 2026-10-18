"""Independent sampler/gradient/full-trajectory checks for every gain-conditioned round."""
import types
import audit_caltech20_rich_local_row_coverage46 as previous
import caltech20_rich_row_gain_continuation46 as m
for n,v in vars(previous).items():
 if not n.startswith('__') and n not in ('m','previous'):globals()[n]=v
for name in ('independent_plan','cpu','audit'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
