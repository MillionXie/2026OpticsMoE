"""Reuse full independent Hessian, selector and every-step state/metric checks."""
import types
import audit_caltech20_rich_L6_accuracy_NLL_bridge as previous
import caltech20_rich_L6_progressive_bridge as m
for n,v in vars(previous).items():
 if not n.startswith('__') and n not in ('m','previous'):globals()[n]=v
for name in ('better','independent_plan','raw_independent','independent_gradient','independent_update','cpu_kernel','cpu','audit'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
