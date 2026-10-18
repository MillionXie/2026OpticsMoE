"""Independent Hessian, sampler, every update and selection checks for both depths."""
import types
import audit_caltech20_rich_L6_progressive_bridge as previous
import caltech20_rich_head_targets26 as m
for n,v in vars(previous).items():
 if not n.startswith('__') and n not in ('m','previous'):globals()[n]=v
for name in ('better','independent_plan','raw_independent','independent_gradient','independent_update','cpu_kernel','cpu'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
# Reuse the independently replayed kernel, correcting its historical L6-only label.
f=previous.audit;constants=list(f.__code__.co_consts);hits=0
for i,c in enumerate(constants):
 if isinstance(c,tuple) and 'L2_L4_not_loaded_or_trained' in c:
  constants[i]=tuple('protected_original_versions_unchanged' if z=='L2_L4_not_loaded_or_trained' else z for z in c);hits+=1
assert hits==1
audit_kernel=types.FunctionType(f.__code__.replace(co_consts=tuple(constants)),globals(),'independent_depth_specific_audit',f.__defaults__,f.__closure__)
def audit():
 audit_kernel();p=m.OUT/'science/audit.json';q=m.s.read(p);q.update(active_depth=m.DEPTH,old_L2_L6_versions_preserved=True,original_L4_and_controls_not_trained=True);m.s.save(p,q)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
