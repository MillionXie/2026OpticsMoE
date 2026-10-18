import types,inspect
import check_caltech20_rich_margin_guard_readout46 as baseline
import caltech20_rich_margin_curriculum6 as m
for n,v in vars(baseline).items():
 if not n.startswith('__') and n!='m':globals()[n]=v
f=baseline.independent_plan;independent_plan=types.FunctionType(f.__code__,globals(),'independent_current_L6_plan',f.__defaults__,f.__closure__)
text=inspect.getsource(baseline.run);assert text.count('for d in (4,6):')==1;text=text.replace('for d in (4,6):','for d in (6,):');exec(compile(text,__file__+':same_independent_hinge_gradient_new_L6_parent','exec'),globals())
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json')
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
