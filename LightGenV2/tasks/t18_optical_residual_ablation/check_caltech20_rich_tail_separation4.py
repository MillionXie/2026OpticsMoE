"""Same independent sampler/cosine derivative checks, explicit L4 scope."""
import types,inspect
import check_caltech20_rich_tail_separation6 as baseline
import caltech20_rich_tail_separation4 as m
for n,v in vars(baseline).items():
 if not n.startswith('__') and n!='m':globals()[n]=v
f=baseline.independent_plan;independent_plan=types.FunctionType(f.__code__,globals(),'independent_L4_plan',f.__defaults__,f.__closure__)
text=inspect.getsource(baseline.run);assert 'len(bad)==78 and len(set(items))==217' in text;text=text.replace('len(bad)==78 and len(set(items))==217','len(bad)==72 and len(set(items))==201');assert "model.stage_slices[5]==(14,15) and model.stages==['expert','global']*3" in text;text=text.replace("model.stage_slices[5]==(14,15) and model.stages==['expert','global']*3","model.stage_slices[3]==(9,10) and model.stages==['expert','global']*2");exec(compile(text,__file__+':same_checks_with_actual_L4_8248_parent','exec'),globals())
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json')
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
