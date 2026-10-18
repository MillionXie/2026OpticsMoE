"""Explicit final EG scope and frozen head, independent sampler/cosine checks."""
import types,inspect
import check_caltech20_rich_tail_separation6 as baseline
import caltech20_rich_lastpair_fixed_head6 as m
for n,v in vars(baseline).items():
 if not n.startswith('__') and n!='m':globals()[n]=v
f=baseline.independent_plan;independent_plan=types.FunctionType(f.__code__,globals(),'same_TRAIN_sampler',f.__defaults__,f.__closure__)
text=inspect.getsource(baseline.run);assert 'expected_beta=min(2.,.2*' in text;text=text.replace('expected_beta=min(2.,.2*',"expected_beta=min(m.config()['triplet_max_beta'],.2*");site="model.stage_slices[5]==(14,15) and model.stages==['expert','global']*3";assert site in text;text=text.replace(site,site+" and model.stage_slices[4]==(10,14) and not model.linear.weight.requires_grad and not model.linear.bias.requires_grad");exec(compile(text,__file__+':independent_fixed_head_last_EG_scope','exec'),globals())
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json')
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
