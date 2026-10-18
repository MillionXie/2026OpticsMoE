import types,inspect
import audit_caltech20_rich_margin_guard_readout46 as baseline
import caltech20_rich_margin_curriculum6 as m
from check_caltech20_rich_margin_curriculum6 import independent_plan
for n,v in vars(baseline).items():
 if not n.startswith('__') and n not in ('m','independent_plan'):globals()[n]=v
text=inspect.getsource(baseline.run);assert text.count('for d in (4,6):')==1;text=text.replace('for d in (4,6):','for d in (6,):');changes={"vw=.9*vw+g*clip;vb=.9*vb+h*clip;w-=.001*vw;b-=.001*vb":"vw=.9*vw+g*clip;vb=.9*vb+h*clip;w-=m.config()['lr']*vw;b-=m.config()['lr']*vb","scale=min(1.,.1/max(":"scale=min(1.,m.config()['delta_radius']/max("}
for a,b in changes.items():assert a in text,a;text=text.replace(a,b)
exec(compile(text,__file__+':complete_independent_L6_current_round_hinge_trajectory','exec'),globals())
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
