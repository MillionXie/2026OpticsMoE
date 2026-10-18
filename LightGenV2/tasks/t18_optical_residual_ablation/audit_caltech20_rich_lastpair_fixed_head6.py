"""Independent tail science with both final alpha projections and frozen head."""
import types,inspect
import audit_caltech20_rich_tail_separation6 as baseline
import caltech20_rich_lastpair_fixed_head6 as m
from check_caltech20_rich_lastpair_fixed_head6 import independent_plan
for n,v in vars(baseline).items():
 if not n.startswith('__') and n not in ('m','independent_plan'):globals()[n]=v
text=inspect.getsource(baseline.run);assert "n=='fusion_logits.5'" in text;text=text.replace("n=='fusion_logits.5'","n.startswith('fusion_logits.')").replace('frozen_prefix5_bytes_unchanged','frozen_prefix4_and_readout_bytes_unchanged');exec(compile(text,__file__+':same_independent_boundary_for_final_EG_nine_parameters','exec'),globals())
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
