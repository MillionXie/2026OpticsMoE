"""Same independent tail science, explicit phase9/prefix3/modelL4 identity."""
import types,inspect
import audit_caltech20_rich_tail_separation6 as baseline
import caltech20_rich_tail_separation4 as m
from check_caltech20_rich_tail_separation4 import independent_plan
for n,v in vars(baseline).items():
 if not n.startswith('__') and n not in ('m','independent_plan'):globals()[n]=v
text=inspect.getsource(baseline.run);assert "n=='fusion_logits.5'" in text;text=text.replace("n=='fusion_logits.5'","n=='fusion_logits.3'").replace('frozen_prefix5_bytes_unchanged','frozen_prefix3_bytes_unchanged');exec(compile(text,__file__+':same_saved_science_bound_to_actual_L4_tail','exec'),globals())
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
