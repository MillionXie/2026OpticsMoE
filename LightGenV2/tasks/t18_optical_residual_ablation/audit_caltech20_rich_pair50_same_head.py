"""Own NumPy audit of inverse-count same-head objective and cached TEST."""
import argparse,ast,inspect,os
import numpy as np,torch
import audit_caltech20_same_linear_convex as frozen
import caltech20_rich_pair50_same_head as m
from audit_caltech20_original_equal60 import metrics,agree
softmax=frozen.softmax;logprob=frozen.logprob;predictions=frozen.predictions
text=inspect.getsource(frozen.independent);assert 'counts**-.5' in text;text=text.replace('counts**-.5','counts**-1.');exec(compile(text,__file__+':independent','exec'),globals())
t=inspect.getsource(frozen.audit)
changes={"parent/'fit/selected.pt'":"parent/'best.pt'","parent/'fit'/('selected_'+r+'.npz')":"parent/('selected_'+r+'.npz')","parent/'fit'/('selected_'+role+'.npz')":"parent/('selected_'+role+'.npz')","parent/'once_test/predictions.npz'":"m.parent.OUT/'once_tests'/key/'predictions.npz'","f['optical_actual_epochs']==80":"f['optical_actual_epochs']==70","q=softmax(tz/cfg['teacher_temperature'])":"q=softmax(tz/cfg['teacher_target_temperature'])","accepted=cv['accuracy']>pv['accuracy']+1e-12 or (abs(cv['accuracy']-pv['accuracy'])<1e-12 and cv['balanced_nll']<pv['balanced_nll']-1e-6)":"accepted=cv['balanced_nll']<pv['balanced_nll']"}
for a,b in changes.items():assert a in t,a;t=t.replace(a,b)
# No use of VAL to fit any scalar. Independently require exact scale1.
begin=t.index('        # Recompute convex positive scalar calibration independently.')
end=t.index("        assert abs(scale-f['calibration_inverse_temperature'])",begin)
t=t[:begin]+'        scale=1.\n'+t[end:]
t=t.replace("q=softmax(tz/cfg['teacher_target_temperature'])","q=softmax(tz/cfg['teacher_target_temperature'])")
exec(compile(t,__file__+':audit','exec'),globals())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
 try:audit(a.case,a.phase)
 except BaseException as e:
  mark=m.folder(a.case)/(a.phase+'_audit_failure.json')
  if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
