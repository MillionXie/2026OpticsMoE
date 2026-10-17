"""Independent NumPy factor10-anchor gradient and cached TEST audit."""
import ast,inspect,argparse,os
import numpy as np,torch
import audit_caltech20_rich_pair50_same_head as previous
import caltech20_rich_pair50_anchored_head as m
from audit_caltech20_original_equal60 import metrics,agree
softmax=previous.softmax;logprob=previous.logprob;predictions=previous.predictions
text=previous.text
assert 'penalty=float(np.square(theta-anchor).sum()/(2*n))' in text and '+(theta-anchor)/n' in text
text=text.replace('penalty=float(np.square(theta-anchor).sum()/(2*n))',"penalty=float(cfg['anchor_scale']*np.square(theta-anchor).sum()/(2*n))").replace('+(theta-anchor)/n',"+cfg['anchor_scale']*(theta-anchor)/n")
exec(compile(text,__file__+':independent','exec'),globals());exec(compile(previous.t,__file__+':audit','exec'),globals())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
 try:audit(a.case,a.phase)
 except BaseException as e:
  mark=m.folder(a.case)/(a.phase+'_audit_failure.json')
  if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
