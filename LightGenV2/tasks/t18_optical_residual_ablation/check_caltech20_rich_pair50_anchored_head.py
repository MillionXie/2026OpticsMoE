"""Own factor10 anchor analytic/finite-difference/LBFGS proof plus parent receipts."""
import os,numpy as np,torch
import check_caltech20_rich_pair50_same_head as previous
import caltech20_rich_pair50_anchored_head as m
from audit_caltech20_rich_pair50_anchored_head import independent
exec(compile(previous.text,__file__+':main','exec'),globals())
if __name__=='__main__':
 try:main()
 except BaseException as e:
  mark=m.CPU/'failure.json' if m.CPU.exists() else m.CPU.with_name(m.CPU.name+'.failure.json')
  if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
