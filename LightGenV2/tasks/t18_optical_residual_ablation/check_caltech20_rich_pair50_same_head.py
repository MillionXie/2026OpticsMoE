"""Own analytic gradient/finite difference and contiguous LBFGS synthetic proof."""
import ast,inspect,os
import numpy as np,torch
import check_caltech20_same_linear_convex as frozen
import caltech20_rich_pair50_same_head as m
from audit_caltech20_rich_pair50_same_head import independent
text=inspect.getsource(frozen.main);text=text.replace('torch.nn.Parameter(torch.tensor(theta))','torch.nn.Parameter(torch.tensor(theta).contiguous())')
text=text.replace('    r=np.random.default_rng(17);',"    m.driver.resources.proof();[m.admit(key) for key in m.config()['cases']];\n    r=np.random.default_rng(17);")
exec(compile(text,__file__+':main','exec'),globals())
if __name__=='__main__':
 try:main()
 except BaseException as e:
  mark=m.CPU/'failure.json' if m.CPU.exists() else m.CPU.with_name(m.CPU.name+'.failure.json')
  if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
