"""Only source-dictionary comparison repaired; all numeric limits unchanged."""
import ast,inspect,argparse,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_robust_head_identity_repair as m
import check_caltech20_rich_robust_head as mathsource
import audit_caltech20_rich_robust_head as old

def source_identity_matches(actual,expected):
    return isinstance(actual,dict) and isinstance(expected,dict) and all(isinstance(k,str) and isinstance(v,str) and len(v)==64 for z in (actual,expected) for k,v in z.items()) and actual==expected

exec(compile(ast.parse(inspect.getsource(mathsource.numpy_objective)),__file__+':unchanged_numpy_math','exec'),globals())
text=inspect.getsource(old.audit);bad="m.s.digest(opt['sources'])==m.s.digest(meta['sources'])";assert bad in text;text=text.replace(bad,"source_identity_matches(opt['sources'],meta['sources'])")
exec(compile(text,__file__+':source_type_repair_only','exec'),globals())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--depth',required=True,type=int);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:
        assert m.s.read(m.CPU/'source_identity_checks.json')['passed'];audit(a.depth,a.phase)
    except BaseException as e:
        out=m.OUT/f'L{a.depth}_rho03';mark=out/(a.phase+'_audit_failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
