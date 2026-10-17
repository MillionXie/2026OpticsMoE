"""Same saved audit, bound to independently frozen precise-coordinate run."""
import ast,inspect,argparse,math,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_robust_precise as m
import audit_caltech20_rich_pair50_alpha as saved
import audit_caltech20_rich_robust as old
exec(compile(ast.parse(inspect.getsource(old.run)),__file__+':same_saved_science','exec'),globals())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
