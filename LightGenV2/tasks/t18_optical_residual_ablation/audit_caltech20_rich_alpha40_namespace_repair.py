"""Unchanged independent saved-math audit in explicit complete namespace."""
import ast,inspect,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_alpha40_namespace_repair as m
import audit_caltech20_rich_alpha40_rebalance as original
import audit_caltech20_rich_pair50_alpha as saved
exec(compile(ast.parse(inspect.getsource(original.run)),__file__+':same_saved_science','exec'),globals())
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:m.failed_preflight();run(a.key);m.failed_preflight()
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
