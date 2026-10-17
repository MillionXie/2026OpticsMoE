"""Saved Adam/EMA/TEST audit plus complete-macro relation-gradient budgets."""
import ast,inspect,argparse,math,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_relation as m
import audit_caltech20_rich_pair50_alpha as saved
import audit_caltech20_rich_robust as old
exec(compile(ast.parse(inspect.getsource(old.run)),__file__+':same_saved_science','exec'),globals())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:
        assert m.s.read(m.CPU/'relation_analytic.json')['passed']
        for row in m.s.read(m.OUT/a.key/'method_trace.json'):
            q=row['relation_info'];assert np.isfinite(list(q.values())).all() and q['scale']<=10 and q['feature_replay_maxabs']<=3e-6
            assert q['scale']*q['relation_feature_grad_norm']<=.3*q['classification_feature_grad_norm']+1e-7
        run(a.key)
    except BaseException as e:
        out=m.OUT/a.key;mark=out/'audit_failure.json'
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
