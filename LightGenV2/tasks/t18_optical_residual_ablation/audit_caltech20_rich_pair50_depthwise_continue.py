"""Independent per-case CFG/prefix/target/KD-budget/Adam/first-min audit."""
import ast,inspect,argparse,hashlib,math,os
from pathlib import Path
import numpy as np,torch
import audit_caltech20_rich_pair50_balancedKD as frozen
import audit_caltech20_rich_pair50_alpha as independent
import caltech20_rich_pair50_depthwise_continue as m
tree=ast.parse(inspect.getsource(frozen.audit))
for node in ast.walk(tree):
    if isinstance(node,ast.Constant) and type(node.value) is int:node.value={50:70,70:90,9850:13790,13790:17730}.get(node.value,node.value)
replaced=0;loops=0
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='qq' for a in node.targets) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='np.exp':
        assert ast.unparse(node.value)=="np.exp(logits / cfg['temperature'] - (logits / cfg['temperature']).max(1, keepdims=True))";node.value=ast.parse("np.exp(logits / cfg['teacher_temperature'] - (logits / cfg['teacher_temperature']).max(1,keepdims=True))",mode='eval').body;replaced+=1
    if isinstance(node,ast.For) and isinstance(node.target,ast.Tuple) and ast.unparse(node.target)=='(key, (depth, rho))' and ast.unparse(node.iter)=='m.CASES.items()':node.body.insert(0,ast.parse('cfg=m.config(key)').body[0]);loops+=1
assert replaced==loops==1
exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),globals())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],default='fit');a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
