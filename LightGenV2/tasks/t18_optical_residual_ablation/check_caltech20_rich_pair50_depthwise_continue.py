"""Own per-depth update algebra and complete70 parent admission."""
import ast,inspect,argparse,os
from pathlib import Path
import numpy as np,torch
import check_caltech20_rich_pair50_balancedKD as frozen
import caltech20_rich_pair50_depthwise_continue as m
SYNTHETIC=m.HERE/'runs/smoke/caltech20_rich_pair50_depthwise90_synthetic_cpu_20261017'
for node in ast.parse(inspect.getsource(frozen)).body:
    if not isinstance(node,ast.FunctionDef):continue
    if node.name=='check':
        for x in ast.walk(node):
            if isinstance(x,ast.Constant) and type(x.value) is int:x.value={50:70,9850:13790}.get(x.value,x.value)
        count=0
        for x in ast.walk(node):
            if isinstance(x,ast.Compare) and ast.unparse(x.left)=="last['scheduler']['last_epoch']":assert len(x.comparators)==1 and x.comparators[0].value==70;x.comparators[0].value=20;count+=1
            if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='min' and isinstance(x.args[0],ast.Constant) and x.args[0].value==.1:x.args[0]=ast.parse("cfg['KD_scale_cap']",mode='eval').body
        assert count==1
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),globals())
original_synthetic_update=synthetic_update
def synthetic_update(cfg):
    guard=m.resources.proof()
    proofs={k:original_synthetic_update(m.config(k)) for k in m.CASES}
    teacher=np.array([[12.,-3.,2.],[-4.,8.,1.]],np.float64);q=np.exp(teacher/16-(teacher/16).max(1,keepdims=True));q/=q.sum(1,keepdims=True)
    st=torch.tensor([[.4,-.2,.1],[-.3,.5,.2]],dtype=torch.float64,requires_grad=True);qt=torch.tensor(q);loss=16*(qt*(qt.log()-torch.log_softmax(st/4,1))).sum(1).mean();g=torch.autograd.grad(loss,st)[0].numpy();expected=4*(torch.softmax(st.detach()/4,1).numpy()-q)/2;err=float(np.abs(g-expected).max());assert err<1e-12
    return dict(resource_guard=guard,synthetic_models=sum(x['synthetic_models'] for x in proofs.values()),synthetic_forward_calls=sum(x['synthetic_forward_calls'] for x in proofs.values()),synthetic_optimizer_steps=sum(x['synthetic_optimizer_steps'] for x in proofs.values()),micro1vs4_maxabs=max(x['micro1vs4_maxabs'] for x in proofs.values()),Adam_raw_maxabs=max(x['Adam_raw_maxabs'] for x in proofs.values()),Adam_moment_maxabs=max(x['Adam_moment_maxabs'] for x in proofs.values()),per_case_proofs=proofs,per_case_config={k:m.config(k) for k in m.CASES},teacher_temperature=16.,student_temperature=4.,independent_calibrated_target_gradient_maxabs=err)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--synthetic',action='store_true');a=p.parse_args()
    try:check_synthetic(a.out) if a.synthetic else check(a.out)
    except BaseException as e:
        mark=a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
