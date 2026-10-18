"""Original CPU/state proofs plus independent semantic feature-loss derivative."""
import types
import numpy as np
import torch
import caltech20_rich_semantic_alignment as m
import check_caltech20_rich_targeted_dualview as previous
import audit_caltech20_rich_semantic_alignment as audit
def run():
    ns=dict(vars(previous));ns.update(m=m,audit=audit);f=types.FunctionType(previous.run.__code__,ns,'same_CPU_math_and_retained_state_proof',previous.run.__defaults__,previous.run.__closure__);f()
    rg=np.random.default_rng(17);x=torch.tensor(rg.normal(size=(5,7)),requires_grad=True);t=torch.tensor(rg.normal(size=(5,7)));mass=torch.tensor([1.,.3,0.,.7,1.]);v=m.feature_loss(x,t,mass);g=torch.autograd.grad(v,x)[0].numpy();expected=mass.numpy()[:,None]*(x.detach().numpy()-t.numpy())/5;assert np.max(np.abs(g-expected))<1e-12
    checks={}
    for d in (4,6):
        features,scale,sha=m.target_data(d);assert features.shape==(3141,256) and 0<scale<=.1 and np.isfinite(features).all();checks[str(d)]=dict(target_sha256=sha,scale=scale)
    q=m.s.read(m.CPU/'checks.json');q.update(semantic_loss_derivative_passed=True,TRAIN_targets=checks,new_teacher_forwards=0,new_inference_parameters=0);m.s.save(m.CPU/'checks.json',q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
