"""Independent double-precision objective gradient and complete namespace check."""
import numpy as np
import torch
import caltech20_rich_alpha40_samehead as m
import audit_caltech20_rich_alpha40_samehead as audit
from check_caltech20_rich_robust_head import numpy_objective
from check_caltech20_rich_alpha40_fresh246 import unresolved
if __name__=='__main__':
    try:
        assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'same_head_encoder50_CPU')
        for name in ('get_parent','extract','fit','test','campaign','one','identity','objective','ready'):
            f=getattr(m,name);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
        assert not unresolved(audit.audit.__code__,audit.audit.__globals__)
        rng=np.random.default_rng(17);X=rng.normal(size=(6,257));X[:,-1]=1;y=np.arange(6)%20;q=rng.uniform(.01,1,(6,20));q/=q.sum(1,keepdims=True);valid=np.ones(6);anchor=rng.normal(0,.01,(257,20));t=torch.tensor(anchor+.003,requires_grad=True)
        value,_=m.objective(t,torch.tensor(X),torch.tensor(y),torch.tensor(q),torch.tensor(valid),torch.tensor(anchor),10);g=torch.autograd.grad(value,t)[0].numpy();nv,ng=numpy_objective(t.detach().numpy(),X,y,q,valid,anchor,10);err=float(np.abs(ng-g).max());assert err<=1e-12 and abs(nv-float(value))<=1e-12
        assert m.config()['label_smoothing']==.02 and m.config()['temperature']==4 and m.config()['anchor_by_depth']=={'2':1,'4':10,'6':10}
        for family,mod in m.PHASES.items():
            for depth in (2,4,6):
                model=mod.Alpha40MoE(depth);assert sum(p.numel() for p in model.linear.parameters())==5140 and len(list(model.parameters()))=={2:12,4:21,6:30}[depth]
        m.robust.controls();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,analytic_gradient_maxabs=err,full_actual_namespaces_resolved=True,real_model_forward_calls=0,TEST_images_read=0,real_model_optimizer_steps=0,new_inference_parameters=0),exclusive=True)
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
