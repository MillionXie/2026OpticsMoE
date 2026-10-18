"""Independent physical-fusion value/derivative and five-card guard proof."""
import numpy as np
import torch
import caltech20_rich_branch_investigation as m
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'branch_diagnosis_math_CPU')
    for name in ('run','gradients','probe','summarize','identity'):
        f=getattr(m,name);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
    assert not unresolved(m.ProbeMoE.forward.__code__,m.ProbeMoE.forward.__globals__)
    rg=np.random.default_rng(17);a=rg.uniform(.1,1,(2,4,4));b=rg.uniform(.1,1,(2,4,4));w=rg.normal(size=a.shape)
    def objective(x):
        e=x[:32].reshape(2,4,4);o=x[32:].reshape(2,4,4);re=np.sqrt(np.mean(e*e,axis=(-2,-1),keepdims=True));ro=np.sqrt(np.mean(o*o,axis=(-2,-1),keepdims=True));z=.4*e/re+.6*o/ro;return float((w*re*z/np.sqrt(np.mean(z*z,axis=(-2,-1),keepdims=True))).sum())
    x=np.r_[a.ravel(),b.ravel()];t=torch.tensor(x,requires_grad=True);e=t[:32].reshape(2,4,4);o=t[32:].reshape(2,4,4);re=m.rms(e);ro=m.rms(o);mixed=.4*e/re+.6*o/ro;value=(torch.tensor(w)*re*mixed/m.rms(mixed)).sum();g=torch.autograd.grad(value,t)[0].numpy();numeric=[]
    for i in range(len(x)):
        plus=x.copy();minus=x.copy();plus[i]+=1e-6;minus[i]-=1e-6;numeric.append((objective(plus)-objective(minus))/2e-6)
    error=float(np.max(np.abs(g-numeric)));assert error<=1e-7 and abs(float(value)-objective(x))<1e-12
    for d in (2,4,6):
        q,pt=m.parents.parent(d);model=m.ProbeMoE(d);model.load_state_dict(m.remap(pt['model'],q['alpha_low']));assert sum(p.numel() for p in model.linear.parameters())==5140 and sum(p.numel() for n,p in model.named_parameters() if n.startswith('electronic.'))==9*d
    m.prior.controls();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,independent_fullRMS_derivative_maxabs=error,resource_guard=m.resources.proof(),actual_namespace_complete=True,real_model_forward_calls=0,real_optimizer_steps=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
