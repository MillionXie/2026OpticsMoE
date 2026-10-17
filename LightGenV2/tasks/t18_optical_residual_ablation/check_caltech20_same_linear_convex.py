"""Independent analytic gradient proof, synthetic data only, no optical network."""
import os
import numpy as np
import torch
import caltech20_same_linear_convex as m
from audit_caltech20_same_linear_convex import independent

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.CPU.exists(); m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'synthetic_convex_CE_KD_anchor_gradient_and_LBFGS'); torch.set_num_threads(4)
    r=np.random.default_rng(17); X=r.normal(size=(57,7)); X[:,-1]=1.; y=np.r_[np.arange(20),r.integers(0,20,37)]; q=m.old.probability(r.normal(size=(57,20))); valid=(np.arange(57)%4!=0).astype(float); anchor=r.normal(scale=.1,size=(7,20)); theta=anchor+r.normal(scale=.02,size=anchor.shape); cfg=m.config()
    t=torch.tensor(theta,requires_grad=True); args=[torch.tensor(X),torch.tensor(y).long(),torch.tensor(q),torch.tensor(valid),torch.tensor(anchor)]
    loss,parts=m.objective(t,*args,cfg); grad=torch.autograd.grad(loss,t)[0].numpy(); el,ep,eg=independent(theta,X,y,q,valid,anchor,cfg)
    error=float(np.abs(grad-eg).max()); assert error<1e-11 and abs(float(loss.detach())-el)<1e-11 and np.max(np.abs(ep-[float(x.detach()) for x in parts]))<1e-11
    eps=1e-6; plus=theta.copy(); minus=theta.copy(); plus[2,7]+=eps; minus[2,7]-=eps; fd=(independent(plus,X,y,q,valid,anchor,cfg)[0]-independent(minus,X,y,q,valid,anchor,cfg)[0])/(2*eps); assert abs(fd-eg[2,7])<1e-8
    parameter=torch.nn.Parameter(torch.tensor(theta)); opt=torch.optim.LBFGS([parameter],max_iter=5,max_eval=10,history_size=3,line_search_fn='strong_wolfe'); calls=[]
    def closure():
        opt.zero_grad(); value,_=m.objective(parameter,*args,cfg); value.backward(); calls.append(float(value.detach())); return value
    opt.step(closure); final,_=m.objective(parameter,*args,cfg); assert float(final.detach())<calls[0] and len(calls)==opt.state[parameter]['func_evals']
    m.CPU.mkdir(); m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,gradient_torch_numpy_maxabs=error,finite_difference_maxabs=abs(fd-eg[2,7]),synthetic_LBFGS_iterations=opt.state[parameter]['n_iter'],synthetic_closure_evaluations=len(calls),synthetic_objective_initial=calls[0],synthetic_objective_final=float(final.detach()),real_models=0,optical_forward_calls=0,teacher_backbone_forward_calls=0,real_images_read=0,TEST_arrays_read=0),exclusive=True)

if __name__=='__main__':
    try: main()
    except BaseException as e:
        mark=m.CPU/'failure.json' if m.CPU.exists() else m.CPU.with_name(m.CPU.name+'.failure.json')
        if not mark.exists(): m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
