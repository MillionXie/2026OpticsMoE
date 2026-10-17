"""Regression proof for real Fortran-strided checkpoint head layout."""
import os
import numpy as np
import torch
import caltech20_same_linear_contiguous_repair as m
from audit_caltech20_same_linear_convex import independent

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'reproduce_Fortran_layout_then_prove_contiguous_equal_values_gradient_and_descent');torch.set_num_threads(4);parent,_=m.admit('L2_rho03');state=torch.load(parent/'fit/selected.pt',map_location='cpu',weights_only=False)['model'];anchor=np.r_[state['linear.weight'].double().numpy().T,state['linear.bias'].double().numpy()[None]];raw=torch.tensor(anchor);fixed=raw.contiguous();assert torch.equal(raw,fixed) and fixed.is_contiguous() and not raw.is_contiguous()
    parameter=torch.nn.Parameter(torch.tensor(np.asfortranarray(np.arange(6,dtype=float).reshape(2,3))));before=parameter.detach().clone();broken=torch.optim.LBFGS([parameter],max_iter=1);caught=False
    def badclosure():broken.zero_grad();loss=parameter.square().sum();loss.backward();return loss
    try:broken.step(badclosure)
    except RuntimeError as e:assert 'view size is not compatible' in str(e);caught=True
    assert caught and torch.equal(parameter,before)
    r=np.random.default_rng(17);X=r.normal(size=(57,7));y=np.r_[np.arange(20),r.integers(0,20,37)];q=m.old.probability(r.normal(size=(57,20)));valid=(np.arange(57)%4!=0).astype(float);a=r.normal(size=(7,20));t=torch.nn.Parameter(torch.tensor(np.asfortranarray(a)).contiguous());args=[torch.tensor(X),torch.tensor(y).long(),torch.tensor(q),torch.tensor(valid),torch.tensor(a)];cfg=m.config();loss,_=m.objective(t,*args,cfg);gradient=torch.autograd.grad(loss,t)[0];expected=independent(a,X,y,q,valid,a,cfg);err=float(np.abs(gradient.numpy()-expected[2]).max());assert err<1e-11 and gradient.is_contiguous()
    opt=torch.optim.LBFGS([t],max_iter=5,max_eval=10,history_size=3,line_search_fn='strong_wolfe');calls=[]
    def closure():opt.zero_grad();value,_=m.objective(t,*args,cfg);value.backward();calls.append(float(value.detach()));return value
    opt.step(closure);end,_=m.objective(t,*args,cfg);assert float(end.detach())<calls[0] and len(calls)==opt.state[t]['func_evals'];assert m.s.sha(m.FAILED/'failure.json')==m.FAILURE_SHA
    m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,preserved_failure_sha256=m.FAILURE_SHA,real_parent_head_values_unchanged=True,real_parent_stride=list(raw.stride()),contiguous_stride=list(fixed.stride()),original_failure_reproduced_without_update=caught,gradient_torch_numpy_maxabs=err,synthetic_LBFGS_iterations=opt.state[t]['n_iter'],synthetic_closure_evaluations=len(calls),optical_forward_calls=0,teacher_backbone_forward_calls=0,real_images_read=0,TEST_arrays_read=0),exclusive=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        mark=m.CPU/'failure.json' if m.CPU.exists() else m.CPU.with_name(m.CPU.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
