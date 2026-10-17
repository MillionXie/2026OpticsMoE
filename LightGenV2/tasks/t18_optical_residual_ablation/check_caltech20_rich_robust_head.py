"""Independent logistic/tempered-KD/anchor derivative, actual frozen paths."""
import os
import numpy as np
import torch
import caltech20_rich_robust_head as m

def numpy_objective(theta,X,y,q,valid,anchor,lam):
    cfg=m.config();n=len(y);z=X@theta;soft=np.eye(20)[y]*(1-cfg['label_smoothing'])+cfg['label_smoothing']/20
    counts=np.asarray(m.robust.base.manifest()['supports']['train'],dtype=float);weights=counts.sum()/(20*counts);soft=soft*weights[None]
    def logsoft(a):a=a-a.max(1,keepdims=True);return a-np.log(np.exp(a).sum(1,keepdims=True))
    logs=logsoft(z);T=cfg['temperature'];kl_logs=logsoft(z/T);p=np.exp(logs);pt=np.exp(kl_logs)
    ce=-(soft*logs).sum()/n;kl=T*T*(valid[:,None]*q*(np.log(np.maximum(q,1e-12))-kl_logs)).sum()/n;penalty=lam*np.square(theta-anchor).sum()/(2*n)
    grad=X.T@(.98*(p*soft.sum(1,keepdims=True)-soft)/n+.02*T*valid[:,None]*(pt*q.sum(1,keepdims=True)-q)/n)+lam*(theta-anchor)/n
    return .98*ce+.02*kl+penalty,grad

def run():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'postrobustness_samehead_CPU_math_and_paths');torch.set_num_threads(4);parents=m.ready();m.controls()
    rg=np.random.default_rng(17);X=rg.normal(size=(9,5));y=np.arange(9)%20;q=rg.uniform(size=(9,20));q/=q.sum(1,keepdims=True);valid=np.ones(9);anchor=rg.normal(size=(5,20));theta=rg.normal(size=(5,20))*.2
    p=torch.tensor(theta,requires_grad=True);loss,_=m.objective(p,torch.tensor(X),torch.tensor(y),torch.tensor(q),torch.tensor(valid),torch.tensor(anchor),10.);g=torch.autograd.grad(loss,p)[0].numpy();expected,eg=numpy_objective(theta,X,y,q,valid,anchor,10.)
    assert abs(float(loss)-expected)<1e-12 and np.max(np.abs(g-eg))<1e-12
    checks={}
    for depth in (2,4,6):
        choice,mod,pt=m.get_parent(depth);model=mod.RobustMoE(depth);model.load_state_dict(pt['ema'],strict=True);assert model.linear.weight.numel()+model.linear.bias.numel()==5140
        pred=m.load_np(m.Path(choice['arm'])/'test_epoch10.npz');assert pred['ids'].tolist()==m.robust.base.manifest()['split_ids']['test'];err=float(np.max(np.abs(m.probabilities(pt['ema'],pred['features'])-pred['probabilities'])));assert err<=3e-6
        checks[str(depth)]=dict(parent_last_sha256=choice['last_sha256'],parent_ema_digest=choice['ema_digest'],linear_cached_probability_maxabs=err,existing_TEST_arrays_only_no_new_inference=True)
    m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,objective_maxabs=abs(float(loss)-expected),analytic_gradient_maxabs=float(np.max(np.abs(g-eg))),real_model_forward_calls=0,TEST_images_read=0,optimizer_steps=0,source_parents=parents),exclusive=True)

if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
