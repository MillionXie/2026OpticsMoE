"""Independent Procrustes optimality and feature CE derivative proof."""
import numpy as np
import torch
import caltech20_rich_teacher_alignment as m
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'TRAIN_teacher_alignment_CPU')
    for name in ('teacher','student','prepare','audit','identity'):
        f=getattr(m,name);assert not unresolved(f.__code__,f.__globals__),(name,unresolved(f.__code__,f.__globals__))
    rg=np.random.default_rng(17);a=rg.normal(size=(19,4));b=rg.normal(size=(19,4));u,sv,vt=np.linalg.svd(a.T@b);r=u@vt;assert np.max(np.abs(r.T@r-np.eye(4)))<1e-12 and np.linalg.norm(a@r-b)<=np.linalg.norm(a-b)+1e-12
    x=torch.tensor(rg.normal(size=(7,4)),requires_grad=True);w=torch.tensor(rg.normal(size=(3,4)));labels=torch.tensor(np.arange(7)%3);loss=torch.nn.functional.cross_entropy(x@w.T,labels);g=torch.autograd.grad(loss,x)[0].numpy();p=m.softmax(x.detach().numpy()@w.numpy().T);p[np.arange(7),labels.numpy()]-=1;error=float(np.abs(g-p@w.numpy()/7).max());assert error<1e-12
    m.prior.old.controls();m.identity();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,analytic_feature_CE_gradient_maxabs=error,Procrustes_optimality=True,real_model_forward_calls=0,real_images_read=0,TEST_images_read=0,optimizer_steps=0),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
