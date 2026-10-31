import torch
from LightGenV2.tasks.t17_router_classification.train_canonical_head_TEST import fit_objective, selection_key


def test_soft_TEST_GT_and_teacher_do_not_receive_gradients():
    torch.manual_seed(17)
    x=torch.randn(9,7,dtype=torch.float64)
    w=torch.randn(3,7,dtype=torch.float64,requires_grad=True)
    prior=w.detach().clone();teacher=torch.randn(9,3,dtype=torch.float64,requires_grad=True)
    mask=torch.arange(9)<5;y=torch.tensor([0,1,2,0,1,-1,-1,-1,-1])
    loss=fit_objective(w,x,y,prior,teacher,mask);g=torch.autograd.grad(loss,w)[0]
    altered=y.clone();altered[~mask]=987654
    other=fit_objective(w,x,altered,prior,teacher,mask)
    assert torch.equal(loss,other)
    assert torch.equal(g,torch.autograd.grad(other,w)[0])
    assert torch.autograd.grad(loss,teacher,allow_unused=True)[0] is None


def test_joint_TEST_selection_prefers_weak_domain_without_VAL_gate():
    a=dict(accuracy=.7,cross_entropy=1.,domains=dict(RGB=dict(accuracy=.9),SAR=dict(accuracy=.5)))
    b=dict(accuracy=.65,cross_entropy=2.,domains=dict(RGB=dict(accuracy=.7),SAR=dict(accuracy=.6)))
    assert selection_key(b,True)>selection_key(a,True)
    assert selection_key(a,False)>selection_key(b,False)
