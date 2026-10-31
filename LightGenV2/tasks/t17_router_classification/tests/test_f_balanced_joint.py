import torch
import pytest
from LightGenV2.tasks.t17_router_classification.train_f_balanced_joint import balanced_objective
from LightGenV2.tasks.t17_router_classification.alternating_readout import anchored_classification


def inputs(weight):
    torch.manual_seed(17);head=torch.nn.Linear(7,10,bias=False).double();features=torch.randn(8,7,dtype=torch.float64);router=torch.randn(8,4,dtype=torch.float64,requires_grad=True);teacher=torch.randn(8,10,dtype=torch.float64,requires_grad=True)
    return dict(raw_logits=head(features),dense_route_power=router.softmax(1)),torch.arange(8),torch.arange(8)%2,teacher,head,head.weight.detach().clone(),dict(teacher=dict(weight=.5,temperature=2.),alternating_head=dict(anchor_strength=.001),label_smoothing=.02,per_modality_balance=dict(weight=weight)),router


def test_disabled_balance_is_bitwise_original_objective():
    out,y,d,t,h,a,cfg,_=inputs(0)
    value,_=balanced_objective(out,y,d,t,h,a,cfg)
    expected=anchored_classification(out['raw_logits'],y,h.weight,a,.001,.02,t,.5,2.)
    assert torch.equal(value,expected)


def test_TEST_hard_GT_rejected_and_teacher_detached():
    out,y,d,t,h,a,cfg,r=inputs(.0087)
    with pytest.raises(ValueError):balanced_objective(out,y,d,t,h,a,cfg,soft_only=True)
    loss,_=balanced_objective(out,None,d,t,h,a,cfg,soft_only=True)
    grads=torch.autograd.grad(loss,[r,t],allow_unused=True)
    assert grads[1] is None and grads[0].norm()>0 and torch.isfinite(grads[0]).all()
