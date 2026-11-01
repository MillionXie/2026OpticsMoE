import pytest
import torch
from LightGenV2.tasks.t17_router_classification.train_f_four_metric import four_metric_key,four_group_objective


def metric(a,b):return dict(domains=dict(RGB=dict(accuracy=a),SAR=dict(accuracy=b)),cross_entropy=1.)


def test_selection_requires_same_PT_all_four_metrics():
    assert four_metric_key(metric(.66,.65),metric(.65,.66))>four_metric_key(metric(.52,.48),metric(.776,.718))
    assert four_metric_key(metric(.70,.66),metric(.698,.578))[0]==.578


def test_duplicating_one_group_cannot_change_group_objective():
    torch.manual_seed(17);logits=torch.randn(8,3,dtype=torch.float64,requires_grad=True);labels=torch.arange(8)%3;groups=torch.arange(8)//2
    a=four_group_objective(logits,labels,groups);index=torch.tensor([0,1,0,1,0,1,2,3,4,5,6,7]);b=four_group_objective(logits[index],labels[index],groups[index])
    assert torch.allclose(a,b,atol=1e-12,rtol=0)


def test_student_TEST_GT_rejected_and_teacher_detached():
    logits=torch.randn(8,3,requires_grad=True);targets=torch.randn(8,3,requires_grad=True);groups=torch.arange(8)//2;labels=torch.arange(8)%3
    with pytest.raises(ValueError):four_group_objective(logits,labels,groups,targets)
    labels[groups>=2]=-1;four_group_objective(logits,labels,groups,targets).backward()
    assert targets.grad is None and torch.isfinite(logits.grad).all()


def test_weak_group_receives_gradient_without_ignoring_other_groups():
    logits=torch.tensor([[4.,0.],[4.,0.],[4.,0.],[0.,4.]],requires_grad=True);labels=torch.zeros(4,dtype=torch.long);groups=torch.arange(4)
    four_group_objective(logits,labels,groups).backward()
    assert logits.grad[3].norm()>logits.grad[0].norm()>0
