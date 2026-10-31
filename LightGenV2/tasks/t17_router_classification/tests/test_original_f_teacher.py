import torch
import pytest
from LightGenV2.tasks.t17_router_classification.original_f_teacher import teacher_KL
from LightGenV2.tasks.t17_router_classification.alternating_readout import anchored_classification,refit_existing_head


def test_teacher_KL_detaches_only_training_teacher_and_changes_student_gradient():
    student=torch.randn(10,10,requires_grad=True);teacher=torch.randn(10,10,requires_grad=True)
    loss=teacher_KL(student,teacher,2.);loss.backward()
    assert teacher.grad is None and torch.isfinite(student.grad).all() and student.grad.norm()>0
    assert abs(float(teacher_KL(student.detach(),student.detach(),2.)))<1e-6
    with pytest.raises(ValueError):teacher_KL(student,teacher,0.)


def test_teacher_disabled_is_bitwise_same_objective():
    w=torch.randn(3,4);a=torch.zeros_like(w);x=torch.randn(8,3);y=torch.arange(8)%3
    old=anchored_classification(x,y,w,a,.001,.02)
    assert torch.equal(old,anchored_classification(x,y,w,a,.001,.02,torch.randn_like(x),0.,2.))


def test_head_refit_uses_same_KD_objective_with_no_teacher_or_feature_gradient():
    torch.set_num_threads(4);head=torch.nn.Linear(12,3,bias=False);x=torch.randn(60,12,requires_grad=True);t=torch.randn(60,3,requires_grad=True);y=torch.arange(60)%3
    result=refit_existing_head(head,x,y,head.weight.detach().clone(),.001,.02,15,t,.25,2.)
    assert x.grad is None and t.grad is None and result['objective_after']<=result['objective_before']
