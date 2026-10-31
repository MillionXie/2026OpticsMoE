import torch
from LightGenV2.tasks.t17_router_classification.joint_hidden_distillation import add_hidden_objective,hidden_payload


def test_disabled_hidden_is_identical_original_tensor():
    value=torch.tensor(2.,requires_grad=True)
    result,detail=add_hidden_objective(value,None,None)
    assert result is value and detail is None and hidden_payload(None)=={}


def test_targets_detached_and_student_features_receive_gradient():
    x=torch.randn(5,7,requires_grad=True);target=torch.randn(5,3,requires_grad=True);projection=torch.nn.Linear(7,3,bias=False)
    loss,_=add_hidden_objective(x.sum()*0,x,dict(target=target,projection=projection,weight=.2))
    loss.backward()
    assert x.grad.norm()>0 and target.grad is None and projection.weight.grad.norm()>0


def test_training_projection_payload_never_registers_into_student():
    student=torch.nn.Linear(7,10,bias=False);projection=torch.nn.Linear(7,3,bias=False);before=set(student.state_dict());aux=dict(projection=projection,optimizer=torch.optim.Adam(projection.parameters()))
    payload=dict(model=student.state_dict(),**hidden_payload(aux));loaded=torch.nn.Linear(7,10,bias=False);loaded.load_state_dict(payload['model'],strict=True)
    x=torch.randn(4,7)
    assert set(student.state_dict())==before and torch.equal(student(x),loaded(x))
    assert 'training_only_joint_hidden_projection' in payload


def test_soft_branch_rejects_TEST_GT_even_with_hidden_auxiliary():
    import pytest
    from LightGenV2.tasks.t17_router_classification.train_f_balanced_joint import balanced_objective
    x=torch.randn(4,7,requires_grad=True);projection=torch.nn.Linear(7,3,bias=False);head=torch.nn.Linear(7,10,bias=False)
    output=dict(raw_logits=head(x),ccd_features=x,dense_route_power=torch.randn(4,4).softmax(1));target=torch.randn(4,3,requires_grad=True);teacher=torch.randn(4,10,requires_grad=True);cfg=dict(teacher=dict(weight=.5,temperature=2.),alternating_head=dict(anchor_strength=.001),per_modality_balance=dict(weight=.0087))
    aux=dict(target=target,projection=projection,weight=.2)
    with pytest.raises(ValueError):balanced_objective(output,torch.arange(4),torch.arange(4)%2,teacher,head,head.weight.detach(),cfg,soft_only=True,hidden=aux)
    value,_=balanced_objective(output,None,torch.arange(4)%2,teacher,head,head.weight.detach(),cfg,soft_only=True,hidden=aux);value.backward()
    assert target.grad is None and teacher.grad is None and x.grad.norm()>0
