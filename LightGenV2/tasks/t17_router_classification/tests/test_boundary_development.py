import torch
from LightGenV2.tasks.t17_router_classification.train_boundary_development import boundary_objective,initialize_main_phases
from LightGenV2.tasks.t17_router_classification.model_four import FourRouterClassification


def test_test_ground_truth_does_not_enter_boundary_loss():
    p=torch.randn(4,8).log_softmax(1).requires_grad_();teacher=torch.randn(4,8).softmax(1);teacher[2:]=torch.tensor([.99]+[.01/7]*7)
    mask=torch.tensor([True,True,False,False]);a=torch.tensor([1,2,3,4]);b=torch.tensor([1,2,6,7])
    x=boundary_objective(p,a,mask,teacher,.1,.8,.1);y=boundary_objective(p,b,mask,teacher,.1,.8,.1)
    assert torch.equal(x,y);x.backward();assert torch.isfinite(p.grad).all() and p.grad[2:].abs().sum()>0


def test_diffuse_initialization_changes_only_main_phases_and_adds_no_parameters():
    model=FourRouterClassification('optical','ccd_grid',num_classes=8);before=model.router_phase.detach().clone()
    initialize_main_phases(model,'diffuse_uniform_phase',17)
    assert torch.equal(before,model.router_phase)
    assert set(dict(model.named_parameters()))=={'first_phase','global_phase','router_phase'}
    assert sum(p.numel() for p in model.parameters())==479364
    assert model.first_phase.std()>1 and torch.isfinite(model.global_phase).all()
