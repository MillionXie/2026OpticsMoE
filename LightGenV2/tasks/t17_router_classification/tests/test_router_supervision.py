import numpy as np
import torch
from LightGenV2.tasks.t17_router_classification.train_boundary_development import class_pair_route_targets,dense_route_supervision
from LightGenV2.tasks.t17_router_classification.model_four import FourRouterClassification


def test_pair_targets_are_deterministic_and_balance_original_train_prior():
    counts=[873,411,288,446,240,517,931,596,724]
    target=class_pair_route_targets(counts)
    assert np.array_equal(target,class_pair_route_targets(counts))
    assert np.allclose(target.sum(1),1) and np.all((target>.02).sum(1)==2)
    expected=(np.asarray(counts)[:,None]*target).sum(0)/sum(counts)
    assert expected.max()-expected.min()<.1


def test_dense_auxiliary_ignores_test_ground_truth_and_revives_low_port_gradient():
    logits=torch.tensor([[2.,1.,0.,-10.],[2.,1.,0.,-10.]],requires_grad=True)
    teacher=torch.tensor([[.9,.1],[.01,.99]])
    table=torch.tensor([[.49,.49,.01,.01],[.01,.01,.49,.49]])
    original=torch.tensor([True,False]);a=torch.tensor([0,0]);b=torch.tensor([0,1])
    x=dense_route_supervision(logits.softmax(1),a,original,teacher,table,.8)
    y=dense_route_supervision(logits.softmax(1),b,original,teacher,table,.8)
    assert torch.equal(x,y);x.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[1,3]<-.1


def test_auxiliary_is_only_a_loss_on_existing_optical_router():
    model=FourRouterClassification('optical','ccd_grid',num_classes=9)
    x=torch.rand(1,224,224);out=model(x,return_dense_route=True)
    target=torch.as_tensor(class_pair_route_targets([1]*9));labels=torch.tensor([2])
    loss=dense_route_supervision(out['dense_route_power'],labels,torch.tensor([True]),torch.nn.functional.one_hot(labels,9).float(),target,.8)
    loss.backward();assert model.router_phase.grad is not None and torch.isfinite(model.router_phase.grad).all() and model.router_phase.grad.abs().sum()>0
    assert set(dict(model.named_parameters()))=={'first_phase','global_phase','router_phase'}
    with torch.no_grad():assert torch.equal(out['probabilities'],model(x)['probabilities'])
