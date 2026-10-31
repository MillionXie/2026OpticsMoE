import pytest
import torch
from LightGenV2.tasks.t17_router_classification.alternating_readout import domain_route_loss, refit_existing_head, anchored_classification


def test_domain_targets_are_power_and_inference_labels_are_not_needed():
    power=torch.full((2,4),.25,requires_grad=True)
    loss=domain_route_loss(power,torch.tensor([0,1]));loss.backward()
    assert torch.isfinite(power.grad).all() and (power.grad != 0).all()
    assert power.grad[0,0] < power.grad[0,2] and power.grad[1,2] < power.grad[1,0]
    correct=torch.tensor([[.49,.49,.01,.01],[.01,.01,.49,.49]])
    assert abs(float(domain_route_loss(correct,torch.tensor([0,1])))) < 1e-6
    with pytest.raises(ValueError):domain_route_loss(power,torch.tensor([0,2]))


def test_bounded_head_block_detaches_features_decreases_same_objective_and_adds_no_parameters():
    torch.set_num_threads(4);torch.manual_seed(11)
    head=torch.nn.Linear(12,3,bias=False)
    features=torch.randn(60,12,requires_grad=True);labels=torch.arange(60)%3
    anchor=head.weight.detach().clone();parameters=list(head.parameters())
    record=refit_existing_head(head,features,labels,anchor,.001,.02,20)
    assert record['objective_after'] <= record['objective_before']
    assert record['internal_steps'] <= 20 and features.grad is None
    assert list(head.parameters())==parameters and head.bias is None
    assert head.weight.dtype==torch.float32 and torch.isfinite(head.weight).all()
    reference=anchored_classification(head(features),labels,head.weight,anchor,.001,.02)
    assert torch.isfinite(reference)


def test_head_penalty_is_in_canonical_weight_coordinates():
    logits=torch.randn(4,3);labels=torch.arange(4)%3
    w=torch.ones(3,6);anchor=torch.zeros_like(w)
    loss=anchored_classification(logits,labels,w,anchor,.1,0)
    assert torch.allclose(loss,torch.nn.functional.cross_entropy(logits,labels)+.9)
