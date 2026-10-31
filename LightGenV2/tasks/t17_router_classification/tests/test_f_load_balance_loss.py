import torch
import pytest
from LightGenV2.tasks.t17_router_classification.f_load_balance import modality_mean_power_KL


def test_marginal_balance_does_not_force_each_image_uniform():
    power=torch.tensor([[.49,.49,.01,.01],[.01,.01,.49,.49]]*2)
    domains=torch.tensor([0,0,1,1])
    assert modality_mean_power_KL(power,domains).item()==pytest.approx(0,abs=1e-7)


def test_joint_average_cannot_hide_opposite_modality_collapse():
    power=torch.tensor([[.49,.49,.01,.01]]*2+[[.01,.01,.49,.49]]*2)
    assert modality_mean_power_KL(power,torch.tensor([0,0,1,1]))>.9


def test_gradient_discourages_overloaded_ports_without_labels():
    logits=torch.tensor([[2.,2.,0.,-3.]]*4,requires_grad=True)
    loss=modality_mean_power_KL(logits.softmax(1),torch.tensor([0,1,0,1]));loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[:,:2]>0).all() and (logits.grad[:,3]<0).all()


def test_unknown_modality_rejected():
    with pytest.raises(ValueError):modality_mean_power_KL(torch.full((2,4),.25),torch.tensor([0,2]))


def test_population_marginal_chain_rule_matches_unequal_chunks():
    torch.manual_seed(17);logits=torch.randn(9,4,dtype=torch.float64,requires_grad=True);domains=torch.tensor([0]*5+[1]*4)
    power=logits.softmax(1);direct=torch.autograd.grad(modality_mean_power_KL(power,domains),logits,retain_graph=True)[0]
    surrogate=0.
    for d in (0,1):
        total=(domains==d).sum();derivative=-.25/power[domains==d].mean(0).detach()
        for start,end in [(0,2),(2,6),(6,9)]:
            subset=power[start:end][domains[start:end]==d]
            surrogate=surrogate+.5*(subset.sum(0)*derivative).sum()/total
    torch.testing.assert_close(torch.autograd.grad(surrogate,logits)[0],direct,rtol=1e-12,atol=1e-12)
