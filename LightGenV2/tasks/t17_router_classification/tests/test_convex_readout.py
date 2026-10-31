import pytest
import torch
from LightGenV2.tasks.t17_router_classification.train_convex_readout import convex_objective


def test_frozen_feature_head_objective_is_convex_and_uses_only_train_labels():
    torch.manual_seed(17);x=torch.rand(16,8,dtype=torch.double);y=torch.arange(16)%3;a=torch.randn(3,8,dtype=torch.double);b=torch.randn_like(a);anchor=torch.zeros_like(a)
    middle=convex_objective((a+b)/2,x,y,anchor,.001)
    assert middle<=.5*(convex_objective(a,x,y,anchor,.001)+convex_objective(b,x,y,anchor,.001))+1e-12


def test_LBFGS_reduces_the_regularized_training_objective_without_modifying_features():
    torch.manual_seed(17);x=torch.randn(30,6,dtype=torch.double);before=x.clone();y=torch.arange(30)%3;anchor=torch.randn(3,6,dtype=torch.double);weight=torch.nn.Parameter(anchor.clone());opt=torch.optim.LBFGS([weight],max_iter=10,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=convex_objective(weight,x,y,anchor,.001);loss.backward();return loss
    initial=float(closure().detach());opt.step(closure)
    assert float(closure().detach())<initial and torch.equal(x,before) and torch.isfinite(weight.grad).all()


@pytest.mark.parametrize('strength',[0.,-1.])
def test_invalid_convex_regularization_is_rejected(strength):
    with pytest.raises(ValueError):convex_objective(torch.zeros(2,4),torch.ones(3,4),torch.zeros(3,dtype=torch.long),torch.zeros(2,4),strength)
