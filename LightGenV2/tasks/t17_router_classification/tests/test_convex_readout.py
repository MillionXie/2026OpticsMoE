import pytest
import torch
from LightGenV2.tasks.t17_router_classification.train_convex_readout import convex_objective,training_coordinates,canonical_weight,coordinate_objective


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


def test_cholesky_coordinates_preserve_logits_and_original_weight_penalty():
    torch.manual_seed(17);x=torch.rand(40,8,dtype=torch.double);w=torch.randn(3,8,dtype=torch.double);y=torch.arange(40)%3
    transformed,v,factor,info=training_coordinates(x,w,'TRAIN_Gram_cholesky')
    assert info['TRAIN_samples']==40 and torch.allclose(canonical_weight(v,factor),w,atol=1e-12,rtol=1e-12)
    assert torch.allclose(x@w.T,transformed@v.T,atol=1e-12,rtol=1e-12)
    v=(v+.1).requires_grad_();base=convex_objective(canonical_weight(v,factor),x,y,w,.001);other=coordinate_objective(v,transformed,y,w,.001,factor)
    assert torch.allclose(base,other,atol=1e-12,rtol=1e-12)
    g1=torch.autograd.grad(base,v,retain_graph=True)[0];g2=torch.autograd.grad(other,v)[0]
    assert torch.allclose(g1,g2,atol=1e-12,rtol=1e-12)


def test_identity_coordinates_leave_original_objective_bitwise_unchanged():
    x=torch.rand(20,6,dtype=torch.double);w=torch.rand(4,6,dtype=torch.double);y=torch.arange(20)%4
    xt,v,factor,_=training_coordinates(x,w,'identity')
    assert xt is x and factor is None and torch.equal(v,w)
    assert torch.equal(coordinate_objective(v,xt,y,w,.001,factor),convex_objective(w,x,y,w,.001))


def test_unknown_training_coordinate_mode_is_rejected():
    with pytest.raises(ValueError):training_coordinates(torch.ones(3,4),torch.ones(2,4),'from_TEST')
