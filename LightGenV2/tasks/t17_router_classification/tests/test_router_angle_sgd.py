import pytest
import torch
from LightGenV2.tasks.t17_router_classification.phase_optimizer import WrappedPhaseAdam
from LightGenV2.tasks.t17_router_classification.router_gradient_projection import validate_projection_profile


def jac(raw):
    s=raw.detach().double().sigmoid();return 2*torch.pi*s*(1-s)


def test_router_angle_sgd_preserves_direction_and_has_no_router_adam_state():
    r=torch.nn.Parameter(torch.tensor([-.4,.3],dtype=torch.float64));e=torch.nn.Parameter(torch.tensor([.2,.4],dtype=torch.float64))
    before=2*torch.pi*r.detach().sigmoid();g=torch.tensor([2.,-.5],dtype=torch.float64);r.grad=g*jac(r);e.grad=torch.ones_like(e)
    opt=WrappedPhaseAdam([r,e],.003,router_parameter=r,router_step_rule='normalized_angle_sgd');opt.step()
    after=2*torch.pi*r.detach().sigmoid();assert torch.allclose(after-before,-.003*g/g.abs().max(),atol=1e-12,rtol=0)
    assert r not in opt.state and e in opt.state and torch.isfinite(r).all()


def test_nonrouter_adam_update_keeps_the_same_shared_clipping():
    r=torch.nn.Parameter(torch.tensor([.2,.4],dtype=torch.float64));e=torch.nn.Parameter(torch.tensor([.1,.3],dtype=torch.float64));rr=torch.nn.Parameter(r.detach().clone());ee=torch.nn.Parameter(e.detach().clone())
    for a,b,g in [(r,rr,torch.tensor([10.,-3.],dtype=torch.float64)),(e,ee,torch.tensor([4.,7.],dtype=torch.float64))]:a.grad=g.clone();b.grad=g.clone()
    WrappedPhaseAdam([r,e],.002,router_parameter=r,router_step_rule='normalized_angle_sgd').step();WrappedPhaseAdam([rr,ee],.002).step()
    assert torch.equal(e,ee)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_zero_router_gradient_keeps_raw_bitwise(dtype):
    r=torch.nn.Parameter(torch.tensor([-.12345,.7543],dtype=dtype));before=r.detach().clone();r.grad=torch.zeros_like(r)
    WrappedPhaseAdam([r],.003,router_parameter=r,router_step_rule='normalized_angle_sgd').step();assert torch.equal(r,before)


def test_router_step_wraps_periodically_and_respects_group_rate():
    r=torch.nn.Parameter(torch.tensor([-12.,12.],dtype=torch.float64));g=torch.tensor([1.,-1.],dtype=torch.float64);before=2*torch.pi*r.detach().sigmoid();r.grad=g*jac(r)
    WrappedPhaseAdam([dict(params=[r],lr=.004)],.02,router_parameter=r,router_step_rule='normalized_angle_sgd').step();after=2*torch.pi*r.detach().sigmoid();delta=torch.remainder(after-before+torch.pi,2*torch.pi)-torch.pi
    assert torch.allclose(delta,-.004*g,atol=1e-7,rtol=0)


def test_explicit_default_adam_remains_bitwise_identical():
    a=torch.nn.Parameter(torch.tensor([-.2,.7]));b=torch.nn.Parameter(a.detach().clone());x=WrappedPhaseAdam([a],.001);y=WrappedPhaseAdam([b],.001,router_step_rule='adam')
    for i in range(3):
        a.grad=torch.tensor([.3+i,-.2]);b.grad=a.grad.clone();x.step();y.step();assert torch.equal(a,b)


def test_router_sgd_rejects_missing_parameter_or_incompatible_training_profile():
    p=torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError):WrappedPhaseAdam([p],.001,router_step_rule='normalized_angle_sgd')
    with pytest.raises(ValueError):validate_projection_profile(dict(router_step_rule='normalized_angle_sgd'))


def test_router_sgd_restores_all_raw_parameters_on_later_nonfinite_gradient():
    r=torch.nn.Parameter(torch.tensor([.2,.3]));e=torch.nn.Parameter(torch.tensor([.4,.5]));before=[r.detach().clone(),e.detach().clone()];r.grad=torch.ones_like(r);e.grad=torch.tensor([1.,float('nan')])
    opt=WrappedPhaseAdam([r,e],.001,router_parameter=r,router_step_rule='normalized_angle_sgd')
    with pytest.raises(RuntimeError):opt.step()
    assert torch.equal(r,before[0]) and torch.equal(e,before[1])
