import math
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.phase_optimization import (
    router_coordinates,router_radian_step,phase_to_raw,circular_router_ema)


def test_radian_adam_matches_direct_phase_optimization_including_wrap():
    raw=torch.nn.Parameter(torch.tensor([-9.,-2.,0.,2.,9.]))
    theta=torch.nn.Parameter((2*math.pi)*raw.detach().sigmoid())
    a=torch.optim.AdamW([dict(params=[raw],kind='router',lr=.003)],weight_decay=0)
    b=torch.optim.AdamW([theta],lr=.003,weight_decay=0)
    for _ in range(5):
        a.zero_grad();b.zero_grad()
        ((2*math.pi)*raw.sigmoid()).sin().sum().backward()
        theta.sin().sum().backward()
        with router_radian_step(a,True):
            torch.testing.assert_close(raw.grad,theta.grad,atol=2e-6,rtol=2e-6)
            torch.nn.utils.clip_grad_norm_([raw],1.)
            a.step()
        torch.nn.utils.clip_grad_norm_([theta],1.);b.step()
        torch.testing.assert_close(torch.exp(1j*(2*math.pi)*raw.sigmoid()),torch.exp(1j*theta),atol=2e-6,rtol=2e-6)
        assert torch.isfinite(raw).all()


def test_radian_context_restores_raw_on_exception_and_does_not_touch_electronics():
    raw=torch.nn.Parameter(torch.tensor([-8.,3.]));raw.grad=torch.ones(2)
    electronic=torch.nn.Parameter(torch.ones(2));electronic.grad=torch.ones(2)
    opt=torch.optim.AdamW([dict(params=[raw],kind='router',lr=.01,weight_decay=0),
                           dict(params=[electronic],kind='electronic',lr=.01)])
    before=raw.detach().clone();gradient=raw.grad
    with pytest.raises(RuntimeError):
        with router_radian_step(opt,True):
            assert not torch.equal(raw,before)
            assert torch.equal(electronic,torch.ones(2))
            raise RuntimeError('interrupt')
    assert torch.equal(raw,before) and raw.grad is gradient
    with router_radian_step(opt,False):assert torch.equal(raw,before) and raw.grad is gradient
    opt.param_groups[0]['weight_decay']=.1
    with pytest.raises(ValueError):
        with router_radian_step(opt,True):pass
    assert torch.equal(raw,before)


def test_periodic_ema_does_not_average_opposite_logits_to_pi():
    a=phase_to_raw(torch.tensor([.01]));b=phase_to_raw(torch.tensor([2*math.pi-.01]))
    circular_router_ema(a,b,.5)
    assert (torch.exp(1j*(2*math.pi)*a.sigmoid())-1).abs().max()<2e-6


def test_radian_config_rejects_unsupported_combinations():
    assert router_coordinates({})=='raw'
    assert router_coordinates({'router_optimizer_coordinates':'radians','sam_rho':0})=='radians'
    for cfg in ({'router_optimizer_coordinates':'bad'},
                {'router_optimizer_coordinates':'radians','sam_rho':.01},
                {'router_optimizer_coordinates':'radians','router_initial_phase_offset_turns':.25}):
        with pytest.raises(ValueError):router_coordinates(cfg)


def test_radian_profile_changes_no_inference_or_other_training_contract():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config
    base=overlay_config({},'domain_distill_joint_restart')
    cfg=overlay_config({},'domain_distill_joint_routerradian')
    assert cfg.pop('router_optimizer_coordinates')=='radians'
    for c in (base,cfg):c.pop('protocol')
    assert base==cfg


def test_preflight_failure_does_not_mutate_previous_router():
    good=torch.nn.Parameter(torch.zeros(2));bad=torch.nn.Parameter(torch.tensor([100.,0.]))
    good.grad=torch.ones(2);bad.grad=torch.ones(2)
    opt=torch.optim.AdamW([dict(params=[good,bad],kind='router',lr=.01)],weight_decay=0)
    before=good.detach().clone()
    with pytest.raises(ValueError):
        with router_radian_step(opt,True):pass
    assert torch.equal(good,before)
