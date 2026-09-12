import copy
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import (
    shift_router_phase_origin,overlay_config,apply_contract,
)
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.optics import Router,phase_modulation


def payload():
    torch.manual_seed(44)
    state={}
    for mode in ('vision','language'):
        raw=torch.randn(224,224)
        raw[:,::2]=-9.2
        state[mode+'.optics.router.raw_router_phase']=raw
    state['untouched']=torch.randn(2)
    return dict(metadata={'fusion_alpha_min':.4001},state_dict=state)


def test_router_origin_is_idempotent_and_only_changes_two_raw_tensors():
    old=payload();before=copy.deepcopy(old);new=shift_router_phase_origin(old,.25)
    assert old['metadata']==before['metadata'] and new['metadata']['router_initial_phase_offset_turns']==.25
    assert shift_router_phase_origin(new,.25) is new
    assert new['state_dict']['untouched'] is old['state_dict']['untouched']
    for name,raw in old['state_dict'].items():
        assert torch.equal(raw,before['state_dict'][name])
        if name.endswith('raw_router_phase'):
            ratio=phase_modulation(new['state_dict'][name])/phase_modulation(raw)
            assert (ratio-1j).abs().max()<2e-6
            assert (new['state_dict'][name].sigmoid()<.01).float().mean()<.02
    for invalid in (0.,1.,True,-.25,'quarter',float('nan')):
        with pytest.raises(ValueError):shift_router_phase_origin(old,invalid)
    with pytest.raises(ValueError):shift_router_phase_origin(new,.5)
    damaged=copy.deepcopy(old);damaged['state_dict']['vision.optics.router.raw_router_phase']=torch.zeros(223,224)
    with pytest.raises(ValueError):shift_router_phase_origin(damaged,.25)


@pytest.mark.parametrize('length',[77,196])
def test_router_origin_preserves_ideal_ccd_but_not_bypass_light(length):
    p=payload();shifted=shift_router_phase_origin(p,.25)
    old,new=Router(),Router()
    with torch.no_grad():
        old.raw_router_phase.copy_(p['state_dict']['vision.optics.router.raw_router_phase'])
        new.raw_router_phase.copy_(shifted['state_dict']['vision.optics.router.raw_router_phase'])
    torch.manual_seed(52)
    amplitude=torch.rand(2,224,224);amplitude[:,length:]=0
    old.eval();new.eval()
    with torch.no_grad():
        a,b=old(amplitude),new(amplitude)
    ca,cb=old.last['intensity'],new.last['intensity']
    assert (ca-cb).norm()/ca.norm()<3e-6
    assert torch.equal(old.last['selected_indices'],new.last['selected_indices'])
    torch.testing.assert_close(a,b,rtol=3e-5,atol=1e-6)
    old.train();new.train()
    with torch.no_grad():
        torch.manual_seed(88);old(amplitude)
        torch.manual_seed(88);new(amplitude)
    ca,cb=old.last['intensity'],new.last['intensity']
    assert (ca-cb).norm()/ca.norm()>1e-3  # Explicitly NOT invariant under bypass.


def test_router_origin_profile_retains_original_training_and_inference_contract():
    base=overlay_config({},'domain_distill_joint_restart')
    cfg=overlay_config({},'domain_distill_joint_routerorigin');new=copy.deepcopy(cfg)
    assert new.pop('router_initial_phase_offset_turns')==.25
    for c in (base,new):c.pop('protocol')
    assert base==new and 'frontend_training' not in new
    p=payload();converted=apply_contract(p,cfg)
    assert converted['metadata']['router_initial_phase_offset_turns']==.25
    assert apply_contract(converted,cfg)['state_dict'] is converted['state_dict']
