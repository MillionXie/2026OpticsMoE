import pytest
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t17_router_classification.model_linear_readout import LinearReadoutLG
from LightGenV2.tasks.t17_router_classification.phase_optimizer import WrappedPhaseAdam
from LightGenV2.tasks.t17_router_classification.phase_training import phase_update_mode,phase_parameter_groups,backward_training_loss


def test_accumulated_gradient_equals_full_batch_on_actual_canonical_optical_model():
    torch.set_num_threads(4);torch.manual_seed(7)
    full=LinearReadoutLG(2,10);split=LinearReadoutLG(2,10);split.load_state_dict(full.state_dict())
    x=torch.rand(4,224,224);y=torch.arange(4)
    F.cross_entropy(full(x)['raw_logits'],y).backward()
    for a,z in [(0,1),(1,4)]:backward_training_loss(F.cross_entropy(split(x[a:z])['raw_logits'],y[a:z]),z-a,4,True)
    for (_,a),(_,b) in zip(full.named_parameters(),split.named_parameters()):
        assert torch.allclose(a.grad,b.grad,rtol=3e-4,atol=2e-7)


def test_default_phase_group_has_bitwise_same_wrapped_update():
    torch.manual_seed(11);m=LinearReadoutLG(2,10);p=m.global_phase;q=torch.nn.Parameter(p.detach().clone());g=torch.randn_like(p)
    p.grad=g.clone();q.grad=g.clone()
    WrappedPhaseAdam([p],.0005).step();WrappedPhaseAdam(phase_parameter_groups(m,[q],dict(phase_lr=.0005)),.0005).step()
    assert torch.equal(p,q)


def test_router_lr_multiplier_does_not_add_or_lose_any_phase_parameter():
    m=LinearReadoutLG(2,10);phases=[p for n,p in m.named_parameters() if n!='shared_head.weight'];groups=phase_parameter_groups(m,phases,dict(phase_lr=.03,router_lr_multiplier=.1))
    assert {id(p) for g in groups for p in g['params']}=={id(p) for p in phases}
    assert groups[1]['params'][0] is m.router_phase and groups[1]['lr']==pytest.approx(.003)


def test_unknown_or_incompatible_accumulation_policy_fails_explicitly():
    assert phase_update_mode({})=='mini_batch'
    with pytest.raises(ValueError):phase_update_mode(dict(phase_update_mode='typo'))
    with pytest.raises(ValueError):phase_update_mode(dict(phase_update_mode='full_TRAIN',architecture='D2NN',phase_optimizer='raw'))
