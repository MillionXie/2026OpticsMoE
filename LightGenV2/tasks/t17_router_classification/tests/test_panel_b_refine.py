import pytest
import torch
from LightGenV2.tasks.t17_router_classification.model_depth_ccd import DepthLG
from LightGenV2.tasks.t17_router_classification.refine_panel_b_depth import configure_trainable, assert_frozen
from LightGenV2.tasks.t17_router_classification.phase_optimizer import WrappedPhaseAdam


@pytest.mark.parametrize('depth',[4,6])
@pytest.mark.parametrize('mode',['frozen_router_control','last_pair_only'])
def test_refinement_optimizes_exact_requested_phases_and_preserves_frozen_fields(depth,mode):
    torch.set_num_threads(4);torch.manual_seed(17);model=DepthLG(depth,8)
    parameters=configure_trainable(model,mode)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    expected=(depth//2)*429188 if mode=='frozen_router_control' else 429188
    assert sum(p.numel() for p in parameters)==expected and 'router_phase' in frozen
    optimizer=WrappedPhaseAdam(parameters,.001)
    loss=-model(torch.rand(1,224,224))['logits'][0,2];loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    optimizer.step();assert_frozen(model,frozen)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(not torch.equal(p,before[n]) for n,p in model.named_parameters() if p.requires_grad)
    with torch.no_grad():model.router_phase[0,0]+=1
    with pytest.raises(RuntimeError,match='Frozen phase changed'):assert_frozen(model,frozen)


def test_human_fixed_two_layer_LG_cannot_enter_refinement():
    with pytest.raises(ValueError,match='human-fixed'):configure_trainable(DepthLG(2,8),'last_pair_only')
