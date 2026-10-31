import copy
import pytest
import torch
from LightGenV2.tasks.t17_router_classification.train_panel_b_depth import build_job,schedule,JOBS
from LightGenV2.tasks.t17_router_classification.phase_optimizer import WrappedPhaseAdam


def test_same_schedule_and_epoch_budget_for_all_D2NN_depths():
    cfg=dict(epochs=120,stage_epochs=60,stage_learning_rates=[.003,.001],stage_focal_gamma=[0.,2.])
    expected=[schedule(e,cfg) for e in range(1,121)]
    for name in ['d2nn3','d2nn5','d2nn7']:
        assert JOBS[name][0]=='d2nn' and [schedule(e,copy.deepcopy(cfg)) for e in range(1,121)]==expected
    assert expected[0]==(0,.003,0.) and expected[60]==(1,.001,2.)
    assert abs(expected[59][1]-.0006)<1e-10 and abs(expected[-1][1]-.0002)<1e-10
    with pytest.raises(ValueError):schedule(121,cfg)


def test_optimizer_updates_extra_depth_masks_instead_of_only_A_two_masks():
    torch.set_num_threads(4);model=build_job('lg4');before={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer=WrappedPhaseAdam(list(model.parameters()),.003)
    (-model(torch.rand(1,224,224))['logits'][0,3]).backward();optimizer.step()
    assert set(before)==set(dict(model.named_parameters()))
    assert all(torch.isfinite(p).all() and not torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert 'extra_expert_phases.0' in before and 'extra_global_phases.0' in before
