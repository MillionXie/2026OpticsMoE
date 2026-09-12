"""A training schedule, not a changed optical forward or extra inference head."""
import copy
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import (
    overlay_config,phase_only_group_frozen,PINNED_TEACHER_PROFILES)
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue import (
    PINNED_TEACHER_PROFILES as QUEUE_PROFILES)


def test_phase_first_changes_only_schedule():
    base=overlay_config({},'domain_distill_joint_restart')
    cfg=overlay_config({},'domain_distill_joint_phasefirst');new=copy.deepcopy(cfg)
    assert new.pop('phase_only_warmup_epochs')==4
    for c in (base,new):c.pop('protocol')
    assert base==new and PINNED_TEACHER_PROFILES==QUEUE_PROFILES
    for epoch in (1,4,5,16):
        for kind in ('phase','router','alpha','electronic','adapter','optical_electronic','readout','auxiliary'):
            assert phase_only_group_frozen(cfg,epoch,kind)==(epoch<=4 and kind not in ('phase','router'))
            assert not phase_only_group_frozen({},epoch,kind)


def test_phase_first_keeps_gradient_path_but_no_frozen_adam_updates():
    kinds=('phase','router','alpha','electronic','adapter','optical_electronic','readout','auxiliary')
    params=[torch.nn.Parameter(torch.tensor(0.3)) for _ in kinds]
    optimizer=torch.optim.AdamW([dict(params=[p],kind=k,lr=.01) for p,k in zip(params,kinds)],weight_decay=.5)
    cfg={'phase_only_warmup_epochs':4}
    original=[p.detach().clone() for p in params]
    for epoch in range(1,6):
        optimizer.zero_grad(set_to_none=True)
        # All parameters participate in the graph, including frozen electronics.
        loss=torch.stack([p.sin()+1 for p in params]).prod();loss.backward()
        assert all(p.grad is not None and p.grad.abs()>0 for p in params)
        for g in optimizer.param_groups:
            frozen=phase_only_group_frozen(cfg,epoch,g['kind'])
            g['lr']=0. if frozen else .01
            if frozen:
                for p in g['params']:p.grad=None
        optimizer.step()
        for p,k,before in zip(params,kinds,original):
            if epoch<=4 and k not in ('phase','router'):
                assert torch.equal(p,before) and p not in optimizer.state
            else:assert not torch.equal(p,before)


@pytest.mark.parametrize('count',[-1,True,1.5,'4'])
def test_invalid_phase_first_schedule_rejected(count):
    with pytest.raises(ValueError):phase_only_group_frozen({'phase_only_warmup_epochs':count},1,'phase')
