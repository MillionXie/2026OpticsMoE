import pytest
import torch

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import (
    PROFILES, learning_rate_multiplier, optical_curriculum_scope, set_parameter_scope, update_trainable_ema)


def test_external_freezes_electronics_then_target_reenables_them():
    profile = PROFILES['sku_optical_pretrain']
    params = [(name, torch.nn.Parameter(torch.ones(2))) for name in
              ['vision.optics.experts.0', 'vision.optics.router.raw_router_phase',
               'readout.linear.weight', 'vision.block1_optical_fusion_logit']]
    optimizer = torch.optim.AdamW([p for _, p in params], lr=.1, weight_decay=.1)
    ema = {n: p.detach().clone() for n, p in params}
    original = {n: p.detach().clone() for n, p in params}
    set_parameter_scope(params, optical_only=optical_curriculum_scope(profile, True, 36))
    optimizer.zero_grad(set_to_none=True)
    sum(p.square().sum() for _, p in params).backward()
    optimizer.step()
    update_trainable_ema(params, ema)
    for n, p in params[2:]:
        assert p.grad is None and torch.equal(p, original[n]) and torch.equal(ema[n], original[n])
    assert all(not torch.equal(p, original[n]) for n, p in params[:2])
    optimizer.state.clear()
    set_parameter_scope(params, optical_only=optical_curriculum_scope(profile, False, 36))
    optimizer.zero_grad(set_to_none=True)
    sum(p.square().sum() for _, p in params).backward()
    optimizer.step()
    assert all(p.requires_grad and not torch.equal(p, original[n]) for n, p in params)


def test_external_phase_rate_does_not_change_router_or_target_rates():
    p = PROFILES['sku_optical_pretrain']
    assert learning_rate_multiplier(p, 'vision.optics.experts.0', True, True) == 5.
    assert learning_rate_multiplier(p, 'vision.optics.experts.0', False, True) == 1.
    assert learning_rate_multiplier(p, 'vision.optics.router.raw_router_phase', True, True) == .1
    assert learning_rate_multiplier(p, 'readout.linear.weight', True, True) == 1.
    for external in [False, True]:
        assert not optical_curriculum_scope(PROFILES['sku_capacity_control'], external, 36)
    assert optical_curriculum_scope(PROFILES['phase_only'], False, 0)


@pytest.mark.parametrize('changes', [dict(), dict(warmup=1), dict(optical_only=True),
                                    dict(teacher_weight=.1), dict(head_expansion='spatial2x2_64')])
def test_bad_optical_pretraining_combinations_fail(changes):
    profile = dict(PROFILES['sku_optical_pretrain'], **changes)
    with pytest.raises(ValueError, match='Optical pretraining'):
        optical_curriculum_scope(profile, True, 36 if changes else 0)
