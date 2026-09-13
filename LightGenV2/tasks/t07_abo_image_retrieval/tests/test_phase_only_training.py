import types

import pytest
import torch

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import (
    PROFILES, optical_parameter, set_parameter_scope, update_trainable_ema, non_optical_digest)


def parameters():
    names=['vision.optics.experts.0','language.optics.global_phase',
           'vision.optics.router.raw_router_phase','vision.encoder.weight',
           'language.block1_optical_fusion_logit','readout.weight']
    return [(n,torch.nn.Parameter(torch.linspace(.01,.99,31))) for n in names]


def test_scope_freezes_electronics_alpha_and_head_but_not_any_phases():
    p=parameters();set_parameter_scope(p,optical_only=True)
    assert [x.requires_grad for _,x in p]==[True,True,True,False,False,False]
    set_parameter_scope(p,router_only=True)
    assert [x.requires_grad for _,x in p]==[False,False,True,False,False,False]
    set_parameter_scope(p)
    assert all(x.requires_grad for _,x in p)
    with pytest.raises(ValueError):set_parameter_scope(p,True,True)


def test_optimizer_and_ema_leave_frozen_electronics_bitwise_unchanged():
    p=parameters();model=types.SimpleNamespace(named_parameters=lambda:iter(p))
    before=non_optical_digest(model);ema={n:x.detach().clone() for n,x in p}
    frozen={n:x.detach().clone() for n,x in p if not optical_parameter(n)}
    set_parameter_scope(p,optical_only=True)
    optimizer=torch.optim.AdamW([x for _,x in p],lr=.01,weight_decay=.1)
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        sum(x.square().sum() for _,x in p).backward();optimizer.step()
        update_trainable_ema(p,ema)
    assert before==non_optical_digest(model)
    assert all(torch.equal(ema[n],v) for n,v in frozen.items())
    assert all(x.grad is None for n,x in p if not optical_parameter(n))
    with torch.no_grad():
        p[3][1][0]+=1
    assert before!=non_optical_digest(model)


def test_optical_lr_pair_changes_only_expert_global_step_size():
    a=dict(PROFILES['phase_only']);b=dict(PROFILES['phase_only_hot'])
    assert a.pop('phase_lr_multiplier')==1 and b.pop('phase_lr_multiplier')==3 and a==b
    assert a['optical_only'] and a['teacher_weight']==0 and a['warmup']==0
    assert a['noise_probability']==.1 and a['router_lr_multiplier']==.1
