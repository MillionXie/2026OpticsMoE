import types

import pytest
import torch

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import (
    PROFILES, optical_parameter, set_parameter_scope, update_trainable_ema, non_optical_digest,
    configure_phase_head_scope, attach_train_readout_dropout, projection_parameter, learning_rate_multiplier)


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


def test_phase_head_scope_freezes_alpha_residual_norm_and_frontend():
    p=parameters()+[(name,torch.nn.Parameter(torch.randn(8))) for name in (
        'readout.projection.weight','readout.projection.bias','readout.norm.weight','frontend.embed_tokens.weight')]
    model=types.SimpleNamespace(named_parameters=lambda:iter(p),readout=types.SimpleNamespace(kind='linear64'))
    digest=configure_phase_head_scope(model)
    expected=lambda n: optical_parameter(n) or projection_parameter(n)
    assert all(x.requires_grad==expected(n) for n,x in p)
    selected=[(n,x) for n,x in p if x.requires_grad]
    ema={n:x.detach().clone() for n,x in selected}
    optimizer=torch.optim.AdamW([x for _,x in selected],lr=.01,weight_decay=.1)
    for _ in range(4):
        set_parameter_scope(selected)
        optimizer.zero_grad(set_to_none=True)
        sum(x.square().sum() for _,x in p).backward(); optimizer.step(); update_trainable_ema(selected,ema)
    assert digest==non_optical_digest(model,exclude_projection=True)
    assert all(x.grad is None for n,x in p if not expected(n))
    with torch.no_grad(): p[-2][1][0]+=1
    assert digest!=non_optical_digest(model,exclude_projection=True)


def test_training_head_hook_preserves_eval_and_checkpoint_structure():
    head=torch.nn.Linear(384,64)
    model=types.SimpleNamespace(readout=types.SimpleNamespace(kind='linear64',projection=head))
    x=torch.randn(4,384); original={k:v.clone() for k,v in head.state_dict().items()}
    head.eval(); expected=head(x)
    handle=attach_train_readout_dropout(model,.1)
    assert torch.equal(head(x),expected)
    head.train(); assert not torch.equal(head(x),expected)
    assert all(torch.equal(v,original[k]) for k,v in head.state_dict().items())
    handle.remove(); assert torch.equal(head(x),expected)
    with pytest.raises(ValueError): attach_train_readout_dropout(model,1.)
    model.readout.kind='relu128'
    with pytest.raises(ValueError): attach_train_readout_dropout(model,.1)


def test_phase_head_profile_lr_has_no_inference_expansion():
    p=PROFILES['sku_phase_head']
    assert p['phase_head_only'] and p['readout_input_dropout']==.1
    assert not any(p.get(k) for k in ['optical_only','external_optical_only','teacher_weight','warmup','head_expansion','electronic_expansion','ccd_readout_modes'])
    assert learning_rate_multiplier(p,'readout.projection.weight',False,True)==3
    assert learning_rate_multiplier(p,'readout.norm.weight',False,True)==1
    assert learning_rate_multiplier(p,'vision.optics.experts.0',False,True)==1
    assert learning_rate_multiplier(p,'vision.optics.router.raw_router_phase',False,True)==.1
