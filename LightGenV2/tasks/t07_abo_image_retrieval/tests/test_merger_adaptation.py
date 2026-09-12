"""Existing V->L linear adaptation; never add a transformer or optical layer."""
import copy
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import OpticalRetrieval
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.frontend import Frontend
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import (
    overlay_config, apply_contract, merger_learning_rate_multiplier,
)


def test_merger_profile_changes_only_existing_frontend_trainability():
    base=overlay_config({},'domain_distill_joint_restart')
    new=overlay_config({},'domain_distill_joint_merger')
    assert new.pop('frontend_training')=='merger_fc2'
    assert new.pop('merger_learning_rate_multiplier')==.05
    for cfg in (base,new):cfg.pop('protocol')
    assert base==new
    cfg=overlay_config({},'domain_distill_joint_merger')
    payload=dict(metadata={'fusion_alpha_min':.4001},state_dict={'p':torch.ones(1)})
    result=apply_contract(payload,cfg)
    assert result['metadata']['frontend_training']=='merger_fc2'
    assert 'frontend_training' not in payload['metadata']
    assert result['state_dict'] is payload['state_dict']
    assert merger_learning_rate_multiplier(cfg)==.05
    for value in (0,-1,2,True,'0.05',float('nan'),float('inf')):
        with pytest.raises(ValueError):merger_learning_rate_multiplier({'merger_learning_rate_multiplier':value})
    with pytest.raises(ValueError):apply_contract(payload,dict(cfg,frontend_training='all_qwen'))


def test_frozen_frontend_preserves_original_merge_computation_exactly():
    torch.manual_seed(123)
    frontend=Frontend(4).to(torch.bfloat16)
    x=torch.randn(1,4,1024).to(torch.bfloat16)
    with torch.no_grad():
        expected=frontend.merger_fc2(frontend.gelu(frontend.merger_fc1(
            frontend.merger_norm(x.reshape(-1,1024)).view(-1,4096))))
        actual=frontend.merge(x)
    assert torch.equal(actual,expected)
    assert not any(p.requires_grad for p in frontend.parameters())


def test_only_existing_fc2_unfrozen_fp32_state_and_gradients():
    metadata=dict(token_count=4,input_rms=1.,fusion_alpha_min=.4001,fusion_alpha_max=.8)
    torch.manual_seed(123)
    base=OpticalRetrieval(metadata)
    model=OpticalRetrieval(dict(metadata,frontend_training='merger_fc2'))
    model.load_state_dict(base.state_dict(),strict=True)
    a,b=base.audit(),model.audit()
    assert b['trainable_parameters']-a['trainable_parameters']==8390656
    assert a['frozen_parameters']-b['frozen_parameters']==8390656
    assert b['frontend_trainable_parameters']==8390656 and a['frontend_trainable_parameters']==0
    assert b['native_transformer_modules']==b['attention_modules']==0
    assert b['capture_count']==6 and b['top_k']==2
    assert b['architecture']==a['architecture']
    assert {n for n,p in model.frontend.named_parameters() if p.requires_grad}=={'merger_fc2.weight','merger_fc2.bias'}
    assert model.state_dict().keys()==base.state_dict().keys()
    for n,p in model.state_dict().items():
        assert torch.equal(p.float(),base.state_dict()[n].float())
    model.train()
    assert not model.frontend.training and model.frontend.merger_fc2.weight.requires_grad
    assert model.frontend.merger_fc2.weight.dtype==torch.float32
    x=torch.randn(1,4,1024).to(torch.bfloat16).requires_grad_(True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        expected=base.frontend.merge(x)
        actual=model.frontend.merge(x)
    assert torch.equal(actual,expected) and actual.dtype==torch.bfloat16
    # Also support the non-autocast CPU diagnostic path with FP32 master weights.
    output=model.frontend.merge(x)
    output.float().square().mean().backward()
    assert output.dtype==torch.bfloat16 and x.grad is not None
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    for n,p in model.frontend.named_parameters():
        if n.startswith('merger_fc2.'):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
        else:assert p.grad is None
    before=model.frontend.merger_fc2.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.frontend.merger_fc2.parameters(),lr=1e-5,weight_decay=0)
    optimizer.step()
    assert not torch.equal(before,model.frontend.merger_fc2.weight)
    saved=copy.deepcopy(model.state_dict())
    restored=OpticalRetrieval(dict(metadata,frontend_training='merger_fc2'))
    restored.load_state_dict(saved,strict=True)
    assert torch.equal(restored.frontend.merger_fc2.weight,model.frontend.merger_fc2.weight)
    # Every physical tensor remains identical to the original model in this probe.
    for name,p in restored.state_dict().items():
        if '.optics.' in name:assert torch.equal(p,base.state_dict()[name])
