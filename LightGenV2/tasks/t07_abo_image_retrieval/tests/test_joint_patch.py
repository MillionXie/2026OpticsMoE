import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.frontend import Frontend
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import OpticalRetrieval
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config,apply_contract,patch_learning_rate_multiplier


def test_only_patch_training_and_rate_change_in_profile():
    old=overlay_config({},'domain_distill_joint_restart');new=overlay_config({},'domain_distill_joint_patch')
    assert new.pop('frontend_training')=='patch'
    assert new.pop('patch_learning_rate_multiplier')==.05
    for c in (old,new):c.pop('protocol')
    assert old==new


def test_patch_master_preserves_bf16_forward_and_has_nonzero_fp32_gradients():
    torch.set_num_threads(2);torch.manual_seed(42)
    frontend=Frontend(3).to(torch.bfloat16)
    pixels=torch.randn(196,1536)
    expected=frontend.patches(pixels,1).detach().clone()
    frontend.patch.float().requires_grad_(True)
    actual=frontend.patches(pixels,1)
    assert actual.dtype==torch.bfloat16
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.float().square().mean().backward()
    for n,p in frontend.named_parameters():
        if n.startswith('patch.'):
            assert p.requires_grad and p.dtype==torch.float32 and torch.isfinite(p.grad).all() and p.grad.norm()>0
        else:assert not p.requires_grad and p.grad is None


def test_only_existing_two_patch_parameters_unfreeze_with_unchanged_shapes():
    metadata={'token_count':3,'input_rms':True,'fusion_alpha_min':.4001,'fusion_alpha_max':.8}
    torch.manual_seed(4);old=OpticalRetrieval(metadata)
    payload={'metadata':metadata,'state_dict':old.state_dict()}
    updated=apply_contract(payload,overlay_config({},'domain_distill_joint_patch'))
    model=OpticalRetrieval(updated['metadata']);model.load_state_dict(updated['state_dict'])
    changed={n for (n,p),(k,q) in zip(old.named_parameters(),model.named_parameters()) if p.requires_grad!=q.requires_grad}
    assert changed=={'frontend.patch.weight','frontend.patch.bias'}
    assert sum(p.numel() for n,p in model.named_parameters() if n in changed)==1573888
    assert {n:tuple(v.shape) for n,v in old.state_dict().items()}=={n:tuple(v.shape) for n,v in model.state_dict().items()}
    assert all(torch.equal(v,old.state_dict()[n]) for n,v in model.state_dict().items())
    assert metadata.get('frontend_training') is None
    assert model.frontend.patch.weight.dtype==torch.float32 and model.frontend.position.weight.dtype==torch.bfloat16
    assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize('value',[0,-.1,1.1,True,float('nan'),'0.05'])
def test_invalid_patch_multiplier_rejected(value):
    with pytest.raises(ValueError):patch_learning_rate_multiplier({'patch_learning_rate_multiplier':value})
