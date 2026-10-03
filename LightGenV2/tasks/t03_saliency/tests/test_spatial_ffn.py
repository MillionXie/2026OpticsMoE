from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t03_saliency.lightweight_residual import (
    PackedSpatialDepthwise, configure_spatial_ffn, initialize_identity_spatial_ffn)
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label, initialize_student, optimizer
from LightGenV2.tasks.t03_saliency.training import staged_epoch
from .test_lightweight_residual import TinyFusion

TASK = Path(__file__).resolve().parents[1]


def pack(grid):
    # Explicit true spatial coordinates in the native 2x2-block traversal.
    return torch.stack([grid[:, :, y, x] for by in range(0,14,2)
                        for bx in range(0,14,2) for y in range(by,by+2)
                        for x in range(bx,bx+2)], dim=1)


@pytest.mark.parametrize('dilation', [1,2])
def test_identity_true_spatial_order_gradients_and_sample_independence(dilation):
    rng = torch.random.get_rng_state().clone()
    m = PackedSpatialDepthwise(3,dilation)
    assert torch.equal(torch.random.get_rng_state(), rng)
    grid = torch.randn(2,3,14,14)
    x = pack(grid).requires_grad_()
    torch.testing.assert_close(m(x), x, atol=0, rtol=0)
    m(x).square().mean().backward()
    assert m.conv.weight.grad[:,:,0,:].abs().sum() > 0
    with torch.no_grad():
        m.conv.weight.copy_(torch.randn_like(m.conv.weight))
    expected = pack(F.conv2d(grid,m.conv.weight,padding=dilation,dilation=dilation,groups=3))
    torch.testing.assert_close(m(x), expected)
    torch.testing.assert_close(m(x[:1]), m(x)[:1])
    with pytest.raises(ValueError): m(torch.randn(1,195,3))


@pytest.mark.parametrize('dilation',[1,2])
def test_full_real_residual_transfer_and_new_optimizer_group(tmp_path,dilation):
    original = load_settings(TASK/'configs/moe_alpha40_cffn_control.yaml')
    source = torch.nn.Module(); source.hybrid = TinyFusion(original)
    head = torch.nn.Linear(192,1)
    path = tmp_path/'source.pt'
    torch.save({'architecture':architecture_label(original),'epoch':5,
                'core':source.state_dict(),'saliency_head':head.state_dict()},path)
    s=load_settings(TASK/f'configs/moe_alpha40_cffn_d{dilation}.yaml')
    s.initialization_checkpoint, s.initialization_checkpoint_sha256 = path,None
    target=torch.nn.Module(); target.hybrid=TinyFusion(s)
    configure_spatial_ffn(target.hybrid,dilation)
    model=SimpleNamespace(core=target,head=torch.nn.Linear(192,1),checkpoint_architecture=architecture_label(s))
    report=initialize_student(model,s)
    assert report['identity_spatial_ffn_added'] and not report['fusion_reset']
    assert sum(p.numel() for p in target.parameters())-sum(p.numel() for p in source.parameters())==6912
    x=torch.randn(2,196,192)
    kwargs=dict(padding_mask=torch.zeros(2,196,dtype=torch.bool),causal=False,spatial_shapes=[(1,14,14)]*2)
    for i in range(2):
        torch.testing.assert_close(target.hybrid.blocks[i](x,**kwargs),source.hybrid.blocks[i](x,**kwargs))
    target.phase=torch.nn.Parameter(torch.zeros(1))
    target.phase_parameters=lambda:[target.phase]
    target.router=torch.nn.Linear(2,2)
    target.readout=torch.nn.Linear(2,2)
    wrapped=torch.nn.Module();wrapped.core=target;wrapped.head=model.head
    opt=optimizer(wrapped,s)
    groups={g['name']:g for g in opt.param_groups}
    assert sum(p.numel() for p in groups['electronic_ffn_spatial']['params'])==6912
    assert groups['electronic_ffn_spatial']['weight_decay']==0
    staged_epoch(opt,s,1)
    assert all(not p.requires_grad for p in groups['electronic']['params'])
    assert all(p.requires_grad for p in groups['electronic_ffn_spatial']['params'])
    assert groups['electronic_ffn_spatial']['lr']==1e-4
    staged_epoch(opt,s,6)
    assert all(p.requires_grad for p in groups['electronic']['params'])


def test_transfer_rejects_unrelated_keys_and_profiles_keep_optics():
    with pytest.raises(RuntimeError):
        initialize_identity_spatial_ffn({'old':torch.ones(1)}, {'old':torch.ones(1),'other':torch.ones(1)})
    configs=[load_settings(TASK/f'configs/moe_alpha40_cffn_{n}.yaml') for n in ('control','d1','d2')]
    for s in configs:
        assert s.initialization_checkpoint_sha256==configs[0].initialization_checkpoint_sha256
        assert s.electronic_spatial_kernel_size==3 and not s.electronic_grn
        assert s.fusion_alpha_min==.4 and not s.reset_fusion_on_warmstart
        assert s.top_k==2 and s.router_backend=='optical'
        assert s.active_size==478 and s.expert_size==224
        assert s.language_optical_phase_zero_order_intensity_min==.2
        assert s.language_optical_phase_zero_order_intensity_max==.3
        assert s.adaptive_plateau_enabled and s.distillation_final_weight==.6
    assert len({architecture_label(s) for s in configs})==3
