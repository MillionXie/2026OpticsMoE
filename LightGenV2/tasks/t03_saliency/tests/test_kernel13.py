from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t03_saliency.modeling import configure_spatial_kernel, initialize_student, architecture_label
from LightGenV2.tasks.t03_saliency.settings import load_settings
from .test_lightweight_residual import TinyFusion
from .test_spatial_ffn import pack

ROOT=Path(__file__).resolve().parents[1]/'configs'

def test_kernel13_identity_budget_rng_and_true_spatial_grid():
    s=load_settings(ROOT/'moe_alpha40_viewreg_control.yaml')
    original=TinyFusion(s);target=deepcopy(original)
    rng=torch.random.get_rng_state().clone()
    configure_spatial_kernel(target,13)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert sum(p.numel() for p in target.parameters())-sum(p.numel() for p in original.parameters())==61440
    grid=torch.randn(2,192,14,14);x=pack(grid);shapes=[(1,14,14)]*2
    for a,b in zip(original.blocks,target.blocks):
        torch.testing.assert_close(a._mix_spatial_tokens(x,shapes),b._mix_spatial_tokens(x,shapes))
        b._mix_spatial_tokens(x,shapes).square().mean().backward()
        assert b.token_depthwise.weight.grad[:,:,0,:].abs().sum()>0
        with torch.no_grad(): b.token_depthwise.weight.copy_(torch.randn_like(b.token_depthwise.weight))
        expected=pack(F.conv2d(grid,b.token_depthwise.weight,padding=6,groups=192))
        torch.testing.assert_close(b._mix_spatial_tokens(x,shapes),expected)
        torch.testing.assert_close(b._mix_spatial_tokens(x[:1],shapes[:1]),expected[:1])

def test_kernel13_strict_checkpoint_transfer_preserves_original_residual(tmp_path):
    base=load_settings(ROOT/'moe_alpha40_viewreg_control.yaml')
    s=load_settings(ROOT/'moe_alpha40_viewreg_kernel13.yaml')
    source=torch.nn.Module();source.hybrid=TinyFusion(base);head=torch.nn.Linear(192,1)
    path=tmp_path/'source.pt'
    torch.save({'architecture':architecture_label(base),'epoch':20,'core':source.state_dict(),
                'saliency_head':head.state_dict()},path)
    s.initialization_checkpoint=path;s.initialization_checkpoint_sha256=None
    target=torch.nn.Module();target.hybrid=TinyFusion(s)
    m=SimpleNamespace(core=target,head=torch.nn.Linear(192,1),checkpoint_architecture=architecture_label(s))
    r=initialize_student(m,s)
    assert '13x13' in r['kernel_transfer'] and not r['fusion_reset']
    x=torch.randn(2,196,192)
    kw=dict(padding_mask=torch.zeros(2,196,dtype=torch.bool),causal=False,spatial_shapes=[(1,14,14)]*2)
    for a,b in zip(source.hybrid.blocks,target.hybrid.blocks):
        torch.testing.assert_close(a(x,**kw),b(x,**kw))

def test_kernel13_profile_keeps_train_and_optical_contracts():
    a=load_settings(ROOT/'moe_alpha40_viewreg_control.yaml')
    b=load_settings(ROOT/'moe_alpha40_viewreg_kernel13.yaml')
    for key in ('student_epochs','student_learning_rate','phase_learning_rate','router_learning_rate',
                'dense_head_learning_rate','weight_decay','distillation_initial_weight',
                'distillation_final_weight','fusion_alpha_min','fusion_alpha_max','active_size','expert_size',
                'router_backend','top_k','initialization_checkpoint_sha256','augmentation_mode',
                'augmentation_end_epoch','language_optical_phase_zero_order_intensity_min',
                'language_optical_phase_zero_order_intensity_max'):
        assert getattr(a,key)==getattr(b,key),key
    assert b.electronic_spatial_kernel_size==13 and b.expand_kernel_on_warmstart
    assert b.fusion_alpha_min==.4 and b.top_k==2 and b.router_backend=='optical'
    assert not b.electronic_grn and not b.electronic_ffn_spatial_dilation
