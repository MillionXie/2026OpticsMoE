from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t03_saliency.lightweight_residual import (
    SpatialTokenLinear,configure_global_mixing,configure_spatial_ffn,
    initialize_identity_global_mixing)
from LightGenV2.tasks.t03_saliency.modeling import architecture_label,initialize_student,optimizer
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.training import staged_epoch
from .test_lightweight_residual import TinyFusion
from .test_spatial_ffn import pack

TASK=Path(__file__).resolve().parents[1]


def test_global_identity_coordinates_gradients_and_batch_isolation():
    old=torch.nn.Linear(3,3)
    state=torch.random.get_rng_state().clone()
    m=SpatialTokenLinear(old)
    assert torch.equal(state,torch.random.get_rng_state())
    assert m.weight is old.weight and m.bias is old.bias
    grid=torch.randn(2,3,14,14)
    x=pack(grid)
    torch.testing.assert_close(m(x),old(x),atol=0,rtol=0)
    assert sum(p.numel() for p in m.parameters())-sum(p.numel() for p in old.parameters())==6272
    torch.testing.assert_close(m.spatial_down@m.spatial_down.T,torch.eye(16),atol=1e-6,rtol=1e-6)
    m(x).square().mean().backward()
    assert m.spatial_up.grad.abs().sum()>0
    with torch.no_grad():m.spatial_up.normal_(std=.01)
    expected=grid.flatten(2)+F.linear(F.gelu(F.linear(grid.flatten(2),m.spatial_down)),m.spatial_up)
    expected=old(pack(expected.reshape_as(grid)))
    torch.testing.assert_close(m(x),expected)
    torch.testing.assert_close(m(x[:1]),m(x)[:1])
    m.zero_grad();m(x).square().mean().backward()
    assert m.spatial_down.grad.abs().sum()>0
    with pytest.raises(ValueError):m(torch.randn(2,195,3))


def test_combined_identity_transfer_and_optimizer(tmp_path):
    old=load_settings(TASK/'configs/moe_alpha40_viewreg_control.yaml')
    s=load_settings(TASK/'configs/moe_alpha40_viewreg_global16.yaml')
    source=torch.nn.Module();source.hybrid=TinyFusion(old)
    head=torch.nn.Linear(192,1)
    file=tmp_path/'source.pt'
    torch.save({'architecture':architecture_label(old),'epoch':20,'core':source.state_dict(),
                'saliency_head':head.state_dict()},file)
    target=torch.nn.Module();target.hybrid=TinyFusion(s)
    configure_spatial_ffn(target.hybrid,1);configure_global_mixing(target.hybrid,16)
    s.initialization_checkpoint,s.initialization_checkpoint_sha256=file,None
    model=SimpleNamespace(core=target,head=torch.nn.Linear(192,1),checkpoint_architecture=architecture_label(s))
    report=initialize_student(model,s)
    assert report['identity_global_mixing_added'] and report['identity_spatial_ffn_added']
    assert not report['fusion_reset']
    assert sum(p.numel() for p in target.parameters())-sum(p.numel() for p in source.parameters())==6912+12544
    x=torch.randn(2,196,192)
    kw=dict(padding_mask=torch.zeros(2,196,dtype=torch.bool),causal=False,spatial_shapes=[(1,14,14)]*2)
    for i in range(2):torch.testing.assert_close(target.hybrid.blocks[i](x,**kw),source.hybrid.blocks[i](x,**kw))
    target.phase=torch.nn.Parameter(torch.zeros(1));target.phase_parameters=lambda:[target.phase]
    target.router=torch.nn.Linear(2,2);target.readout=torch.nn.Linear(2,2)
    wrapped=torch.nn.Module();wrapped.core=target;wrapped.head=model.head
    opt=optimizer(wrapped,s)
    groups={g['name']:g for g in opt.param_groups}
    assert sum(p.numel() for p in groups['electronic_global_spatial']['params'])==12544
    staged_epoch(opt,s,1)
    assert all(p.requires_grad for p in groups['electronic_global_spatial']['params'])
    assert groups['electronic_global_spatial']['lr']==.0002
    assert all(not p.requires_grad for p in groups['electronic']['params'])
    # Loading an already-expanded checkpoint must not reset trained projections.
    with torch.no_grad():target.hybrid.blocks[0].token_pointwise.spatial_up.add_(.01)
    torch.save({'architecture':architecture_label(s),'epoch':1,'core':target.state_dict(),
                'saliency_head':model.head.state_dict()},file)
    report=initialize_student(model,s)
    assert not report['identity_global_mixing_added'] and not report['identity_spatial_ffn_added']
    assert torch.count_nonzero(target.hybrid.blocks[0].token_pointwise.spatial_up)>0


def test_global_rejects_unrelated_keys_and_profile_contract():
    with pytest.raises(RuntimeError):initialize_identity_global_mixing({'x':torch.ones(1)},{'y':torch.ones(1)})
    s=load_settings(TASK/'configs/moe_alpha40_viewreg_global16.yaml')
    base=load_settings(TASK/'configs/moe_alpha40_viewreg_cffn_kd2.yaml')
    assert architecture_label(s)==architecture_label(base)+'_global_r16'
    for key in ['initialization_checkpoint_sha256','fusion_alpha_min','student_epochs','top_k',
                'router_backend','active_size','expert_size','pixel_pitch_um','distillation_initial_weight',
                'distillation_final_weight','augmentation_apply_probability','augmentation_end_epoch',
                'language_optical_phase_zero_order_intensity_min','language_optical_phase_zero_order_intensity_max']:
        assert getattr(s,key)==getattr(base,key)
