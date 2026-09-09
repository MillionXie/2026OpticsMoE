from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t03_saliency.lightweight_residual import (
    configure_spatial_ffn, configure_wide_ffn, widen_ffn_checkpoint)
from LightGenV2.tasks.t03_saliency.modeling import architecture_label, initialize_student, optimizer
from LightGenV2.tasks.t03_saliency.settings import load_settings
from .test_lightweight_residual import TinyFusion

TASK=Path(__file__).resolve().parents[1]


def source_core(settings):
    core=torch.nn.Module();core.hybrid=TinyFusion(settings)
    configure_spatial_ffn(core.hybrid,1)
    # A trained, nonidentity spatial kernel must also be duplicated correctly.
    with torch.no_grad():
        for b in core.hybrid.blocks:b.mlp[1][0].conv.weight.add_(torch.randn_like(b.mlp[1][0].conv.weight)*.02)
    return core


def test_real_residual_function_budget_rng_and_symmetry_breaking():
    settings=load_settings(TASK/'configs/moe_alpha40_sam_wide576.yaml')
    old=source_core(settings).eval();new=deepcopy(old)
    rng=torch.random.get_rng_state().clone()
    configure_wide_ffn(new.hybrid)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())==151296
    x=torch.randn(2,196,192)
    kw=dict(padding_mask=torch.zeros(2,196,dtype=torch.bool),causal=False,spatial_shapes=[(1,14,14)]*2)
    for i in range(2):
        a,b=old.hybrid.blocks[i],new.hybrid.blocks[i]
        b.eval() # newly constructed modules default training=True
        torch.testing.assert_close(b(x,**kw),a(x,**kw),atol=2e-6,rtol=2e-6)
        b(x,**kw).square().mean().backward()
        grad=b.mlp[0].weight.grad
        assert torch.isfinite(grad).all() and grad.abs().sum()>0
        assert not torch.equal(grad[:192],grad[384:])
        torch.testing.assert_close(grad[:192]*1.5,grad[384:],atol=1e-8,rtol=1e-4)
    # Tensor names not involved in widening are retained exactly.
    for key,value in old.state_dict().items():
        other=new.state_dict()[key]
        if value.shape==other.shape:assert torch.equal(value,other)


def test_strict_warmstart_reload_and_optimizer(tmp_path):
    base=load_settings(TASK/'configs/moe_alpha40_sam005.yaml')
    s=load_settings(TASK/'configs/moe_alpha40_sam_wide576.yaml')
    old=source_core(base);new=deepcopy(old);configure_wide_ffn(new.hybrid)
    head=torch.nn.Linear(192,1);path=tmp_path/'source.pt'
    torch.save(dict(architecture=architecture_label(base),epoch=65,core=old.state_dict(),saliency_head=head.state_dict()),path)
    s.initialization_checkpoint,s.initialization_checkpoint_sha256=path,None
    model=SimpleNamespace(core=new,head=deepcopy(head),checkpoint_architecture=architecture_label(s))
    report=initialize_student(model,s)
    assert report['ffn_widened_384_to_576'] and not report['fusion_reset']
    with torch.no_grad():new.hybrid.blocks[0].mlp[0].weight[500].add_(.03)
    expected=deepcopy(new.state_dict())
    torch.save(dict(architecture=architecture_label(s),epoch=1,core=expected,saliency_head=head.state_dict()),path)
    report=initialize_student(model,s)
    assert not report['ffn_widened_384_to_576']
    for k,v in new.state_dict().items():assert torch.equal(v,expected[k])
    new.phase=torch.nn.Parameter(torch.zeros(1));new.phase_parameters=lambda:[new.phase]
    new.router=torch.nn.Linear(2,2);new.readout=torch.nn.Linear(2,2)
    wrapped=torch.nn.Module();wrapped.core=new;wrapped.head=head
    groups={g['name']:g for g in optimizer(wrapped,s).param_groups}
    assert sum(p.numel() for p in groups['electronic_ffn_spatial']['params'])==10368
    ids=[id(p) for g in groups.values() for p in g['params']]
    assert len(ids)==len(set(ids))


def test_widening_rejects_other_changes():
    s=load_settings(TASK/'configs/moe_alpha40_sam_wide576.yaml')
    old=source_core(s);new=deepcopy(old);configure_wide_ffn(new.hybrid)
    src,tgt=old.state_dict(),new.state_dict()
    with pytest.raises(RuntimeError):widen_ffn_checkpoint(src,{**tgt,'alien':torch.zeros(1)})
    bad=dict(tgt);bad['hybrid.blocks.0.mlp.3.bias']=torch.zeros(193)
    with pytest.raises(RuntimeError):widen_ffn_checkpoint(src,bad)


def test_single_variable_profile_contract():
    base=load_settings(TASK/'configs/moe_alpha40_sam005.yaml')
    s=load_settings(TASK/'configs/moe_alpha40_sam_wide576.yaml')
    assert architecture_label(s)==architecture_label(base)+'_ffn576'
    assert s.widen_ffn_on_warmstart and s.electronic_ffn_hidden_width==576
    for key in ['sam_rho','initialization_checkpoint_sha256','student_epochs','fusion_alpha_min',
                'active_size','expert_size','pixel_pitch_um','router_backend','top_k','electronic_width',
                'augmentation_enabled','distillation_initial_weight','distillation_final_weight',
                'language_optical_phase_zero_order_intensity_min','language_optical_phase_zero_order_intensity_max']:
        assert getattr(s,key)==getattr(base,key)
