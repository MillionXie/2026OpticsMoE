from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t03_saliency.lightweight_residual import configure_grouped_ffn,expand_ffn_group_checkpoint
from LightGenV2.tasks.t03_saliency.modeling import architecture_label,initialize_student,optimizer
from LightGenV2.tasks.t03_saliency.settings import load_settings
from .test_ffn_widening import source_core

TASK=Path(__file__).resolve().parents[1]


def test_grouped_preserves_function_rng_and_learns_cross_channel_weights():
    s=load_settings(TASK/'configs/moe_alpha40_sam005.yaml')
    old=source_core(s).eval();new=deepcopy(old)
    rng=torch.random.get_rng_state().clone();configure_grouped_ffn(new.hybrid)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())==34560
    x=torch.randn(2,196,192)
    kw=dict(padding_mask=torch.zeros(2,196,dtype=torch.bool),causal=False,spatial_shapes=[(1,14,14)]*2)
    for i in range(2):
        a,b=old.hybrid.blocks[i],new.hybrid.blocks[i]
        torch.testing.assert_close(b(x,**kw),a(x,**kw),atol=2e-6,rtol=2e-6)
        conv=b.mlp[1][0].conv
        assert conv.groups==64 and tuple(conv.weight.shape)==(384,6,3,3)
        b(x,**kw).square().mean().backward()
        mask=torch.ones_like(conv.weight,dtype=torch.bool)
        rows=torch.arange(384);mask[rows,rows%6]=False
        assert torch.count_nonzero(conv.weight[mask])==0
        assert torch.isfinite(conv.weight.grad).all() and conv.weight.grad[mask].abs().sum()>0
        # Batch isolation: no sample may influence another sample's result.
        single_kw=dict(padding_mask=kw['padding_mask'][:1],causal=False,spatial_shapes=[(1,14,14)])
        torch.testing.assert_close(b(x[:1],**single_kw),b(x,**kw)[:1])
    for k,v in old.state_dict().items():
        if v.shape==new.state_dict()[k].shape:assert torch.equal(v,new.state_dict()[k])


def test_strict_transfer_reload_and_optimizer(tmp_path):
    base=load_settings(TASK/'configs/moe_alpha40_sam005.yaml')
    s=load_settings(TASK/'configs/moe_alpha40_sam_group64.yaml')
    old=source_core(base);new=deepcopy(old);configure_grouped_ffn(new.hybrid)
    head=torch.nn.Linear(192,1);path=tmp_path/'source.pt'
    torch.save(dict(architecture=architecture_label(base),epoch=65,core=old.state_dict(),saliency_head=head.state_dict()),path)
    s.initialization_checkpoint,s.initialization_checkpoint_sha256=path,None
    model=SimpleNamespace(core=new,head=deepcopy(head),checkpoint_architecture=architecture_label(s))
    report=initialize_student(model,s)
    assert report['ffn_grouped64_transfer'] and not report['fusion_reset']
    with torch.no_grad():new.hybrid.blocks[0].mlp[1][0].conv.weight[0,1].add_(.05)
    expected=deepcopy(new.state_dict())
    torch.save(dict(architecture=architecture_label(s),epoch=1,core=expected,saliency_head=head.state_dict()),path)
    report=initialize_student(model,s)
    assert not report['ffn_grouped64_transfer']
    for k,v in new.state_dict().items():assert torch.equal(v,expected[k])
    new.phase=torch.nn.Parameter(torch.zeros(1));new.phase_parameters=lambda:[new.phase]
    new.router=torch.nn.Linear(2,2);new.readout=torch.nn.Linear(2,2)
    wrapped=torch.nn.Module();wrapped.core=new;wrapped.head=head
    groups={g['name']:g for g in optimizer(wrapped,s).param_groups}
    assert sum(p.numel() for p in groups['electronic_ffn_spatial']['params'])==41472
    assert groups['electronic_ffn_spatial']['lr']==5e-5


def test_group_transfer_rejects_other_changes_and_config_stays_paired():
    s=load_settings(TASK/'configs/moe_alpha40_sam_group64.yaml')
    base=load_settings(TASK/'configs/moe_alpha40_sam005.yaml')
    assert architecture_label(s)==architecture_label(base)+'_cffn_g64'
    for k in ['student_epochs','sam_rho','initialization_checkpoint_sha256','augmentation_enabled',
              'distillation_loss','distillation_initial_weight','distillation_final_weight','fusion_alpha_min',
              'top_k','router_backend','active_size','expert_size','electronic_ffn_hidden_width',
              'language_optical_phase_zero_order_intensity_min','language_optical_phase_zero_order_intensity_max']:
        assert getattr(s,k)==getattr(base,k)
    old=source_core(base);new=deepcopy(old);configure_grouped_ffn(new.hybrid)
    a,b=old.state_dict(),new.state_dict()
    with pytest.raises(RuntimeError):expand_ffn_group_checkpoint(a,{**b,'alien':torch.ones(1)})
    bad=dict(b);bad['hybrid.blocks.0.mlp.3.bias']=torch.ones(193)
    with pytest.raises(RuntimeError):expand_ffn_group_checkpoint(a,bad)
