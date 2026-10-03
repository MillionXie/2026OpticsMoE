from pathlib import Path
from copy import deepcopy
import pytest
import torch
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from LightGenV2.tasks.t03_saliency.fusion_training import differentiable_fusion,enable_exact_fusion_backward
from .test_lightweight_residual import TinyFusion

TASK=Path(__file__).resolve().parents[1]


def setup_fusion():
    s=load_settings(TASK/'configs/moe_alpha40_sam_exactfusion.yaml')
    a=TinyFusion(s)
    torch.manual_seed(120)
    e=torch.randn(2,7,8,requires_grad=True)
    o=torch.randn_like(e,requires_grad=True)
    alpha=torch.tensor(.43,requires_grad=True)
    mask=torch.zeros(2,7,dtype=torch.bool);mask[0,-2:]=True
    return a,e,o,alpha,mask


def test_forward_state_parameters_and_rng_unchanged_backward_is_exact():
    a,e,o,alpha,mask=setup_fusion()
    state={k:v.clone() for k,v in a.state_dict().items()}
    ref=a._fuse(e,o,alpha,mask,'test')
    probe=torch.randn_like(ref)
    old_grad=torch.autograd.grad((ref*probe).sum(),(e,o,alpha))
    # RNG snapshot immediately before installing the training-only method.
    rng=torch.get_rng_state().clone();enable_exact_fusion_backward(a)
    assert torch.equal(rng,torch.get_rng_state())
    assert a.state_dict().keys()==state.keys()
    for k,v in a.state_dict().items():assert torch.equal(v,state[k])
    actual=a._fuse(e,o,alpha,mask,'test')
    assert torch.equal(actual,ref)
    grads=torch.autograd.grad((actual*probe).sum(),(e,o,alpha))
    exact=differentiable_fusion(e,o,alpha,mask,a.fusion_rms_epsilon)
    expected=torch.autograd.grad((exact*probe).sum(),(e,o,alpha))
    for g,h in zip(grads,expected):torch.testing.assert_close(g,h,atol=0,rtol=0)
    assert not torch.allclose(grads[1],old_grad[1])
    # Per-sample optical scale is a radial null direction of normalized O.
    torch.testing.assert_close((grads[1]*o).sum(),torch.tensor(0.),atol=3e-5,rtol=0)
    a.eval();assert torch.equal(a._fuse(e,o,alpha,mask,'test'),ref)
    # A copied module must never call a bound method of the original object.
    copied=deepcopy(a);copied.set_fusion_ablation('remove_optical')
    assert torch.equal(copied._fuse(e,None,alpha,mask,'test'),e.masked_fill(mask.unsqueeze(-1),0))
    assert a.fusion_ablation_mode=='none'
    with pytest.raises(ValueError):enable_exact_fusion_backward(a)


def test_directional_derivative_matches_forward_finite_difference():
    a,e,o,alpha,mask=setup_fusion();enable_exact_fusion_backward(a)
    probe=torch.randn_like(e);direction=torch.randn_like(o)
    value=a._fuse(e,o,alpha,mask,'test')
    grad=torch.autograd.grad((value*probe).sum(),o)[0]
    h=.002
    with torch.no_grad():
        plus=(a._fuse(e,o+h*direction,alpha,mask,'test')*probe).sum()
        minus=(a._fuse(e,o-h*direction,alpha,mask,'test')*probe).sum()
    torch.testing.assert_close((grad*direction).sum(),(plus-minus)/(2*h),atol=.003,rtol=.003)


@pytest.mark.parametrize('all_padding',[False,True])
def test_zero_inputs_padding_and_ablations_are_finite(all_padding):
    a,e,o,alpha,mask=setup_fusion();enable_exact_fusion_backward(a)
    e=torch.zeros_like(e,requires_grad=True);o=torch.zeros_like(o,requires_grad=True)
    mask.fill_(all_padding)
    value=a._fuse(e,o,alpha,mask,'test')
    grads=torch.autograd.grad(value.sum(),(e,o,alpha))
    assert torch.isfinite(value).all() and all(torch.isfinite(g).all() for g in grads)
    a.set_fusion_ablation('remove_optical')
    torch.testing.assert_close(a._fuse(e,None,alpha,mask,'test'),e)


def test_profile_preserves_inference_and_training_controls():
    s=load_settings(TASK/'configs/moe_alpha40_sam_exactfusion.yaml')
    base=load_settings(TASK/'configs/moe_alpha40_sam005.yaml')
    assert s.exact_fusion_backward and not base.exact_fusion_backward
    assert architecture_label(s)==architecture_label(base)
    for k in ['student_epochs','sam_rho','initialization_checkpoint_sha256','augmentation_enabled',
              'distillation_initial_weight','distillation_final_weight','distillation_loss','fusion_mode',
              'fusion_alpha_min','top_k','router_backend','active_size','expert_size',
              'electronic_ffn_hidden_width','electronic_ffn_groups','kl_weight','cc_weight','nss_weight']:
        assert getattr(s,k)==getattr(base,k)
