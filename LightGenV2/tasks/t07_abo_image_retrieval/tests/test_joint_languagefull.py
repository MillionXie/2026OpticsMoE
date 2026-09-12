"""Existing full-field electronic decoder under the pinned joint recipe."""
import pytest
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config, apply_contract, PINNED_TEACHER_PROFILES
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue import PINNED_TEACHER_PROFILES as QUEUE
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.optics import OpticalPath


def test_only_language_pooling_contract_changes_from_joint_restart():
    base=overlay_config({},'domain_distill_joint_restart')
    candidate=overlay_config({},'domain_distill_joint_languagefull')
    assert candidate['ccd_readout_modes']=={'vision':'prefix_rows','language':'fullfield_rows'}
    assert base['ccd_readout_modes']=={'vision':'prefix_rows','language':'prefix_rows'}
    for cfg in (base,candidate):
        cfg.pop('protocol');cfg.pop('ccd_readout_modes')
    assert base==candidate
    assert 'domain_distill_joint_languagefull' in PINNED_TEACHER_PROFILES
    assert 'domain_distill_joint_languagefull' in QUEUE


def test_metadata_only_conversion_keeps_all_weight_values_and_source_unchanged():
    cfg=overlay_config({},'domain_distill_joint_languagefull')
    p={'metadata':{'fusion_alpha_min':.4001},'state_dict':{'phase':torch.randn(3,3)}}
    result=apply_contract(p,cfg)
    assert result['state_dict'] is p['state_dict']
    assert p['metadata']=={'fusion_alpha_min':.4001}
    assert result['metadata']['ccd_readout_modes']['language']=='fullfield_rows'


@pytest.mark.parametrize('final',[False,True])
def test_fullfield_decode_matches_explicit_full_sensor_pooling_and_backprop(final):
    torch.set_num_threads(2);torch.manual_seed(42)
    optics=OpticalPath();optics.readout_mode='fullfield_rows'
    names={k:tuple(v.shape) for k,v in optics.state_dict().items()}
    raw=torch.rand(2,478,478,requires_grad=True)
    value=raw.float().clamp_min(0)
    relative=(value/value.mean((-2,-1),keepdim=True).clamp_min(1e-6)).clamp_max(12)
    rows=F.adaptive_avg_pool2d(torch.log1p(relative)[:,None],(77,224))[:,0]
    rows=F.relu(F.layer_norm(rows,(224,),eps=1e-5))
    output=optics.global_output if final else optics.expert_output
    expected=output(rows.reshape(-1,224)).reshape(2,77,192)
    actual=optics.decode(raw,77,torch.float32,final)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.square().mean().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[:,400:].abs().sum()>0
    assert actual.shape==(2,77,192)
    assert names=={k:tuple(v.shape) for k,v in optics.state_dict().items()}


def test_default_prefix_decode_remains_the_original_formula():
    torch.manual_seed(1);optics=OpticalPath();raw=torch.rand(1,478,478)
    value=(raw/raw.mean((-2,-1),keepdim=True).clamp_min(1e-6)).clamp_max(12)
    rows=F.adaptive_avg_pool2d(torch.log1p(value)[:,None],(224,224))[:,0]
    rows=F.relu(F.layer_norm(rows,(224,),eps=1e-5))[:,:77]
    expected=optics.expert_output(rows.reshape(-1,224)).reshape(1,77,192)
    torch.testing.assert_close(optics.decode(raw,77,torch.float32,False),expected,rtol=0,atol=0)
