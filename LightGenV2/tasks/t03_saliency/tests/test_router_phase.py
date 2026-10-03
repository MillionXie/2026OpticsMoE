from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
from LightGenV2.tasks.t03_saliency.router_phase import (
    RadiansOpticalRouter,ROUTER_KEY,RADIANS_SUFFIX,checkpoint_phase,convert_router_checkpoint)
from LightGenV2.tasks.t03_saliency.modeling import architecture_label,initialize_student
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.training import phase_change_report
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.tests.test_router import (
    _TinyGeometry,_settings,_input_fields,OpticalDetectorTopKRouter)
from experiments.hardware_sdk.workflows.reconstruct_slm import encode_active_phase

TASK=Path(__file__).resolve().parents[1]


def test_initial_phase_detector_routing_and_hardware_codes_are_identical():
    old=OpticalDetectorTopKRouter(_TinyGeometry(),_settings()).eval()
    state=torch.get_rng_state().clone()
    new=RadiansOpticalRouter(_TinyGeometry(),_settings()).eval()
    assert torch.equal(state,torch.get_rng_state())
    assert sum(p.numel() for p in new.parameters())==sum(p.numel() for p in old.parameters())
    assert new.state_dict().keys()==old.state_dict().keys()
    torch.testing.assert_close(new.phase(),old.phase(),atol=0,rtol=0)
    torch.testing.assert_close(new.active_phase(),old.active_phase(),atol=0,rtol=0)
    np.testing.assert_array_equal(encode_active_phase(new.active_phase().detach().numpy()),
                                  encode_active_phase(old.active_phase().detach().numpy()))
    a,b=old(_input_fields()),new(_input_fields())
    for k in a:
        if torch.is_tensor(a[k]):torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
    torch.testing.assert_close(old.last_detector_intensity,new.last_detector_intensity,atol=0,rtol=0)
    weights=torch.arange(1,5,dtype=torch.float32)
    (a['probabilities']*weights).sum().backward()
    (b['probabilities']*weights).sum().backward()
    jacobian=2*math.pi*old.raw_router_phase.detach().sigmoid()*(1-old.raw_router_phase.detach().sigmoid())
    torch.testing.assert_close(old.raw_router_phase.grad,new.raw_router_phase.grad*jacobian,atol=1e-7,rtol=1e-4)
    assert new.raw_router_phase.grad.norm()>0


def test_periodic_radians_export_and_checkpoint_conversion():
    old={ROUTER_KEY:torch.tensor([-10.,0.,10.]),'other':torch.ones(2)}
    target={k:torch.zeros_like(v) for k,v in old.items()}
    new,changed=convert_router_checkpoint(old,target,'old','old'+RADIANS_SUFFIX)
    assert changed and new['other'] is old['other']
    torch.testing.assert_close(new[ROUTER_KEY],checkpoint_phase(ROUTER_KEY,old[ROUTER_KEY],'old'),atol=0,rtol=0)
    same,changed=convert_router_checkpoint(new,target,'old'+RADIANS_SUFFIX,'old'+RADIANS_SUFFIX)
    assert same is new and not changed
    # No second sigmoid during reload or export. Values can cross either endpoint.
    raw=torch.tensor([[-.5,2*math.pi+.5]])
    decoded=checkpoint_phase(ROUTER_KEY,raw,'old'+RADIANS_SUFFIX)
    assert torch.equal(decoded,raw)
    np.testing.assert_array_equal(encode_active_phase(decoded.numpy()),np.array([[235,20]],dtype=np.uint8))
    for source_arch in ['unrelated','old_router_radians_extra']:
        with pytest.raises(RuntimeError):convert_router_checkpoint(old,target,source_arch,'old'+RADIANS_SUFFIX)
    with pytest.raises(RuntimeError):convert_router_checkpoint({},target,'old','old'+RADIANS_SUFFIX)


def test_real_warmstart_metadata_and_phase_audit(tmp_path):
    old_settings=load_settings(TASK/'configs/moe_alpha40_extra_control.yaml')
    s=load_settings(TASK/'configs/moe_alpha40_router_radians.yaml')
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__();self.hybrid=torch.nn.Module();self.hybrid.optical_branch=torch.nn.Module()
            self.hybrid.optical_branch.core=torch.nn.Module()
            self.hybrid.optical_branch.core.router=torch.nn.Module()
            self.hybrid.optical_branch.core.router.raw_router_phase=torch.nn.Parameter(torch.tensor([-10.,0.,10.]))
    core=Core();head=torch.nn.Linear(1,1)
    old=dict(architecture=architecture_label(old_settings),epoch=5,core=core.state_dict(),saliency_head=head.state_dict())
    p=tmp_path/'old.pt';torch.save(old,p)
    s.initialization_checkpoint=p;s.initialization_checkpoint_sha256=None
    m=SimpleNamespace(core=Core(),head=torch.nn.Linear(1,1),checkpoint_architecture=architecture_label(s))
    report=initialize_student(m,s)
    assert report['router_phase_sigmoid_to_radians'] and not report['fusion_reset']
    new=dict(architecture=architecture_label(s),core=m.core.state_dict())
    audit=phase_change_report(new,s)[ROUTER_KEY]
    assert audit['raw_rms_change'] is None and audit['circular_phase_rms_rad']==0
    torch.save(dict(new,epoch=0,saliency_head=m.head.state_dict()),p)
    # A checkpoint already in radians must not be converted again.
    assert not initialize_student(m,s)['router_phase_sigmoid_to_radians']


def test_profile_is_single_coordinate_change_without_new_training_data():
    a=load_settings(TASK/'configs/moe_alpha40_extra_control.yaml')
    b=load_settings(TASK/'configs/moe_alpha40_router_radians.yaml')
    allowed={'config','config_path','output_dir','router_phase_coordinates','convert_router_phase_on_warmstart'}
    assert {k for k in vars(a) if getattr(a,k)!=getattr(b,k)}<=allowed
    assert architecture_label(b)==architecture_label(a)+RADIANS_SUFFIX
    assert b.unlabeled_weight==b.semantic_weight==0 and b.fusion_alpha_min==.4 and b.top_k==2
    assert b.phase_weight_decay==0 and b.router_backend=='optical'
