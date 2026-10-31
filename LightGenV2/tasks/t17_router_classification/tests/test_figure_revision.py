import numpy as np
import pytest
import torch
from LightGenV2.tasks.t17_router_classification import gpu_policy
from LightGenV2.tasks.t17_router_classification.model_four import ccd_grid_energies
from LightGenV2.tasks.t17_router_classification.model_linear_readout import LinearReadoutLG
from LightGenV2.tasks.t17_router_classification.model_depth_ccd import DepthD2NN
from LightGenV2.tasks.t17_router_classification.train_figure_revision import augment_packed,configure_parameters,classification_objective
from LightGenV2.demo_check.pure_optical.models import PhaseOnly,encode


def test_new_allowlist_allows_only_explicitly_authorized_A100_without_relaxing_legacy(monkeypatch):
    inventory={'GPU-a':'NVIDIA A100-PCIE-40GB','GPU-b':'RTX4090'}
    monkeypatch.setattr(gpu_policy,'gpu_inventory',lambda:inventory);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-a')
    assert gpu_policy.require_authorized_gpu(['GPU-a'])=='GPU-a'
    with pytest.raises(ValueError,match='authorization'):gpu_policy.require_authorized_gpu(['GPU-b'])
    with pytest.raises(ValueError,match='prohibits'):gpu_policy.require_non_a100_gpu()


@pytest.mark.parametrize('classes',[8,9,10])
def test_CCD_full_coverage_all_supported_class_counts(classes):
    field=torch.rand(2,478,478);energy=ccd_grid_energies(field,classes)
    assert energy.shape==(2,classes) and torch.allclose(energy.sum(1),field.sum((-2,-1)),rtol=1e-6)


def test_ten_class_models_use_exact_same_OEO_and_only_LG_has_one_Linear():
    torch.set_num_threads(4);x=torch.rand(1,224,224)
    lg=LinearReadoutLG(2,10);d=DepthD2NN(3,10)
    assert type(lg).oeo is PhaseOnly.oeo and type(d).oeo is PhaseOnly.oeo
    for model,heads in [(lg,1),(d,0)]:
        assert sum(isinstance(m,torch.nn.Linear) for m in model.modules())==heads
        output=model(x);assert output['probabilities'].shape==(1,10)
        assert torch.allclose(output['probabilities'].sum(1),torch.ones(1))


@pytest.mark.parametrize('code',range(8))
def test_D4_augmentation_keeps_channel_tile_positions_and_zero_tile(code):
    torch.set_num_threads(4);image=torch.randint(0,256,(1,28,28,3),dtype=torch.uint8)
    expected=torch.rot90(image,code%4,(1,2))
    if code>=4:expected=expected.flip(2)
    got=augment_packed(encode(image),torch.tensor([code]))
    assert torch.allclose(got,encode(expected),rtol=1e-4,atol=1e-7)
    assert torch.count_nonzero(got[:,112:,112:])==0


@pytest.mark.parametrize('depth',[4,6])
def test_refinement_trains_only_existing_last_pair_and_single_head(depth):
    model=LinearReadoutLG(depth,8);phases,head=configure_parameters(model,dict(training_scope='last_pair_and_head',architecture='LG',main_layers=depth))
    assert len(phases)==2 and len(head)==1
    assert sum(p.numel() for p in phases)==429188 and head[0].numel()==6272
    assert not model.router_phase.requires_grad and not model.first_phase.requires_grad and not model.global_phase.requires_grad


def test_objective_has_no_test_teacher_or_domain_label_argument():
    logits=torch.randn(4,10,requires_grad=True);truth=torch.tensor([0,1,2,3]);cfg=dict(architecture='LG',label_smoothing=.02)
    loss=classification_objective(dict(raw_logits=logits),truth,cfg)
    assert torch.equal(loss,torch.nn.functional.cross_entropy(logits,truth,label_smoothing=.02))
    loss.backward();assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('scope,depth,count',[('last_pair_fixed_head',6,429188),('all_optics_fixed_head',2,479364)])
def test_mature_head_cannot_update_during_optical_refinement(scope,depth,count):
    model=LinearReadoutLG(depth,8);phases,head=configure_parameters(model,dict(training_scope=scope,architecture='LG',main_layers=depth))
    assert not head and not model.shared_head.weight.requires_grad
    assert sum(p.numel() for p in phases)==count


def test_training_scope_typo_cannot_silently_unfreeze_everything():
    with pytest.raises(ValueError,match='Unknown'):configure_parameters(LinearReadoutLG(2,8),dict(training_scope='last_typo',architecture='LG',main_layers=2))
