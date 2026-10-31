import pytest
import torch
from LightGenV2.tasks.t17_router_classification.semantic_distillation import semantic_loss,ridge_projection,balanced_aux_weight,semantic_checkpoint
from LightGenV2.tasks.t17_router_classification.model_linear_readout import LinearReadoutLG


def test_feature_auxiliary_detaches_teacher_but_updates_features_and_separate_projector():
    torch.manual_seed(3);x=torch.randn(8,5,requires_grad=True);target=torch.randn(8,3,requires_grad=True);p=torch.nn.Linear(5,3,bias=False)
    semantic_loss(x,target,p).backward();assert x.grad.abs().sum()>0 and p.weight.grad.abs().sum()>0 and target.grad is None


def test_affine_ridge_map_is_folded_into_same_mean_one_feature_coordinates():
    torch.manual_seed(4);x=torch.rand(40,5);x=x/x.mean(1,keepdim=True);truth=torch.randn(3,5);bias=torch.randn(3);y=x@truth.T+bias
    weight,record=ridge_projection(x,y,ridge_ratio=1e-6)
    assert torch.allclose(x@weight.T,y,atol=2e-4,rtol=2e-4) and record['affine_intercept_absorbed_in_mean_one_features']
    with pytest.raises(ValueError):ridge_projection(x*2,y)


def test_calibration_is_train_gradient_rule_and_projection_never_enters_model_state():
    assert balanced_aux_weight(1.,2.)==pytest.approx(.1)
    m=LinearReadoutLG(2,10);before=set(m.state_dict());p=torch.nn.Linear(784,256,bias=False);opt=torch.optim.Adam(p.parameters(),lr=.0001);aux=dict(projection=p,optimizer=opt)
    ck=semantic_checkpoint(aux);assert set(m.state_dict())==before and sum(isinstance(v,torch.nn.Linear) for v in m.modules())==1
    assert 'training_only_semantic_projection' in ck and p.weight.numel()==200704 and semantic_checkpoint(None)=={}


def test_actual_canonical_phase_receives_auxiliary_gradient_without_new_inference_module():
    torch.set_num_threads(4);torch.manual_seed(17);m=LinearReadoutLG(2,10);p=torch.nn.Linear(784,256,bias=False);result=m(torch.rand(2,224,224));semantic_loss(result['ccd_features'],torch.randn(2,256),p).backward()
    assert m.first_phase.grad is not None and torch.isfinite(m.first_phase.grad).all() and m.first_phase.grad.abs().sum()>0
    assert m.global_phase.grad.abs().sum()>0 and 'training_only_semantic_projection' not in m.state_dict()
