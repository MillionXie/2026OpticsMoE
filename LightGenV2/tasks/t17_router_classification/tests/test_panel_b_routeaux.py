import pytest
import torch
from LightGenV2.tasks.t17_router_classification.model_depth_ccd import DepthLG
from LightGenV2.tasks.t17_router_classification.refine_panel_b_depth import configure_trainable, refinement_objective
from LightGenV2.tasks.t17_router_classification.train_boundary_development import boundary_objective, class_pair_route_targets
from LightGenV2.tasks.t17_router_classification.phase_optimizer import WrappedPhaseAdam


CFG=dict(training_temperature=.1,teacher_confidence=.8,energy_nll_weight=.1,focal_gamma=2.,route_aux_weight=.1)


@pytest.mark.parametrize('depth',[4,6])
@pytest.mark.parametrize('mode',['joint_router_control','joint_dense_pair_aux'])
def test_joint_phase_updates_preserve_canonical_model_and_gradient_support(depth,mode):
    torch.set_num_threads(4);torch.manual_seed(17);model=DepthLG(depth,8)
    parameters=configure_trainable(model,mode)
    assert all(p.requires_grad for p in model.parameters())
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    out=model(torch.rand(1,224,224),return_dense_route=True)
    y=torch.tensor([2]);original=torch.tensor([True]);teacher=torch.nn.functional.one_hot(y,8).float()
    targets=torch.tensor(class_pair_route_targets([437]*8),dtype=torch.float32)
    loss,aux=refinement_objective(out,y,original,teacher,CFG,mode,targets)
    dense_gradient,=torch.autograd.grad(loss,out['dense_route_power'],retain_graph=True)
    excluded=out['route_power']==0
    if mode=='joint_router_control':assert torch.count_nonzero(dense_gradient[excluded])==0 and aux==0
    else:assert torch.count_nonzero(dense_gradient[excluded])>0 and aux>0
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    WrappedPhaseAdam(parameters,.001).step()
    assert all(not torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert not any(isinstance(m,(torch.nn.Linear,torch.nn.Conv2d)) for m in model.modules())


def test_zero_auxiliary_retains_exact_existing_frozen_classification_loss():
    torch.manual_seed(17);logits=torch.rand(3,8).softmax(1).log();y=torch.tensor([1,2,3])
    teacher=torch.nn.functional.one_hot(y,8).float();original=torch.tensor([True,True,False])
    expected=boundary_objective(logits,y,original,teacher,.1,.8,.1,focal_gamma=2.)
    actual,aux=refinement_objective(dict(logits=logits),y,original,teacher,CFG,'frozen_router_control')
    assert torch.equal(actual,expected) and aux==0


def test_teacher_pseudo_labels_exclude_original_TEST_GT_from_both_losses():
    logits=torch.rand(3,8).softmax(1).log();dense=torch.rand(3,4).softmax(1)
    teacher=torch.nn.functional.one_hot(torch.tensor([1,2,3]),8).float();original=torch.zeros(3,dtype=torch.bool)
    targets=torch.tensor(class_pair_route_targets([437]*8),dtype=torch.float32)
    out=dict(logits=logits,dense_route_power=dense)
    first=refinement_objective(out,torch.tensor([1,2,3]),original,teacher,CFG,'joint_dense_pair_aux',targets)
    second=refinement_objective(out,torch.tensor([5,6,7]),original,teacher,CFG,'joint_dense_pair_aux',targets)
    assert all(torch.equal(a,b) for a,b in zip(first,second))
