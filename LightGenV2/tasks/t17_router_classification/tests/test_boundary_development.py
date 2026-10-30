import copy
import torch
from LightGenV2.tasks.t17_router_classification.train_boundary_development import boundary_objective,initialize_main_phases,configure_phase_optimizer,training_error_class_weights
from LightGenV2.tasks.t17_router_classification.model_four import FourRouterClassification


def test_test_ground_truth_does_not_enter_boundary_loss():
    p=torch.randn(4,8).log_softmax(1).requires_grad_();teacher=torch.randn(4,8).softmax(1);teacher[2:]=torch.tensor([.99]+[.01/7]*7)
    mask=torch.tensor([True,True,False,False]);a=torch.tensor([1,2,3,4]);b=torch.tensor([1,2,6,7])
    x=boundary_objective(p,a,mask,teacher,.1,.8,.1);y=boundary_objective(p,b,mask,teacher,.1,.8,.1)
    assert torch.equal(x,y);x.backward();assert torch.isfinite(p.grad).all() and p.grad[2:].abs().sum()>0


def test_diffuse_initialization_changes_only_main_phases_and_adds_no_parameters():
    model=FourRouterClassification('optical','ccd_grid',num_classes=8);before=model.router_phase.detach().clone()
    initialize_main_phases(model,'diffuse_uniform_phase',17)
    assert torch.equal(before,model.router_phase)
    assert set(dict(model.named_parameters()))=={'first_phase','global_phase','router_phase'}
    assert sum(p.numel() for p in model.parameters())==479364
    assert model.first_phase.std()>1 and torch.isfinite(model.global_phase).all()


def test_optional_joint_router_preserves_inference_parameters_and_lr_ratio():
    model=FourRouterClassification('optical','ccd_grid',num_classes=9)
    optimizer,parameters=configure_phase_optimizer(model,.001,dict(train_router=True,router_lr=.00005))
    assert model.router_phase.requires_grad and sum(p.numel() for p in parameters)==479364
    assert [g['lr'] for g in optimizer.param_groups]==[.001,.00005]
    optimizer,parameters=configure_phase_optimizer(model,.001,{})
    assert not model.router_phase.requires_grad and sum(p.numel() for p in parameters)==429188
    assert len(optimizer.param_groups)==1 and optimizer.param_groups[0]['lr']==.001


def test_default_frozen_training_step_is_bitwise_equal_to_original_optimizer():
    original=FourRouterClassification('optical','ccd_grid',num_classes=9)
    original.router_phase.requires_grad_(False);current=copy.deepcopy(original)
    legacy=torch.optim.Adam([original.first_phase,original.global_phase],lr=.003)
    optimizer,parameters=configure_phase_optimizer(current,.003,{})
    x=torch.rand(1,224,224);y=torch.tensor([2]);teacher=torch.nn.functional.one_hot(y,9).float()
    for model,opt,params in [(original,legacy,[original.first_phase,original.global_phase]),(current,optimizer,parameters)]:
        loss=boundary_objective(model(x)['logits'],y,torch.ones(1,dtype=torch.bool),teacher,.1,.8,.1)
        loss.backward();torch.nn.utils.clip_grad_norm_(params,5);opt.step()
    assert all(torch.equal(original.state_dict()[k],current.state_dict()[k]) for k in original.state_dict())


def test_bounded_weights_focus_low_TRAIN_recall_and_do_not_use_TEST_labels():
    import numpy as np
    weights=training_error_class_weights([[9,1],[8,2]])
    assert .5<=weights[0]<weights[1]<=2 and np.isfinite(weights).all()
    logits=torch.randn(3,2).log_softmax(1).requires_grad_();teacher=torch.tensor([[.9,.1],[.1,.9],[.1,.9]])
    mask=torch.tensor([True,False,False]);w=torch.as_tensor(weights,dtype=torch.float32)
    a=boundary_objective(logits,torch.tensor([0,0,0]),mask,teacher,.1,.8,.1,w)
    b=boundary_objective(logits,torch.tensor([0,1,1]),mask,teacher,.1,.8,.1,w)
    assert torch.equal(a,b);a.backward();assert torch.isfinite(logits.grad).all() and logits.grad[1:].abs().sum()>0
    labels=torch.tensor([0,1,1]);all_train=torch.ones(3,dtype=torch.bool)
    legacy=torch.nn.functional.cross_entropy(logits/.1,labels)+.1*torch.nn.functional.nll_loss(logits,labels)
    assert torch.equal(boundary_objective(logits,labels,all_train,teacher,.1,.8,.1),legacy)


def test_common_phase_gauge_preserves_optical_probabilities_and_relative_expert_phase():
    torch.manual_seed(17);model=FourRouterClassification('optical','ccd_grid',num_classes=9).eval()
    with torch.no_grad():
        model.first_phase.copy_(torch.randn_like(model.first_phase)*4)
        model.global_phase.copy_(torch.randn_like(model.global_phase)*4)
        original_first=model.transmission(model.first_phase).clone();router=model.router_phase.clone()
        x=torch.rand(1,224,224);before=model(x)
        diagnostics=initialize_main_phases(model,'common_phase_gauge',17);after=model(x)
        ratio=model.transmission(model.first_phase)/original_first
    assert torch.allclose(before['probabilities'],after['probabilities'],atol=1e-6,rtol=1e-5)
    assert torch.equal(before['route_power'],after['route_power']) and torch.equal(router,model.router_phase)
    assert torch.allclose(ratio,ratio.flatten()[0].expand_as(ratio),atol=2e-6,rtol=1e-5)
    assert len(diagnostics)==2 and all(d['geometric_Jacobian_gain']>=1 for d in diagnostics)
