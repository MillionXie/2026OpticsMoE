import pytest
import torch
from LightGenV2.tasks.t17_router_classification.alternating_readout import anchored_classification,refit_existing_head
from LightGenV2.tasks.t17_router_classification.f_cross_domain import validate_same_source_parent,soft_only_source_test_loss


def test_masked_test_hard_labels_have_no_effect_on_objective_or_gradient():
    torch.manual_seed(5);logits=torch.randn(4,3,requires_grad=True);teacher=torch.randn(4,3);w=torch.randn(3,5);prior=w.clone();mask=torch.tensor([True,True,False,False])
    a=anchored_classification(logits,torch.tensor([0,1,-1,-1]),w,prior,.001,.02,teacher,.5,2.,mask)
    b=anchored_classification(logits,torch.tensor([0,1,2,0]),w,prior,.001,.02,teacher,.5,2.,mask)
    assert torch.equal(a,b) and torch.equal(torch.autograd.grad(a,logits,retain_graph=True)[0],torch.autograd.grad(b,logits)[0])


def test_soft_only_loss_updates_student_but_detaches_teacher_without_gt_argument():
    logits=torch.randn(3,10,requires_grad=True);teacher=torch.randn(3,10,requires_grad=True);w=torch.randn(10,5,requires_grad=True)
    loss=soft_only_source_test_loss(logits,teacher,w,w.detach().clone(),.001,.5,2.);loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0 and teacher.grad is None


def test_same_domain_parent_must_match_dataset_and_never_accepts_joint_parent():
    cfg=dict(training_domain='RGB',dataset='eurosat_original_f',architecture='LG',main_layers=2,num_classes=10,data={},source_test_distillation=True)
    validate_same_source_parent(dict(metadata=dict(profile=dict(cfg))),cfg)
    for change in [dict(training_domain='SAR'),dict(training_domain=None),dict(data={'different':True})]:
        with pytest.raises(ValueError):validate_same_source_parent(dict(metadata=dict(profile=dict(cfg,**change))),cfg)


def test_masked_head_block_keeps_same_single_float32_head_and_reduces_mixed_objective():
    torch.manual_seed(2);head=torch.nn.Linear(5,3,bias=False);x=torch.rand(8,5);labels=torch.tensor([0,1,2,0,-1,-1,-1,-1]);mask=labels>=0;teacher=torch.randn(8,3);prior=head.weight.detach().clone()
    record=refit_existing_head(head,x,labels,prior,.001,.02,8,teacher,.5,2.,mask)
    assert record['objective_after']<=record['objective_before'] and record['hard_label_samples']==4 and not record['TRAIN_only']
    assert head.weight.dtype==torch.float32 and head.weight.shape==(3,5)
