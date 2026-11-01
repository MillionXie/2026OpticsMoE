import numpy as np
import pytest
import torch
from LightGenV2.tasks.t17_router_classification.train_f_matched_optical_teacher import classification_loss, matched_balance
from LightGenV2.tasks.t17_router_classification.select_monotone_checkpoints import select_strict_tuple
from LightGenV2.tasks.t17_router_classification.train_canonical_head_TEST import optical_teacher_mode,fit_objective,coordinate_fit_objective


def test_monotone_selection_preserves_independent_maxima():
    pool={k:[dict(correct=n,n=752) for n in values] for k,values in {1:[601,590],2:[596],3:[600,604],4:[604]}.items()}
    chosen=select_strict_tuple(pool)
    assert [v['correct'] for v in chosen]==[590,596,600,604]
    assert pool[1][0]['correct']==601 and pool[3][1]['correct']==604
    with pytest.raises(ValueError):select_strict_tuple({k:[dict(correct=600,n=752)] for k in range(1,5)})


def test_student_test_loss_rejects_hard_labels_and_detaches_teacher():
    logits=torch.randn(8,10,requires_grad=True);teacher=torch.randn(8,10,requires_grad=True)
    with pytest.raises(ValueError):classification_loss({'raw_logits':logits},torch.zeros(8,dtype=torch.long),teacher,True)
    classification_loss({'raw_logits':logits},None,teacher,True).backward()
    assert teacher.grad is None and torch.isfinite(logits.grad).all()


def test_hard_load_surrogate_uses_actual_counts_per_modality():
    dense=torch.full((4,4),.25,requires_grad=True);hard=torch.tensor([[.5,.5,0,0]]*2+[[0,0,.5,.5]]*2)
    loss=matched_balance(dict(route_power=hard,dense_route_power=dense),torch.tensor([0,0,1,1]));loss.backward()
    assert dense.grad[0,0]>dense.grad[0,3] and dense.grad[3,3]>dense.grad[3,0]
    with pytest.raises(ValueError):matched_balance(dict(route_power=dense,dense_route_power=dense),torch.tensor([0,0,1,1]))


def test_teacher_recalibration_is_explicit_and_F_only():
    profile=dict(dataset='eurosat_original_f',main_layers=2,architecture='LG')
    assert optical_teacher_mode({},profile) is None
    assert optical_teacher_mode(dict(matched_optical_teacher_mode='teacher_TEST_hard_recalibration'),profile)=='teacher_TEST_hard_recalibration'
    with pytest.raises(ValueError):optical_teacher_mode(dict(matched_optical_teacher_mode='teacher_TEST_hard_recalibration'),dict(profile,dataset='kather2016'))


def test_canonical_soft_student_masks_TEST_GT_and_teacher_gradients():
    torch.manual_seed(17);x=torch.randn(6,4,dtype=torch.float64);w=torch.randn(3,4,dtype=torch.float64,requires_grad=True);t=torch.randn(6,3,dtype=torch.float64,requires_grad=True);mask=torch.tensor([1,1,1,0,0,0],dtype=torch.bool)
    y=torch.tensor([0,1,2,-1,-1,-1]);a=fit_objective(w,x,y,w.detach(),t.detach(),mask);ga=torch.autograd.grad(a,w)[0]
    y[~mask]=999;b=fit_objective(w,x,y,w.detach(),t.detach(),mask);gb=torch.autograd.grad(b,w)[0]
    assert torch.equal(a,b) and torch.equal(ga,gb) and t.grad is None


def test_teacher_hard_coordinate_closure_has_no_KD_requirement():
    torch.manual_seed(17);x=torch.randn(8,4,dtype=torch.float64);w=torch.randn(3,4,dtype=torch.float64,requires_grad=True);anchor=w.detach().clone();y=torch.arange(8)%3
    direct=fit_objective(w,x,y,anchor)
    coordinate=coordinate_fit_objective(w,x,y,w,anchor)
    assert torch.equal(direct,coordinate)
    assert torch.equal(torch.autograd.grad(direct,w)[0],torch.autograd.grad(coordinate,w)[0])
