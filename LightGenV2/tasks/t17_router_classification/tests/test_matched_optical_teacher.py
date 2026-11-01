import numpy as np
import pytest
import torch
from LightGenV2.tasks.t17_router_classification.train_f_matched_optical_teacher import classification_loss, matched_balance
from LightGenV2.tasks.t17_router_classification.select_monotone_checkpoints import select_strict_tuple


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
