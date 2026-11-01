import numpy as np
import torch
from LightGenV2.tasks.t17_router_classification.f_router_load_recovery import router_load_objective,replay_order,load_key


def test_error_replay_keeps_every_case_and_repeats_only_fixed_errors():
    order=replay_order(np.random.default_rng(17),[0,1,0,1,0])
    assert np.array_equal(np.bincount(order,minlength=5),[1,3,1,3,1])


def test_modality_balancing_has_finite_gradient_for_unused_port():
    logits=torch.tensor([[1.,2.,5.,-6.]]*6,requires_grad=True);q=logits.softmax(1)
    loss=router_load_objective(q,torch.tensor([0,0,0,1,1,1]));loss.backward()
    assert torch.isfinite(logits.grad).all() and bool((logits.grad[:,3]<0).all())


def test_actual_hard_load_and_both_modalities_determine_warmup_selection():
    a={s:dict(unused_experts=[4],selection_share_CV=.5)for s in ('RGB','SAR')}
    b={s:dict(unused_experts=[],selection_share_CV=.2)for s in ('RGB','SAR')}
    assert load_key(b)<load_key(a)
    b['SAR']['unused_experts']=[4];assert load_key(b)[0]==1
