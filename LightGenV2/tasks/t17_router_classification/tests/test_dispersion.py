import torch
from LightGenV2.tasks.t17_router_classification.train_regularized import router_balance_loss,candidate_key,update_ema

def test_top2_balance_moves_logits_toward_unused_experts():
    logits=torch.tensor([[0.,0.,2.,1.5]]*8,requires_grad=True)
    loss=router_balance_loss(logits.softmax(1),'top2_load')
    loss.backward()
    assert (logits.grad[:,:2]<0).all()
    assert (logits.grad[:,2:]>0).all()

def test_equal_batch_load_has_no_balance_gradient():
    logits=torch.tensor([[2.,1.,0.,0.],[0.,0.,2.,1.]],requires_grad=True)
    router_balance_loss(logits.softmax(1),'top2_load').backward()
    assert logits.grad.abs().max()<1e-7

def test_accuracy_floor_prevents_diverse_but_bad_selection():
    def metrics(score,effective,pair):
        return dict(accuracy=score,balanced_accuracy=score,cross_entropy=.5,
            router=dict(effective_experts=effective,dominant_pair_fraction=pair))
    parent=metrics(.81,2.,1.)
    acceptable=metrics(.805,3.5,.4)
    failed=metrics(.79,4.,.2)
    assert candidate_key(acceptable,.81,.01)>candidate_key(parent,.81,.01)
    assert candidate_key(failed,.81,.01)<candidate_key(parent,.81,.01)
    assert candidate_key(acceptable,.81)<candidate_key(parent,.81)

def test_ema_preserves_frozen_router_exactly():
    from copy import deepcopy
    layer=torch.nn.Linear(4,4)
    layer.requires_grad_(False)
    average=deepcopy(layer)
    for _ in range(100):update_ema(average,layer,.99)
    assert all(torch.equal(a,b) for a,b in zip(layer.parameters(),average.parameters()))
