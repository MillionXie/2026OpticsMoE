import pytest
import torch
from LightGenV2.tasks.t17_router_classification.model_four import sparse_top2,route_amplitudes
from LightGenV2.tasks.t17_router_classification.model_linear_readout import LinearReadoutLG
from LightGenV2.tasks.t17_router_classification.train_figure_revision import training_forward


def test_surrogate_keeps_hard_amplitudes_bitwise_and_opens_masked_backward_only():
    dense=torch.tensor([[.10,.20,.65,.05]],requires_grad=True);q=sparse_top2(dense)
    hard=route_amplitudes(q);surrogate=route_amplitudes(q,dense,True)
    assert torch.equal(hard,surrogate) and int((surrogate>0).sum())==2
    weights=torch.tensor([[1.,2.,3.,4.]])
    a=torch.autograd.grad((hard*weights).sum(),dense,retain_graph=True)[0]
    b=torch.autograd.grad((surrogate*weights).sum(),dense)[0]
    assert torch.count_nonzero(a[q==0])==0 and torch.all(b[q==0]!=0) and torch.isfinite(b).all()


def test_real_optics_surrogate_forward_bitwise_equal_and_masked_dense_gradients_nonzero():
    torch.set_num_threads(4);torch.manual_seed(71);m=LinearReadoutLG(2,10).train();x=torch.rand(1,224,224)
    old=m(x,return_dense_route=True);new=m(x,return_dense_route=True,training_route='top2_straight_through')
    for key in ('probabilities','raw_logits','route_power','dense_route_power','ccd_features'):assert torch.equal(old[key],new[key])
    loss=torch.nn.functional.cross_entropy(new['raw_logits'],torch.tensor([3]));g=torch.autograd.grad(loss,new['dense_route_power'])[0]
    assert torch.isfinite(g).all() and g[new['route_power']==0].abs().sum()>0
    assert len(m.state_dict())==4 and sum(p.numel() for p in m.parameters())==487204 and sum(isinstance(v,torch.nn.Linear) for v in m.modules())==1
    m.eval()
    with torch.no_grad():assert torch.equal(old['probabilities'],m(x)['probabilities'])
    with pytest.raises(ValueError,match='training-only'):m(x,training_route='top2_straight_through')


def test_default_training_wrapper_is_compatible_and_rejects_unsupported_scopes():
    torch.set_num_threads(4);m=LinearReadoutLG(2,10).train();x=torch.rand(1,224,224)
    assert torch.equal(m(x)['probabilities'],training_forward(m,x,{})['probabilities'])
    for cfg in [dict(training_route='unknown',architecture='LG',main_layers=2),dict(training_route='top2_straight_through',architecture='LG',main_layers=4)]:
        with pytest.raises(ValueError):training_forward(m,x,cfg)
