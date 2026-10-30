import pytest
import torch
from LightGenV2.tasks.t17_router_classification.model_four import FourRouterClassification
from LightGenV2.tasks.t17_router_classification.optical_training_methods import augment_rgb_tiles,actual_top2_load_loss


def test_dense_warmup_trains_all_four_experts_and_eval_remains_strict_top2():
    torch.set_num_threads(4);torch.manual_seed(17)
    model=FourRouterClassification('optical','ccd_grid',num_classes=8).train()
    x=torch.rand(2,224,224)
    out=model(x,return_dense_route=True,training_route='dense_warmup',training_uniform_mix=.5)
    assert (out['route_power']>0).sum(1).tolist()==[4,4]
    assert torch.allclose(out['route_power'].sum(1),torch.ones(2))
    torch.nn.functional.cross_entropy(out['logits'],torch.tensor([0,7])).backward()
    assert all(torch.isfinite(model.first_phase.grad[i]).all() and model.first_phase.grad[i].abs().sum()>0 for i in range(4))
    model.eval()
    with pytest.raises(ValueError,match='training-only'):model(x,training_route='dense_warmup')
    hard=model(x,return_dense_route=True)
    assert (hard['route_power']>0).sum(1).tolist()==[2,2]
    assert set(dict(model.named_parameters()))=={'first_phase','global_phase','router_phase'}


def test_d4_augmentation_preserves_colour_identity_zero_tile_and_power():
    torch.manual_seed(17)
    base=torch.arange(112*112).reshape(112,112).float()/20000
    x=torch.cat((torch.cat((base,2*base+2),-1),torch.cat((3*base+4,torch.zeros_like(base)),-1)),-2)[None].repeat(16,1,1)
    out=augment_rgb_tiles(x,1.)
    red=out[:,:112,:112];green=out[:,:112,112:];blue=out[:,112:,:112]
    assert torch.allclose(green,2*red+2) and torch.allclose(blue,3*red+4)
    assert not out[:,112:,112:].count_nonzero()
    assert torch.allclose(out.square().sum((-2,-1)),x.square().sum((-2,-1)))
    assert not torch.equal(out,x)


def test_load_loss_has_gradient_when_soft_mean_is_uniform_but_actual_selection_collapses():
    logits=torch.zeros(8,4,requires_grad=True);dense=logits.softmax(1)
    assert (dense.topk(2,dim=1).indices==dense.topk(2,dim=1).indices[0]).all()
    loss=actual_top2_load_loss(dense);loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
