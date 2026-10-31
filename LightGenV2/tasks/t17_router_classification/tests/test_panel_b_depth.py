import pytest
import torch
from LightGenV2.tasks.t17_router_classification.model_four import FourRouterClassification
from LightGenV2.tasks.t17_router_classification.model_depth_ccd import DepthLG, DepthD2NN, depth_contract


def test_lowest_LG_loads_exact_A_state_and_reproduces_nontrivial_output():
    torch.set_num_threads(4);torch.manual_seed(17)
    a=FourRouterClassification('optical','ccd_grid',num_classes=8)
    with torch.no_grad():
        for p in a.parameters():p.normal_(0,.5)
    b=DepthLG(2);b.load_state_dict(a.state_dict(),strict=True)
    x=torch.rand(2,224,224)
    with torch.no_grad():
        before=a(x,return_debug=True,return_dense_route=True)
        after=b(x,return_debug=True,return_dense_route=True)
    assert set(a.state_dict())==set(b.state_dict())
    for key in before:assert torch.equal(before[key],after[key]),key


@pytest.mark.parametrize('depth',[2,4,6])
def test_depth_shapes_OEO_counts_no_heads_and_finite_gradients(depth):
    torch.set_num_threads(4);contract=depth_contract(depth)
    for model,expected_calls,count in [(DepthLG(depth),depth,contract['LG_phase_parameters']),
                                       (DepthD2NN(depth+1),depth+1,contract['D2NN_phase_parameters'])]:
        assert sum(p.numel() for p in model.parameters())==count
        assert not any(isinstance(m,(torch.nn.Linear,torch.nn.Conv2d,torch.nn.LayerNorm,torch.nn.ReLU)) for m in model.modules())
        calls=[];original=model.oeo
        def tracked(field):calls.append(field.shape);return original(field)
        model.oeo=tracked
        out=model(torch.rand(1,224,224));assert len(calls)==expected_calls
        assert out['probabilities'].shape==(1,8) and torch.allclose(out['probabilities'].sum(1),torch.ones(1),atol=1e-6)
        (-out['logits'][0,2]).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())
        if isinstance(model,DepthLG):assert (out['route_power']>0).sum(1).tolist()==[2]
        else:assert all(p.shape==(478,478) for p in model.phases)


def test_OEO_is_identical_to_A_including_active_crop_and_unit_power():
    a=FourRouterClassification('optical','ccd_grid',num_classes=8)
    d=DepthD2NN(3);field=torch.randn(2,518,518,dtype=torch.complex64)
    assert torch.equal(a.oeo(field),d.oeo(field))
    result=d.oeo(field)
    assert torch.all(result.imag==0) and torch.all(result.real>=0)
    assert torch.allclose(result.abs().square().sum((-2,-1)),torch.ones(2),atol=1e-6)


@pytest.mark.parametrize('constructor,bad',[(DepthLG,3),(DepthLG,8),(DepthD2NN,2),(DepthD2NN,6)])
def test_reject_ambiguous_layer_counts(constructor,bad):
    with pytest.raises(ValueError):constructor(bad)
