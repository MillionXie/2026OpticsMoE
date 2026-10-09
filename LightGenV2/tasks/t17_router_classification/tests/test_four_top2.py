import pytest
import torch
from LightGenV2.tasks.t17_router_classification.model_four import (
    FourRouterClassification,CONTRACT,sparse_top2)
from LightGenV2.demo_check.pure_optical.models import PhaseOnly

def test_sparse_top2_power_and_gradient():
    weights=torch.tensor([[.1,.4,.2,.3]],requires_grad=True)
    q=sparse_top2(weights)
    assert torch.equal(q>0,torch.tensor([[False,True,False,True]]))
    assert torch.allclose(q.sum(1),torch.ones(1))
    coefficients=torch.where(q>0,q.clamp_min(1e-20).sqrt(),torch.zeros_like(q))
    (coefficients*torch.arange(4)).sum().backward()
    assert torch.isfinite(weights.grad).all()
    assert weights.grad[0,0]==weights.grad[0,2]==0

@pytest.mark.parametrize('variant',['optical','electronic','d2nn'])
def test_four_geometry_backward_and_two_oeo(variant):
    torch.set_num_threads(4)
    model=FourRouterClassification(variant)
    assert model.global_phase.shape==(478,478)
    assert model.first_phase.shape==((478,478) if variant=='d2nn' else (4,224,224))
    assert all(torch.count_nonzero(p)==0 for n,p in model.named_parameters() if 'phase' in n)
    calls=[];original=model.oeo
    def checked_oeo(x):calls.append(x.shape);return original(x)
    model.oeo=checked_oeo
    x=torch.rand(1,224,224)
    out=model(x,return_debug=True)
    assert calls==[torch.Size([1,518,518])]*2
    if variant!='d2nn':
        assert (out['route_power']>0).sum()==2
        assert torch.allclose(out['route_power'].sum(1),torch.ones(1))
    out['logits'].square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.global_phase.grad.abs().sum()>0

def test_old_four_source_main_path_equivalence():
    torch.set_num_threads(4)
    model=FourRouterClassification('optical')
    reference=PhaseOnly('dynamic_four',dict(CONTRACT))
    x=torch.rand(1,224,224)
    x=x/x.square().sum((-2,-1),keepdim=True).sqrt()
    q,_=model.route(x)
    reference.route=lambda amplitude:(q,None)
    actual=model(x,return_debug=True)
    expected=reference.forward_amplitude(x)
    energies=model.detect(actual['post_oeo_intensity'],reference.class_centers,32)
    probabilities=(energies+1e-12)/(energies.sum(1,keepdim=True)+10e-12)
    assert torch.allclose(probabilities,expected['probabilities'],rtol=1e-5,atol=1e-7)

def test_identical_moe_phases_and_head():
    a=FourRouterClassification('optical');b=FourRouterClassification('electronic')
    c=FourRouterClassification('d2nn')
    assert torch.equal(a.first_phase,b.first_phase)
    assert torch.equal(a.global_phase,b.global_phase)
    assert torch.equal(a.shared_head.weight,b.shared_head.weight)
    assert torch.equal(a.shared_head.weight,c.shared_head.weight)
    assert sum(p.numel() for p in a.parameters())==486420
    assert sum(p.numel() for p in b.parameters())==486744
    assert sum(p.numel() for p in c.parameters())==464024
