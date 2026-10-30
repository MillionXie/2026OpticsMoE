import pytest
import torch
from LightGenV2.tasks.t17_router_classification.model_four import (
    FourRouterClassification,ccd_grid_energies,CCD_EDGES)

def test_grid_is_disjoint_full_coverage_and_correct_labels():
    active=torch.zeros(9,478,478)
    for k in range(9):
        r,c=divmod(k,3)
        active[k,CCD_EDGES[r]:CCD_EDGES[r+1],CCD_EDGES[c]:CCD_EDGES[c+1]]=1
    energies=ccd_grid_energies(active)
    assert torch.equal(energies.argmax(1),torch.arange(9))
    assert torch.equal(energies.sum(1),active.sum((-2,-1)))
    assert (energies>0).sum()==9

@pytest.mark.parametrize('variant',['optical','electronic','d2nn'])
def test_ccd_no_head_same_optics_and_finite_training(variant):
    torch.set_num_threads(4)
    a=FourRouterClassification(variant)
    b=FourRouterClassification(variant,'ccd_grid')
    assert not hasattr(b,'shared_head')
    assert sum(p.numel() for p in a.parameters())-sum(p.numel() for p in b.parameters())==7056
    for name,value in b.state_dict().items():assert torch.equal(value,a.state_dict()[name])
    x=torch.rand(2,224,224)
    debug=b(x,return_debug=True)
    old=a(x,return_debug=True)
    assert torch.equal(debug['post_oeo_intensity'],old['post_oeo_intensity'])
    assert torch.allclose(debug['logits'].exp().sum(1),torch.ones(2))
    torch.nn.functional.cross_entropy(debug['logits'],torch.tensor([0,8])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())
    assert b.global_phase.grad.abs().sum()>0
    if variant!='d2nn':assert (debug['route_power']>0).sum(1).tolist()==[2,2]


@pytest.mark.parametrize('classes',[8,9])
def test_unified_optical_four_ccd_layout_and_no_learned_electronics(classes):
    from torch import nn
    torch.set_num_threads(4)
    m=FourRouterClassification('optical','ccd_grid',num_classes=classes)
    assert m.first_phase.shape==(4,224,224) and m.global_phase.shape==(478,478)
    assert m.router_phase.shape==(224,224)
    assert sum(p.numel() for p in m.parameters())==479364
    assert not hasattr(m,'shared_head') and not hasattr(m,'electronic_router')
    assert not any(isinstance(layer,(nn.Linear,nn.Conv2d)) for layer in m.modules())
    calls=[];oeo=m.oeo
    def counted(field):calls.append(field.shape);return oeo(field)
    m.oeo=counted
    x=torch.rand(2,224,224)
    out=m(x,return_debug=True,return_dense_route=True)
    assert len(calls)==2 and out['probabilities'].shape==(2,classes)
    assert torch.allclose(out['probabilities'].sum(1),torch.ones(2))
    assert (out['route_power']>0).sum(1).tolist()==[2,2]
    assert torch.equal(out['class_energies'],ccd_grid_energies(out['post_oeo_intensity'][:,20:498,20:498],classes))
    # Match the exact eight/nine-class detector layout used by the frozen D2NN runs.
    active=torch.rand(2,478,478)
    ys=(0,239,478) if classes==8 else CCD_EDGES
    xs=(0,119,239,358,478) if classes==8 else CCD_EDGES
    expected=torch.stack([active[:,ys[r]:ys[r+1],xs[c]:xs[c+1]].sum((-2,-1))
                          for r in range(len(ys)-1) for c in range(len(xs)-1)],1)
    assert torch.equal(ccd_grid_energies(active,classes),expected)
    assert torch.allclose(expected.sum(1),active.sum((-2,-1)))
    torch.nn.functional.cross_entropy(out['logits'],torch.tensor([0,classes-1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters())
