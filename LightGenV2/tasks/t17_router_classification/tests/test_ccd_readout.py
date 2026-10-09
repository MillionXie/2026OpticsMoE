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
