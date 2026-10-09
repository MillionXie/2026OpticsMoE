"""Independent T17 tests with a unique module name for joint collection."""
import torch
from LightGenV2.tasks.t17_router_classification.model import RouterClassification
from LightGenV2.tasks.t17_router_classification.train import encode

def test_shared_parameters_and_only_router_replaced():
    optical=RouterClassification('optical')
    electronic=RouterClassification('electronic')
    d2nn=RouterClassification('d2nn')
    for model in (optical,electronic,d2nn):
        assert model.shared_head.weight.shape==(9,784)
        assert torch.equal(model.shared_head.weight,optical.shared_head.weight)
        assert int(model.active_count)==16
        assert torch.count_nonzero(model.global_phase)==0
    assert electronic.router_phase is None
    assert sum(p.numel() for p in electronic.electronic_router.parameters())==51280
    for a,b in zip(optical.first_phase,electronic.first_phase): assert torch.equal(a,b)
    assert torch.equal(optical.global_phase,electronic.global_phase)
    x=torch.rand(2,224,224)
    q,_,_=electronic.route_with_efficiency(x)
    assert torch.allclose(q.sum(1),torch.ones(2))
    assert torch.allclose(q,torch.full_like(q,1/16))

def test_encoding_preserves_channels():
    import numpy as np
    image=np.zeros((1,224,224,3),dtype=np.uint8)
    image[...,0]=255; image[...,1]=128
    x=encode(image)
    assert x.shape==(1,224,224)
    assert torch.all(x[:,:112,:112]==1)
    assert torch.allclose(x[:,:112,112:],torch.full((1,112,112),128/255))
    assert torch.count_nonzero(x[:,112:,:])==0
