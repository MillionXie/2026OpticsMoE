import hashlib
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t17_router_classification.model_depth_ccd import DepthLG
from LightGenV2.tasks.t17_router_classification.model_linear_readout import LinearReadoutLG,load_optical_parent,initialize_CCD_region_prior
from LightGenV2.tasks.t17_router_classification.readout_data import align_raw_pixels


@pytest.mark.parametrize('depth,classes',[(2,8),(2,9),(4,8),(6,8)])
def test_single_linear_changes_readout_without_changing_optical_field(depth,classes):
    torch.set_num_threads(4);torch.manual_seed(17);old=DepthLG(depth,classes)
    with torch.no_grad():
        for p in old.parameters():p.uniform_(-1,1)
    new=LinearReadoutLG(depth,classes);load_optical_parent(new,old.state_dict());x=torch.rand(1,224,224)
    expected=old(x,return_debug=True)['post_oeo_intensity']
    output=new(x,return_debug=True)
    assert torch.equal(output['post_oeo_intensity'],expected)
    features=F.adaptive_avg_pool2d(expected[:,20:498,20:498,None].permute(0,3,1,2),28).flatten(1)
    features=features/features.mean(1,keepdim=True)
    assert torch.equal(features,output['ccd_features'])
    assert torch.equal(output['raw_logits'],F.linear(features,new.shared_head.weight,None))
    assert sum(isinstance(m,torch.nn.Linear) for m in new.modules())==1 and new.shared_head.bias is None
    assert new.shared_head.weight.numel()==784*classes
    frozen={n:p.detach().clone() for n,p in new.named_parameters() if n!='shared_head.weight'}
    for n,p in new.named_parameters():p.requires_grad_(n=='shared_head.weight')
    optimizer=torch.optim.AdamW(new.shared_head.parameters(),lr=.001)
    F.cross_entropy(new(x)['raw_logits'],torch.tensor([2])).backward();optimizer.step()
    assert all(p.grad is None and torch.equal(p,frozen[n]) for n,p in new.named_parameters() if n in frozen)


def test_optical_parent_loading_cannot_silently_omit_a_phase_tensor():
    model=LinearReadoutLG(2,8);state=DepthLG(2,8).state_dict();state.pop('first_phase')
    with pytest.raises(ValueError,match='every optical tensor'):load_optical_parent(model,state)


def test_fast_raw_cache_enforces_labels_and_exact_pixel_order():
    images=np.arange(24,dtype=np.uint8).reshape(2,2,2,3);labels=np.array([0,1])
    rows=[dict(class_id=str(i),pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()) for i,image in enumerate(images)]
    result=align_raw_pixels(images,labels,rows,np.array(['a','b']))
    assert np.array_equal(result['images'],images)
    with pytest.raises(ValueError,match='label order'):align_raw_pixels(images,labels[::-1],rows,np.array(['a','b']))
    with pytest.raises(ValueError,match='pixel order'):align_raw_pixels(images[::-1],labels,rows,np.array(['a','b']))


@pytest.mark.parametrize('classes',[8,9])
def test_CCD_prior_is_only_an_initialization_of_the_same_one_Linear(classes):
    model=LinearReadoutLG(2,classes);before={n:p.detach().clone() for n,p in model.named_parameters() if n!='shared_head.weight'}
    initialize_CCD_region_prior(model)
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before)
    ys=(0,239,478) if classes==8 else (0,159,318,478)
    xs=(0,119,239,358,478) if classes==8 else ys
    examples=[]
    for y0,y1 in zip(ys,ys[1:]):
        for x0,x1 in zip(xs,xs[1:]):
            x=torch.ones(478,478);x[y0:y1,x0:x1]=3;examples.append(x)
    features=F.adaptive_avg_pool2d(torch.stack(examples)[:,None],28).flatten(1)
    features=features/features.mean(1,keepdim=True)
    assert torch.equal(model.shared_head(features).argmax(1),torch.arange(classes))
    assert sum(isinstance(m,torch.nn.Linear) for m in model.modules())==1
