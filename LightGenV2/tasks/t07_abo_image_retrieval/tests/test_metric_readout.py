import copy
import pytest
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout import fold_metric, metric_loss, validate_cache


def test_fold_preserves_exact_algebra_and_all_other_tensors():
    torch.manual_seed(2)
    p = dict(metadata={'retrieval_head':'linear64'},state_dict={
        'readout.projection.weight':torch.randn(64,384,dtype=torch.float64),
        'readout.projection.bias':torch.randn(64,dtype=torch.float64),
        'vision.optics.experts.0':torch.randn(4,4)})
    old=copy.deepcopy(p)
    a=torch.eye(64,dtype=torch.float64)+.02*torch.randn(64,64,dtype=torch.float64)
    q=fold_metric(p,a); x=torch.randn(9,384,dtype=torch.float64)
    before=F.normalize(F.linear(x,p['state_dict']['readout.projection.weight'],p['state_dict']['readout.projection.bias']),dim=1)
    after=F.normalize(F.linear(x,q['state_dict']['readout.projection.weight'],q['state_dict']['readout.projection.bias']),dim=1)
    assert torch.allclose(F.normalize(before@a.T,dim=1),after,atol=1e-12)
    assert q['state_dict'].keys()==p['state_dict'].keys()
    assert q['state_dict']['vision.optics.experts.0'] is p['state_dict']['vision.optics.experts.0']
    assert all(torch.equal(old['state_dict'][k],v) for k,v in p['state_dict'].items())


def test_training_only_matrix_gradients_and_self_exclusion():
    torch.manual_seed(3)
    a=torch.eye(64,requires_grad=True); z=F.normalize(torch.randn(8,64),dim=1)
    labels=torch.tensor([0,0,1,1,2,2,3,3])
    loss=metric_loss(a,z,labels,torch.arange(8),1.)
    loss.backward()
    assert torch.isfinite(a.grad).all() and a.grad.norm()>0 and z.grad is None
    with pytest.raises(ValueError,match='nonself'):
        metric_loss(a,z,torch.arange(8),torch.arange(8),1.)


def test_wrong_cache_order_or_manifest_rejected():
    rows=[dict(sample_id='a'),dict(sample_id='b')]
    cache=dict(manifest_sha256='sha',ids=['a','b'],vectors=torch.ones(2,64))
    validate_cache(cache,rows,'sha')
    with pytest.raises(ValueError): validate_cache(cache,rows[::-1],'sha')
    with pytest.raises(ValueError): validate_cache(cache,rows,'other')


def test_identity_fold_and_unsupported_head():
    p=dict(metadata={},state_dict={'readout.projection.weight':torch.randn(64,384),
        'readout.projection.bias':torch.randn(64)})
    q=fold_metric(p,torch.eye(64))
    assert all(torch.equal(v,q['state_dict'][k]) for k,v in p['state_dict'].items())
    p['metadata']['retrieval_head']='relu128'
    with pytest.raises(ValueError): fold_metric(p,torch.eye(64))
