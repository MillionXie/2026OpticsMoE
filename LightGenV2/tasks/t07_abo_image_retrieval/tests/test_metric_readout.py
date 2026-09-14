import copy
import pytest
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout import (
    fold_metric, metric_loss, validate_cache, replace_projection,
    projection_loss, validate_projection_cache, train_ranking_loss)


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


def test_direct_projection_changes_only_original_weight_bias():
    p=dict(metadata={'retrieval_head':'linear64'},state_dict={
        'readout.projection.weight':torch.randn(64,384),
        'readout.projection.bias':torch.randn(64), 'phase':torch.randn(3,3)})
    old=copy.deepcopy(p)
    w=p['state_dict']['readout.projection.weight']+.01
    b=p['state_dict']['readout.projection.bias']-.02
    q=replace_projection(p,w,b)
    assert q['metadata'] is p['metadata']
    assert q['state_dict']['phase'] is p['state_dict']['phase']
    assert q['state_dict'].keys()==p['state_dict'].keys()
    assert all(torch.equal(v,old['state_dict'][k]) for k,v in p['state_dict'].items())
    x=torch.randn(8,384)
    assert torch.equal(F.linear(x,w,b), F.linear(x,q['state_dict']['readout.projection.weight'],q['state_dict']['readout.projection.bias']))
    with pytest.raises(ValueError): replace_projection(p,w[:32],b)


def test_projection_loss_updates_head_not_frozen_inputs_or_reference():
    torch.manual_seed(8)
    x=torch.randn(8,384); y=torch.arange(4).repeat_interleave(2)
    w=torch.randn(64,384,requires_grad=True); b=torch.zeros(64,requires_grad=True)
    ref=(w.detach().clone(),b.detach().clone())
    loss=projection_loss(w,b,x,y,torch.arange(8),ref,1.)
    loss.backward()
    assert w.grad.norm()>0 and b.grad.norm()>0 and torch.isfinite(w.grad).all()
    assert x.grad is None and ref[0].grad is None
    with pytest.raises(ValueError,match='nonself'):
        projection_loss(w,b,x,torch.arange(8),torch.arange(8),ref,1.)
    with pytest.raises(ValueError,match='detached'):
        projection_loss(w,b,x.requires_grad_(),y,torch.arange(8),ref,1.)


def test_projection_cache_requires_same_checkpoint_and_reproduced_vectors():
    p=dict(state_dict={'readout.projection.weight':torch.randn(64,384),
        'readout.projection.bias':torch.randn(64)})
    x=torch.randn(5,384)
    z=F.normalize(F.linear(x,p['state_dict']['readout.projection.weight'],p['state_dict']['readout.projection.bias']),dim=1)
    c=dict(checkpoint_sha256='sha',ids=list('abcde'),readout_inputs=x,vectors=z)
    validate_projection_cache(c,'sha',p)
    with pytest.raises(ValueError): validate_projection_cache(c,'wrong',p)
    with pytest.raises(ValueError): validate_projection_cache(dict(c,readout_inputs=x.flip(0)),'sha',p)


def test_linear_input_hook_is_passive_and_captures_post_norm():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import RetrievalHead
    h=RetrievalHead('linear64').eval(); x=torch.randn(3,77,192)
    expected=h(x); captured=[]
    hook=h.projection.register_forward_pre_hook(lambda module,args: captured.append(args[0].detach().clone()))
    actual=h(x); hook.remove()
    assert torch.equal(actual,expected) and captured[0].shape==(3,384)
    assert torch.equal(actual,F.normalize(h.projection(captured[0]),dim=-1))


def test_projection_input_dropout_is_train_only_and_does_not_mutate_inputs():
    torch.manual_seed(12)
    x=torch.randn(8,384); original=x.clone(); y=torch.arange(4).repeat_interleave(2)
    w=torch.randn(64,384,requires_grad=True); b=torch.zeros(64,requires_grad=True)
    ref=(w.detach().clone(),b.detach().clone()); idx=torch.arange(8)
    clean=projection_loss(w,b,x,y,idx,ref,1.)
    assert torch.equal(clean,projection_loss(w,b,x,y,idx,ref,1.,0.))
    first=projection_loss(w,b,x,y,idx,ref,1.,.1)
    second=projection_loss(w,b,x,y,idx,ref,1.,.1)
    assert first!=second and torch.equal(original,x)
    first.backward()
    assert torch.isfinite(w.grad).all() and w.grad.norm()>0 and x.grad is None
    assert torch.equal(clean,projection_loss(w,b,x,y,idx,ref,1.))
    for invalid in [-.1,1.,float('nan')]:
        with pytest.raises(ValueError,match='dropout'):
            projection_loss(w,b,x,y,idx,ref,1.,invalid)


def test_top1_surrogate_uses_nearest_correct_and_incorrect_not_self():
    logits=torch.tensor([[100.,8.,4.,7.,1.]],requires_grad=True)
    positive=torch.tensor([[False,True,True,False,False]])
    excluded=torch.tensor([[True,False,False,False,False]])
    loss=train_ranking_loss(logits,positive,excluded,'top1_softplus')
    assert torch.allclose(loss,F.softplus(torch.tensor(-.8)))
    loss.backward()
    assert logits.grad[0,0]==0 and logits.grad[0,2]==0 and logits.grad[0,4]==0
    assert logits.grad[0,1]<0 and logits.grad[0,3]>0
    stronger=logits.detach().clone(); stronger[0,1]+=1
    assert train_ranking_loss(stronger,positive,excluded,'top1_softplus')<loss


def test_ranking_nll_preserves_original_and_rejects_invalid_sets():
    logits=torch.tensor([[10.,3.,2.,1.]])
    pos=torch.tensor([[False,True,True,False]]); exc=torch.tensor([[True,False,False,False]])
    expected=logits.masked_fill(exc,-torch.inf).logsumexp(1)-logits.masked_fill(~pos,-torch.inf).logsumexp(1)
    assert torch.equal(train_ranking_loss(logits,pos,exc),expected.mean())
    for kind in ['nll','top1_softplus']:
        with pytest.raises(ValueError): train_ranking_loss(logits,torch.zeros_like(pos),exc,kind)
        with pytest.raises(ValueError): train_ranking_loss(logits,~exc,exc,kind)
    with pytest.raises(ValueError): train_ranking_loss(logits,pos,exc,'test_rerank')


def test_top1_projection_has_only_original_head_gradients():
    torch.manual_seed(48)
    x=torch.randn(8,384); y=torch.arange(4).repeat_interleave(2)
    w=torch.randn(64,384,requires_grad=True); b=torch.zeros(64,requires_grad=True)
    ref=(w.detach().clone(),b.detach().clone())
    loss=projection_loss(w,b,x,y,torch.arange(8),ref,.1,.1,'top1_softplus')
    loss.backward()
    assert torch.isfinite(w.grad).all() and w.grad.norm()>0 and b.grad.norm()>0
    assert x.grad is None and ref[0].grad is None and ref[1].grad is None


def test_hybrid_is_fixed_equal_mixture_with_no_self_gradient():
    logits=torch.tensor([[100.,8.,4.,7.,1.]],requires_grad=True)
    positive=torch.tensor([[False,True,True,False,False]])
    excluded=torch.tensor([[True,False,False,False,False]])
    hybrid=train_ranking_loss(logits,positive,excluded,'hybrid_nll_top1')
    expected=.5*(train_ranking_loss(logits,positive,excluded,'nll')+
                 train_ranking_loss(logits,positive,excluded,'top1_softplus'))
    assert torch.equal(hybrid,expected)
    hybrid.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0,0]==0
    assert logits.grad[0,1]<0 and logits.grad[0,2]<0 and logits.grad[0,3]>0
