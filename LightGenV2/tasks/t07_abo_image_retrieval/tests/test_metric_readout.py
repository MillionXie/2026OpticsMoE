import copy
import pytest
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout import (
    fold_metric, metric_loss, validate_cache, replace_projection,
    projection_loss, validate_projection_cache, train_ranking_loss, fit_step, projection_vectors, centered_projection_bias)


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
    for kind in ['nll','top1_softplus','top1_squared_hinge']:
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


def test_fit_step_zero_sam_exactly_preserves_adam_and_random_stream():
    torch.manual_seed(41)
    x=torch.randn(8,384); y=torch.arange(4).repeat_interleave(2); idx=torch.arange(8)
    w=torch.nn.Parameter(torch.randn(64,384)); b=torch.nn.Parameter(torch.zeros(64))
    ref=(w.detach().clone(),b.detach().clone())
    w2=torch.nn.Parameter(w.detach().clone()); b2=torch.nn.Parameter(b.detach().clone())
    old=torch.optim.Adam([w,b],lr=.0001); new=torch.optim.Adam([w2,b2],lr=.0001)
    rng=torch.get_rng_state()
    old.zero_grad(set_to_none=True)
    expected=projection_loss(w,b,x,y,idx,ref,1.,.1,'top1_softplus')
    expected.backward(); torch.nn.utils.clip_grad_norm_([w,b],1.,error_if_nonfinite=True); old.step()
    expected_rng=torch.get_rng_state()
    torch.set_rng_state(rng)
    loss,audit=fit_step(lambda:projection_loss(w2,b2,x,y,idx,ref,1.,.1,'top1_softplus'),new,[w2,b2])
    assert torch.equal(loss,expected.detach()) and torch.equal(w,w2) and torch.equal(b,b2)
    assert torch.equal(torch.get_rng_state(),expected_rng) and audit['rho']==0


def test_fit_step_sam_replays_dropout_and_keeps_inputs_frozen():
    torch.manual_seed(43)
    x=torch.randn(8,384); x0=x.clone(); y=torch.arange(4).repeat_interleave(2); idx=torch.arange(8)
    w=torch.nn.Parameter(torch.randn(64,384)); b=torch.nn.Parameter(torch.zeros(64))
    ref=(w.detach().clone(),b.detach().clone()); states=[]
    optimizer=torch.optim.Adam([w,b],lr=.0001)
    def closure():
        states.append(torch.get_rng_state().clone())
        return projection_loss(w,b,x,y,idx,ref,1.,.1,'top1_softplus')
    loss,audit=fit_step(closure,optimizer,[w,b],.01)
    assert len(states)==2 and torch.equal(states[0],states[1])
    assert torch.isfinite(loss) and audit['rho']==.01 and audit['gradient_norm']>0
    assert torch.equal(x,x0) and x.grad is None and ref[0].grad is None
    assert not torch.equal(w,ref[0]) and torch.isfinite(w).all()


def test_fit_step_sam_exception_restores_head_and_does_not_step():
    w=torch.nn.Parameter(torch.randn(3)); initial=w.detach().clone(); count=0
    optimizer=torch.optim.Adam([w],lr=.001)
    def closure():
        nonlocal count
        count+=1
        if count==2:
            raise RuntimeError('second-pass failure')
        return w.square().sum()
    with pytest.raises(RuntimeError,match='second-pass failure'):
        fit_step(closure,optimizer,[w],.01)
    assert torch.equal(w,initial) and not optimizer.state


def test_selection_cpu_is_exact_legacy_and_cannot_backpropagate_query():
    x=torch.randn(12,384,requires_grad=True); w=torch.randn(64,384,requires_grad=True)
    b=torch.randn(64,requires_grad=True)
    actual=projection_vectors(x,w,b)
    assert torch.equal(actual,F.normalize(F.linear(x,w,b),dim=-1))
    assert not actual.requires_grad and x.grad is None and w.grad is None
    with pytest.raises(ValueError,match='precision'):
        projection_vectors(x,w,b,'change_inference')


def test_cuda_selection_fails_closed_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='requires'):
        projection_vectors(torch.randn(4,384),torch.randn(64,384),torch.zeros(64),'cuda_bf16')


def test_two_view_loss_uses_two_distinct_nonself_positive_photos():
    logits=torch.tensor([[100.,8.,6.,4.,7.5,1.]],requires_grad=True)
    positive=torch.tensor([[False,True,True,True,False,False]])
    excluded=torch.tensor([[True,False,False,False,False,False]])
    loss=train_ranking_loss(logits,positive,excluded,'two_view_softplus')
    assert torch.allclose(loss,F.softplus(torch.tensor(.7)))
    loss.backward()
    assert logits.grad[0,0]==0 and logits.grad[0,3]==0 and logits.grad[0,5]==0
    assert logits.grad[0,1]<0 and torch.equal(logits.grad[0,1],logits.grad[0,2])
    assert logits.grad[0,4]>0
    positive[0,2:4]=False
    with pytest.raises(ValueError,match='two distinct'):
        train_ranking_loss(logits,positive,excluded,'two_view_softplus')


def test_two_view_projection_keeps_training_inputs_and_reference_frozen():
    torch.manual_seed(52)
    x=torch.randn(12,384); y=torch.arange(4).repeat_interleave(3)
    w=torch.randn(64,384,requires_grad=True); b=torch.zeros(64,requires_grad=True)
    reference=(w.detach().clone(),b.detach().clone())
    loss=projection_loss(w,b,x,y,torch.arange(12),reference,1.,.1,'two_view_softplus')
    loss.backward()
    assert torch.isfinite(w.grad).all() and w.grad.norm()>0 and b.grad.norm()>0
    assert x.grad is None and reference[0].grad is None and reference[1].grad is None


def test_squared_hinge_zero_gradient_for_margin_satisfied_queries():
    logits=torch.tensor([[100.,8.,4.,7.,1.]],requires_grad=True)
    positive=torch.tensor([[False,True,True,False,False]])
    excluded=torch.tensor([[True,False,False,False,False]])
    loss=train_ranking_loss(logits,positive,excluded,'top1_squared_hinge')
    assert loss.item()==0
    loss.backward()
    assert torch.equal(logits.grad,torch.zeros_like(logits))


def test_squared_hinge_has_expected_margin_and_nonself_gradients():
    logits=torch.tensor([[100.,6.,4.,7.,1.]],requires_grad=True)
    positive=torch.tensor([[False,True,True,False,False]])
    excluded=torch.tensor([[True,False,False,False,False]])
    loss=train_ranking_loss(logits,positive,excluded,'top1_squared_hinge')
    assert torch.allclose(loss,torch.tensor(.72))
    loss.backward()
    assert torch.allclose(logits.grad,torch.tensor([[0.,-1.2,0.,1.2,0.]]))
    stronger=logits.detach().clone(); stronger[0,1]+=1
    assert train_ranking_loss(stronger,positive,excluded,'top1_squared_hinge')<loss


def test_squared_hinge_projection_only_trains_existing_head():
    torch.manual_seed(53)
    x=torch.randn(12,384); y=torch.arange(4).repeat_interleave(3)
    w=torch.randn(64,384,requires_grad=True); b=torch.zeros(64,requires_grad=True)
    reference=(w.detach().clone(),b.detach().clone())
    loss=projection_loss(w,b,x,y,torch.arange(12),reference,1.,0.,'top1_squared_hinge')
    loss.backward()
    assert torch.isfinite(w.grad).all() and w.grad.norm()>0 and b.grad.norm()>0
    assert x.grad is None and reference[0].grad is None and reference[1].grad is None


def test_train_centering_folds_mean_into_bias_without_mutation_or_gradients():
    torch.manual_seed(54)
    w=torch.randn(64,384,dtype=torch.float64,requires_grad=True)
    b=torch.randn(64,dtype=torch.float64,requires_grad=True)
    train=torch.randn(30,384,dtype=torch.float64)
    oldw=w.detach().clone(); oldb=b.detach().clone(); oldx=train.clone()
    raw=F.linear(train,w,b).detach()
    centered=centered_projection_bias(w,b,train,1.)
    assert torch.allclose(F.linear(train,w,centered),raw-raw.mean(0),atol=1e-12,rtol=1e-12)
    assert F.linear(train,w,centered).mean(0).abs().max()<1e-12
    half=centered_projection_bias(w,b,train,.5)
    assert torch.allclose(F.linear(train,w,half),raw-.5*raw.mean(0),atol=1e-12,rtol=1e-12)
    assert torch.equal(centered_projection_bias(w,b,train,0.),b)
    assert not centered.requires_grad and w.grad is None and b.grad is None
    assert torch.equal(w,oldw) and torch.equal(b,oldb) and torch.equal(train,oldx)


def test_train_centering_rejects_grad_inputs_invalid_geometry_and_nonfinite():
    w=torch.randn(64,384); b=torch.randn(64); train=torch.randn(8,384)
    for strength in [-.1,1.1,float('nan')]:
        with pytest.raises(ValueError): centered_projection_bias(w,b,train,strength)
    for invalid in [train.clone().requires_grad_(),train[:0],train[:,:383],torch.full_like(train,float('nan'))]:
        with pytest.raises(ValueError): centered_projection_bias(w,b,invalid,1.)
    with pytest.raises(ValueError): centered_projection_bias(w[:63],b,train,1.)
