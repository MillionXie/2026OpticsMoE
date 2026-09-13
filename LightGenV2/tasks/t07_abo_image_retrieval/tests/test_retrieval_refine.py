import copy
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import (
    validate_continuation, balanced_targets, route_objective, load_train_teacher,
    relational_loss, selection_score)
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import sha256
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import training_pairs


def test_new_abo_resume_only_accepts_same_manifest():
    protocol='abo200_enrolled_sku_hash8train4query_v1'
    validate_continuation(protocol, {}, 'new', True)
    validate_continuation(protocol, {'manifest_sha256':'new','test_selected':True}, 'new', False)
    for bad in ({}, {'manifest_sha256':'old','test_selected':True}):
        with pytest.raises(ValueError,match='EXACT'):
            validate_continuation(protocol,bad,'new',False)


def test_balanced_targets_detached_and_physical_loss_has_gradient():
    energy=torch.tensor([[40.,57.,1.,2.]]*8,requires_grad=True)
    p=energy.softmax(1)
    q=balanced_targets(p)
    assert not q.requires_grad and torch.allclose(q.sum(1),torch.ones(8),atol=1e-5)
    assert torch.allclose(q.sum(0),torch.full((4,),2.),atol=1e-4)
    branch=SimpleNamespace(optics=SimpleNamespace(router=SimpleNamespace(last={'energy':energy,'probabilities':p})))
    loss=route_objective(SimpleNamespace(vision=branch,language=branch))
    loss.backward()
    assert torch.isfinite(energy.grad).all() and energy.grad[:,2].mean()<0


def test_teacher_returns_train_only_and_rejects_test_fitting(tmp_path):
    train=[dict(sample_id=f't{i}') for i in range(2)]
    query=[dict(sample_id='q')]
    groups=dict(train=train,gallery=train,query=query)
    fit=dict(train=train,gallery=train)
    path=tmp_path/'features.pt'
    torch.save(dict(ids=['t0','t1','q'],manifest_sha256='m',vectors=torch.randn(3,64)),path)
    values=load_train_teacher(path,sha256(path),'m',groups,fit)
    assert set(values)=={'t0','t1'}
    with pytest.raises(ValueError,match='Non-TRAIN'):
        load_train_teacher(path,sha256(path),'m',groups,dict(train=query,gallery=train))
    with pytest.raises(ValueError,match='SHA'):
        load_train_teacher(path,'wrong','m',groups,fit)


def test_relational_distillation_no_teacher_or_bank_gradient():
    z=torch.randn(4,64,requires_grad=True)
    bank=torch.randn(8,64,requires_grad=True)
    tq=torch.randn(2,64,requires_grad=True)
    tb=torch.randn(8,64,requires_grad=True)
    mask=torch.zeros(2,8,dtype=torch.bool);mask[0,0]=mask[1,1]=True
    loss=relational_loss(z,bank,tq,tb,mask)
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad[:2].norm()>0
    assert bank.grad is None and tq.grad is None and tb.grad is None


def test_balance_eligibility_precedes_raw_accuracy():
    good=dict(test=dict(hit_at_1=.8,map_at_10=.7),router={m:dict(selection_share=[.25]*4,most_common_top2_fraction=.4,unique_top2_sets=6) for m in ('vision','language')})
    bad=copy.deepcopy(good);bad['test']['hit_at_1']=.9;bad['router']['language']['selection_share']=[.5,.5,0,0]
    assert selection_score(good,True)>selection_score(bad,True)
    assert selection_score(good,False)<selection_score(bad,False)


def test_hard_category_batches_do_not_mix_queries_or_repeat_product():
    import random
    g={s:[dict(sample_id=f'{s}{i}',product_id=str(i),split=s,category_id=i//4) for i in range(12)] for s in ('train','gallery')}
    rows,labels=training_pairs(g,random.Random(42),4,1.)
    assert len({r['category_id'] for r in rows})==1
    assert len(set(labels.tolist()))==4
