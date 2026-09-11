import copy
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.data import Sample
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import sha256
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config,restore_auxiliary_head
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.teacher_relations import load_teacher_cache,gallery_relation_loss,select_agreeing_external,fit_feature_alignment,aligned_feature_loss


def test_training_feature_alignment_is_orthogonal_and_does_not_change_student():
    torch.manual_seed(71)
    teacher=torch.randn(32,8,dtype=torch.double,requires_grad=True)
    q=torch.linalg.qr(torch.randn(8,8,dtype=torch.double)).Q
    student=(teacher.detach()@q).requires_grad_();before=student.detach().clone()
    rotation=fit_feature_alignment(teacher,student)
    torch.testing.assert_close(rotation,q.float(),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(rotation.T@rotation,torch.eye(8),atol=1e-6,rtol=1e-6)
    assert torch.equal(student,before) and not rotation.requires_grad
    with pytest.raises(ValueError):fit_feature_alignment(teacher[:2],student[:2])
    with pytest.raises(ValueError):fit_feature_alignment(teacher,student[:,:4])
    cfg=overlay_config({},'domain_distill_aligned_feature')
    assert cfg['relation_teacher_weight']==0 and cfg['teacher_feature_weight']==.5
    assert not cfg.get('teacher_agreement_external_only',False)


def test_aligned_feature_loss_gates_teacher_and_detaches_targets():
    query=torch.tensor([[.8,.2]],requires_grad=True);target=torch.tensor([[1.,0.]],requires_grad=True)
    bank=torch.tensor([[1.,0.],[.99,.1],[0.,1.],[.1,.99]],requires_grad=True)
    bank_labels=torch.tensor([0,0,1,1]);own=torch.tensor([0]);labels=torch.tensor([0])
    loss,correct=aligned_feature_loss(query,target,own,labels,torch.tensor([[1.,0.]]),bank,bank_labels)
    assert loss>0 and correct==1
    loss.backward();assert torch.isfinite(query.grad).all() and query.grad.abs().sum()>0
    assert target.grad is None and bank.grad is None
    wrong,_=aligned_feature_loss(query,target,own,labels,torch.tensor([[0.,1.]]),bank,bank_labels)
    assert wrong==0


def test_external_selection_preserves_originals_and_excludes_self():
    labels=[0,1,0,0,0,1,1,1]
    samples=[Sample(f's{i}',f'p{i}',c,str(c),'train',Path(f'{i}.png')) for i,c in enumerate(labels)]
    vectors=torch.tensor([[-1.,0.],[0.,1.],[1.,0.],[1.,.01],[0.,1.1],[0.,.99],[0.,1.],[1.01,0.]],requires_grad=True)
    selected,v,rows=select_agreeing_external(samples,vectors,2)
    assert selected[:2]==samples[:2]  # Original wrong teacher prediction still retained.
    assert rows[0]['teacher_category_id'] != rows[0]['category_id']
    assert rows[0]['kept'] and not rows[4]['kept'] and not rows[7]['kept']
    assert all(r['kept']==(r['original_train'] or r['teacher_category_id']==r['category_id']) for r in rows)
    indices=[i for i,r in enumerate(rows) if r['kept']]
    torch.testing.assert_close(v,vectors.detach()[indices])
    assert not v.requires_grad and len(rows)==len(samples)
    with pytest.raises(ValueError):select_agreeing_external([replace(samples[0],split='test')]+samples[1:],vectors,2)
    with pytest.raises(ValueError):select_agreeing_external(samples,vectors,0)
    cfg=overlay_config({},'domain_distill_teacher_agreement')
    assert cfg['teacher_agreement_external_only'] and cfg['relation_teacher_weight']==.3


def test_relation_uses_teacher_affinities_not_coordinate_matching():
    bank=torch.tensor([[1.,0.],[.99,.1],[0.,1.],[.1,.99]],requires_grad=True)
    labels=torch.tensor([0,0,1,1]);own=torch.tensor([0]);label=torch.tensor([0])
    teacher=torch.tensor([[1.,0.]],requires_grad=True)
    good=torch.tensor([[1.,0.]],requires_grad=True);bad=torch.tensor([[0.,1.]],requires_grad=True)
    a,_=gallery_relation_loss(good,own,label,bank,labels,teacher,bank)
    b,audit=gallery_relation_loss(bad,own,label,bank,labels,teacher,bank)
    assert a.item()<b.item() and audit['teacher_correct_fraction'].item()==1.
    b.backward();assert torch.isfinite(bad.grad).all() and bad.grad.abs().sum()>0
    assert teacher.grad is None and bank.grad is None
    rotated=torch.cat((bank.detach().flip(-1),torch.zeros(4,2)),dim=-1)
    rotated_query=torch.tensor([[0.,1.,0.,0.]])
    c,_=gallery_relation_loss(good,own,label,bank,labels,rotated_query,rotated)
    torch.testing.assert_close(c,a)


def test_wrong_teacher_is_gated_and_self_product_excluded():
    bank=torch.tensor([[1.,0.],[.99,.1],[0.,1.],[.1,.99]])
    labels=torch.tensor([0,0,1,1]);q=torch.tensor([[.8,.2]],requires_grad=True)
    own=torch.tensor([0]);label=torch.tensor([0])
    loss,audit=gallery_relation_loss(q,own,label,bank,labels,torch.tensor([[0.,1.]]),bank)
    assert loss.item()==0. and audit['teacher_correct_fraction'].item()==0.
    loss.backward();assert torch.isfinite(q.grad).all()
    a,_=gallery_relation_loss(q,own,label,bank,labels,torch.tensor([[1.,0.]]),bank)
    changed=bank.clone();changed[0]=torch.tensor([-1.,0.])
    b,_=gallery_relation_loss(q,own,label,changed,labels,torch.tensor([[1.,0.]]),changed)
    torch.testing.assert_close(a,b)


def test_cache_rejects_order_split_manifest_and_nonfinite(tmp_path):
    target=tmp_path/'target';pool=tmp_path/'pool';(target/'data').mkdir(parents=True);pool.mkdir()
    (target/'data/abo_similarity10_manifest.csv').write_text('target')
    (pool/'manifest.csv').write_text('pool')
    samples=[Sample('s0','p0',0,'chair','train',tmp_path/'p0.png'),Sample('s1','p1',1,'rug','train',tmp_path/'p1.png')]
    for s in samples:s.image_path.write_bytes(s.sample_id.encode())
    cache=dict(schema=1,frozen_teacher=True,target_manifest_sha256=sha256(target/'data/abo_similarity10_manifest.csv'),
               pool_manifest_sha256=sha256(pool/'manifest.csv'),ids=['s0','s1'],image_sha256=[sha256(s.image_path) for s in samples],vectors=torch.ones(2,2048))
    path=tmp_path/'cache.pt';torch.save(cache,path)
    vectors,audit=load_teacher_cache(path,samples,target,pool,'cpu')
    assert vectors.shape==(2,2048) and audit['training_images']==2
    with pytest.raises(ValueError):load_teacher_cache(path,samples[::-1],target,pool,'cpu')
    with pytest.raises(ValueError):load_teacher_cache(path,[replace(samples[0],split='test'),samples[1]],target,pool,'cpu')
    wrong=copy.deepcopy(cache);wrong['pool_manifest_sha256']='wrong';torch.save(wrong,path)
    with pytest.raises(ValueError):load_teacher_cache(path,samples,target,pool,'cpu')
    wrong=copy.deepcopy(cache);wrong['vectors'][0,0]=float('nan');torch.save(wrong,path)
    with pytest.raises(ValueError):load_teacher_cache(path,samples,target,pool,'cpu')


def test_distillation_profiles_change_training_only():
    a=overlay_config({},'domain_distill_light');b=overlay_config({},'domain_distill_strong')
    assert a['relation_teacher_weight']==.1 and b['relation_teacher_weight']==.3
    b['relation_teacher_weight']=.1
    assert a==b and a['expected_pool_products_per_category']==250
    assert a['view_consistency_weight']==0.
    c=overlay_config({},'domain_distill_stronger')
    assert c['relation_teacher_weight']==.6
    c['relation_teacher_weight']=.1
    assert c==a


def test_auxiliary_restore_is_pinned_strict_and_training_only():
    head=torch.nn.Linear(3,2)
    state={k:torch.ones_like(v) for k,v in head.state_dict().items()}
    payload=dict(selection_variant='live',auxiliary_head_not_used_at_inference=True,auxiliary_training_head=state)
    before=copy.deepcopy(head.state_dict())
    with pytest.raises(ValueError):restore_auxiliary_head(head,payload,'wrong','expected')
    assert all(torch.equal(v,before[k]) for k,v in head.state_dict().items())
    with pytest.raises(ValueError):restore_auxiliary_head(head,dict(payload,selection_variant='ema'),'expected','expected')
    with pytest.raises(ValueError):restore_auxiliary_head(head,dict(payload,auxiliary_head_not_used_at_inference=False),'expected','expected')
    with pytest.raises(ValueError):restore_auxiliary_head(head,dict(payload,auxiliary_training_head={'weight':torch.ones(2,3)}),'expected','expected')
    restore_auxiliary_head(head,payload,'expected','expected')
    assert all(torch.equal(v,state[k]) for k,v in head.state_dict().items())
    a=overlay_config({},'domain_distill_strong');b=overlay_config({},'domain_distill_resumeaux')
    assert len(b.pop('restore_auxiliary_source_sha256'))==64 and a==b


def test_teacher_temperature_is_separate_and_default_unchanged():
    bank=torch.tensor([[1.,0.],[.7,.71414284],[.5,.8660254],[.4,.9165151]])
    labels=torch.tensor([0,0,1,1]);q=torch.tensor([[.8,.2]],requires_grad=True)
    own=torch.tensor([0]);label=torch.tensor([0]);teacher=torch.tensor([[1.,0.]])
    a,weak=gallery_relation_loss(q,own,label,bank,labels,teacher,bank)
    b,_=gallery_relation_loss(q,own,label,bank,labels,teacher,bank,.1,.1)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    c,strong=gallery_relation_loss(q,own,label,bank,labels,teacher,bank,.1,.03)
    assert strong['teacher_confidence']>weak['teacher_confidence']
    c.backward();assert torch.isfinite(q.grad).all()
    for t in (0.,-1.,float('nan')):
        with pytest.raises(ValueError):gallery_relation_loss(q,own,label,bank,labels,teacher,bank,.1,t)
    x=overlay_config({},'domain_distill_strong');y=overlay_config({},'domain_distill_sharpteacher')
    assert y.pop('relation_teacher_target_temperature')==.03 and x==y
