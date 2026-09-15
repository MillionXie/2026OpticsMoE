import json
import torch
import pytest
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout import (
    consistent_teacher_targets, teacher_relation_loss, load_consistent_teacher,
    entropy_matched_teacher_temperature)
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import sha256


def fixture():
    labels=torch.tensor([0,0,1,1,2,2])
    vectors=torch.tensor([[1.,0.,0.],[1.,.05,0.],[0.,1.,0.],[0.,1.,.05],
                          [1.,-.02,0.],[0.,1.,-.02]])
    return vectors,labels


def test_only_correct_strict_margin_teacher_rows_eligible_self_excluded():
    vectors,labels=fixture(); targets,eligible=consistent_teacher_targets(vectors,labels)
    assert eligible.tolist()==[False,True,False,True,False,False]
    assert not targets.requires_grad and targets.diagonal().eq(0).all()
    assert torch.allclose(targets.sum(1),torch.ones(6))
    _,stronger=consistent_teacher_targets(vectors,labels,.02)
    assert (stronger<=eligible).all()
    tied=consistent_teacher_targets(torch.ones(6,3),labels)[1]
    assert not tied.any()


def test_wrong_teacher_rows_give_no_query_loss_and_targets_frozen():
    vectors,labels=fixture(); targets,eligible=consistent_teacher_targets(vectors,labels)
    student=torch.randn(6,64,requires_grad=True)
    zero=teacher_relation_loss(student,targets,eligible,torch.tensor([0,2,4,5]))
    zero.backward(); assert zero==0 and student.grad.eq(0).all()
    student.grad=None
    loss=teacher_relation_loss(student,targets,eligible,torch.arange(6))
    loss.backward(); assert torch.isfinite(student.grad).all() and student.grad.norm()>0
    assert targets.grad is None


def test_exact_teacher_relation_matches_zero_kl_without_nan():
    vectors,labels=fixture();targets,eligible=consistent_teacher_targets(vectors,labels)
    loss=teacher_relation_loss(vectors,targets,eligible,torch.arange(6))
    assert torch.isfinite(loss) and abs(float(loss))<1e-6
    with pytest.raises(ValueError):
        consistent_teacher_targets(vectors.requires_grad_(),labels)
    with pytest.raises(ValueError):
        teacher_relation_loss(vectors.detach(),targets.requires_grad_(),eligible,torch.arange(6))


def test_teacher_loader_copies_train_only_and_rejects_wrong_source(tmp_path):
    torch.manual_seed(82)
    rows=[dict(sample_id=str(i),split='gallery' if i<1600 else 'query') for i in range(2400)]
    cache=dict(manifest_sha256='m',ids=[r['sample_id'] for r in rows],vectors=torch.randn(2400,64))
    path=tmp_path/'normal_features.pt'
    report=dict(status='complete',model_kind='qwen64',frozen=True,trainable_parameters=0,manifest_sha256='m')
    (tmp_path/'final_report.json').write_text(json.dumps(report))
    torch.save(cache,path); labels=torch.arange(200).repeat_interleave(8)
    targets,eligible,audit=load_consistent_teacher(path,sha256(path),rows,'m',labels)
    cache['vectors'][1600:]=torch.randn(800,64)*100
    torch.save(cache,path)
    other,other_eligible,_=load_consistent_teacher(path,sha256(path),rows,'m',labels)
    assert torch.equal(other,targets) and torch.equal(other_eligible,eligible)
    assert targets.shape==(1600,1600) and audit['query_excluded'] and not audit['inference_teacher_required']
    with pytest.raises(ValueError,match='SHA'):
        load_consistent_teacher(path,'wrong',rows,'m',labels)
    report['frozen']=False;(tmp_path/'final_report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='frozen'):
        load_consistent_teacher(path,sha256(path),rows,'m',labels)


def test_entropy_matching_uses_train_only_and_recovers_identity_temperature():
    vectors,labels=fixture();_,eligible=consistent_teacher_targets(vectors,labels)
    temperature,audit=entropy_matched_teacher_temperature(vectors,vectors.clone(),eligible)
    assert abs(temperature-.1)<1e-5
    assert abs(audit['teacher_entropy_after']-audit['reference_entropy'])<1e-5
    assert not audit['fitted_to_test'] and not audit['changes_teacher_rankings']


def test_entropy_calibration_sharpens_flat_teacher_without_changing_gate():
    torch.manual_seed(94)
    reference=torch.randn(20,64)
    teacher=reference+4*torch.ones(20,64)
    eligible=torch.ones(20,dtype=torch.bool)
    temperature,audit=entropy_matched_teacher_temperature(teacher,reference,eligible)
    assert .005<temperature<.1
    assert abs(audit['teacher_entropy_after']-audit['reference_entropy'])<1e-4
    labels=torch.arange(5).repeat_interleave(4)
    before=consistent_teacher_targets(teacher,labels)[1]
    after=consistent_teacher_targets(teacher,labels,temperature=temperature)[1]
    assert torch.equal(before,after)


def test_entropy_calibration_rejects_empty_gate_and_trainable_reference():
    vectors,labels=fixture();_,eligible=consistent_teacher_targets(vectors,labels)
    with pytest.raises(ValueError):
        entropy_matched_teacher_temperature(vectors,vectors,torch.zeros(6,dtype=torch.bool))
    with pytest.raises(ValueError):
        entropy_matched_teacher_temperature(vectors,vectors.clone().requires_grad_(),eligible)
