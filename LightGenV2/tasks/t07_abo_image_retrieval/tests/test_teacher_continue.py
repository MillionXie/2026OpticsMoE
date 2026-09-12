import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import (
    learning_rate_multiplier, overlay_config, supervised_loss_scale,
)
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.teacher_relations import load_feature_alignment
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import sha256


def alignment():
    return dict(rotation=torch.eye(8), fit_sample_ids=[f'train-{i}' for i in range(16)],
                source_checkpoint_sha256='a'*64, teacher_cache_sha256='b'*64,
                teacher_prefix_dimensions=8, teacher_only=True)


def load(path, expected):
    return load_feature_alignment(path, expected, [f'train-{i}' for i in range(16)], 'b'*64, 'a'*64, 8)


def test_reuse_exact_training_basis(tmp_path):
    a=alignment();path=tmp_path/'alignment.pt';torch.save(a,path)
    result=load(path,sha256(path))
    torch.testing.assert_close(result['rotation'],a['rotation'],atol=0,rtol=0)
    assert not result['rotation'].requires_grad
    assert result['source_checkpoint_sha256']=='a'*64
    with pytest.raises(ValueError,match='SHA'):load(path,'0'*64)


@pytest.mark.parametrize('key,value',[
    ('fit_sample_ids',['test-0']),('teacher_cache_sha256','c'*64),
    ('source_checkpoint_sha256','c'*64),('teacher_prefix_dimensions',64),
    ('teacher_only',False),('rotation',torch.ones(8,8)),('rotation',torch.full((8,8),float('nan'))),
])
def test_alignment_identity_and_orthogonality_checked(tmp_path,key,value):
    a=alignment();a[key]=value;path=tmp_path/'alignment.pt';torch.save(a,path)
    with pytest.raises(ValueError):load(path,sha256(path))


def test_continuation_profile_and_old_defaults():
    assert learning_rate_multiplier({})==1.
    for value in (0.,-1.,1.1,float('nan')):
        with pytest.raises(ValueError):learning_rate_multiplier({'learning_rate_multiplier':value})
    cfg=overlay_config({},'domain_distill_teacher_continue')
    assert learning_rate_multiplier(cfg)==.25
    assert cfg['teacher_feature_weight']==2 and cfg['relation_teacher_weight']==0
    assert cfg['preserve_restored_category_proxies']
    assert len(cfg['restore_auxiliary_source_sha256'])==64
    assert len(cfg['teacher_alignment_sha256'])==64
    assert cfg['input_preprocessing']=='contain_white'
    assert cfg.get('retrieval_head','linear64')=='linear64'
    assert [supervised_loss_scale(i,cfg) for i in range(1,13)]==[1.]*12
    assert learning_rate_multiplier(overlay_config({},'domain_distill_teacher_first'))==1.


def test_sam_continuation_changes_only_training_update():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import PROFILES, PINNED_TEACHER_PROFILES
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization_queue import PINNED_TEACHER_PROFILES as queued
    assert queued==PINNED_TEACHER_PROFILES
    assert all(p in PROFILES for p in PINNED_TEACHER_PROFILES)
    plain=overlay_config({},'domain_distill_teacher_continue')
    sam=overlay_config({},'domain_distill_teacher_continue_sam')
    assert sam['sam_rho']==.02 and sam['sam_warmup_epochs']==3
    ignored={'sam_rho','sam_warmup_epochs','protocol'}
    assert {k:v for k,v in plain.items() if k not in ignored}=={k:v for k,v in sam.items() if k not in ignored}


def test_softgt_continuation_changes_only_final_label_loss_scale():
    plain=overlay_config({},'domain_distill_teacher_continue')
    soft=overlay_config({},'domain_distill_teacher_continue_softgt')
    ignored={'supervised_warmup_epochs','supervised_recovery_epochs','supervised_warmup_scale','protocol'}
    assert {k:v for k,v in plain.items() if k not in ignored}=={k:v for k,v in soft.items() if k not in ignored}
    assert [supervised_loss_scale(i,soft) for i in range(1,13)]==[.5]*12
    assert supervised_loss_scale(13,soft)==1.
    assert soft['sam_rho']==0 and soft['teacher_feature_weight']==2.


def test_fp32_gallery_profile_is_only_loss_precision_change():
    plain=overlay_config({},'domain_distill_teacher_continue')
    fp32=overlay_config({},'domain_distill_teacher_continue_fp32gallery')
    assert fp32['gallery_loss_full_precision'] is True
    ignored={'gallery_loss_full_precision','protocol'}
    assert {k:v for k,v in plain.items() if k not in ignored}=={k:v for k,v in fp32.items() if k not in ignored}


def test_joint_curriculum_changes_training_recipe_not_inference_contract():
    plain=overlay_config({},'domain_distill_teacher_continue')
    joint=overlay_config({},'domain_distill_joint_curriculum')
    ignored={'relation_teacher_weight','learning_rate_multiplier','supervised_warmup_epochs',
             'supervised_recovery_epochs','supervised_warmup_scale','protocol'}
    assert {k:v for k,v in plain.items() if k not in ignored}=={k:v for k,v in joint.items() if k not in ignored}
    assert joint['teacher_feature_weight']==2 and joint['relation_teacher_weight']==.3
    assert joint['relation_teacher_warmup_epochs']==3 and learning_rate_multiplier(joint)==1.
    expected=[.2]*4+[.4,.6,.8,1.]+[1.]*8
    assert [supervised_loss_scale(i,joint) for i in range(1,17)]==pytest.approx(expected)
    assert joint['sam_rho']==0 and joint.get('retrieval_head','linear64')=='linear64'


@pytest.mark.parametrize('balanced',[False,True])
def test_fp32_gallery_matches_reference_inside_autocast(balanced):
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_training import gallery_loss
    torch.manual_seed(1729)
    source=torch.randn(8,64)
    bank=torch.nn.functional.normalize(torch.randn(12,64),dim=-1).requires_grad_(True)
    bank_labels=torch.arange(12)//4
    own=torch.arange(8);labels=bank_labels[own]
    q=source.clone().requires_grad_(True)
    reference=gallery_loss(q,labels,own,bank,bank_labels,class_balance=balanced)
    sum(reference[:2]).backward();expected_gradient=q.grad.clone()
    q=source.clone().requires_grad_(True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        actual=gallery_loss(q,labels,own,bank,bank_labels,class_balance=balanced,full_precision=True)
        legacy=gallery_loss(q,labels,own,bank,bank_labels,class_balance=balanced)
        # Class-count correction already promotes the old NLL to FP32, but
        # its similarity matrix was still multiplied in BF16 beforehand.
        assert legacy[0].dtype==(torch.float32 if balanced else torch.bfloat16)
    sum(actual[:2]).backward()
    assert actual[0].dtype==torch.float32 and bank.grad is None
    for a,b in zip(actual,reference):torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close(q.grad,expected_gradient,atol=0,rtol=0)
