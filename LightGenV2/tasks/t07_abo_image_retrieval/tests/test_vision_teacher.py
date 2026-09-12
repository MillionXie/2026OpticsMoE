from dataclasses import replace
from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.data import Sample
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.vision_teacher import (
    PREPROCESSING,validate_cache,processor_identity,patch_cosine_loss,
)


def fixture():
    samples=[Sample(str(i),str(i),i,'category','train',Path(str(i))) for i in range(2)]
    x=torch.nn.functional.normalize(torch.randn(2,49,2048),dim=-1).half()
    cache=dict(schema=1,kind='vision_patch_teacher_cache',frozen_teacher=True,teacher_trainable_parameters=0,
        student_inference_teacher_required=False,complete_training_pool=True,excluded_splits=['val','test'],
        preprocessing=PREPROCESSING,ids=['0','1'],image_sha256=['image0','image1'],features=x,
        target_manifest_sha256='target',pool_manifest_sha256='pool',processor_sha256='processor')
    return samples,cache


def check(samples,cache):return validate_cache(cache,samples,'target','pool','processor',['image0','image1'])


def test_train_only_cache_and_loss_gradients():
    torch.manual_seed(42);samples,cache=fixture();target=check(samples,cache).float().requires_grad_()
    student=torch.randn_like(target,requires_grad=True);loss=patch_cosine_loss(student,target);loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all() and student.grad.abs().sum()>0
    assert target.grad is None
    assert patch_cosine_loss(target,target).abs()<1e-6


@pytest.mark.parametrize('key,value',[
    ('ids',['1','0']),('complete_training_pool',False),('excluded_splits',['test']),
    ('preprocessing','crop224'),('teacher_trainable_parameters',1),('processor_sha256','changed'),
    ('target_manifest_sha256','changed'),('pool_manifest_sha256','changed'),('image_sha256',['changed','image1']),
    ('student_inference_teacher_required',True),('features',torch.ones(2,49,2048).half()),
    ('features',torch.ones(2,48,2048).half()),('features',torch.full((2,49,2048),float('nan')).half()),
])
def test_reject_changed_identity_incomplete_or_invalid_targets(key,value):
    samples,cache=fixture();cache[key]=value
    with pytest.raises(ValueError):check(samples,cache)


def test_reject_nontraining_and_duplicate_ids():
    samples,cache=fixture()
    for changed in ([replace(samples[0],split='test'),samples[1]],[samples[0],samples[0]]):
        with pytest.raises(ValueError):check(changed,cache)
    with pytest.raises(ValueError):patch_cosine_loss(torch.ones(2,49,64),torch.ones(2,49,64))


def test_processor_identity_tracks_bytes_and_paths(tmp_path):
    with pytest.raises(ValueError):processor_identity(tmp_path)
    (tmp_path/'config.json').write_text('a');a=processor_identity(tmp_path)
    (tmp_path/'config.json').write_text('b');b=processor_identity(tmp_path);assert a!=b
    (tmp_path/'config.json').rename(tmp_path/'other.json');assert processor_identity(tmp_path)!=b
