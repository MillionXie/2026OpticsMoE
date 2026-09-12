from dataclasses import replace
from pathlib import Path
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.data import Sample
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.vision_teacher import (
    PREPROCESSING,validate_cache,processor_identity,patch_cosine_loss,patch_step_weight,merged_vision,
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


def test_periodic_weight_preserves_epoch_mean_and_default_is_off():
    cfg=dict(vision_patch_teacher_weight=.2,vision_patch_teacher_every=4,vision_patch_teacher_warmup_epochs=3)
    for epoch in (1,2,3,16):
        weights=[patch_step_weight(cfg,epoch,i) for i in range(128)]
        assert sum(w>0 for w in weights)==32
        assert sum(weights)/128==pytest.approx(.2*min(1.,epoch/3))
        assert all(patch_step_weight({},epoch,i)==0 for i in range(128))


@pytest.mark.parametrize('change',[
    {'vision_patch_teacher_weight':-1.},{'vision_patch_teacher_weight':float('nan')},
    {'vision_patch_teacher_every':0},{'vision_patch_teacher_warmup_epochs':1.5},
])
def test_bad_patch_schedule_rejected(change):
    with pytest.raises(ValueError):patch_step_weight(change,1,0)


def test_visual_profile_only_adds_training_supervision():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config
    a=overlay_config({},'domain_distill_teacher_continue');b=overlay_config({},'domain_distill_vision_patch')
    ignored={'vision_patch_teacher_weight','vision_patch_teacher_every','vision_patch_teacher_warmup_epochs','protocol'}
    assert {k:v for k,v in a.items() if k not in ignored}=={k:v for k,v in b.items() if k not in ignored}
    assert b['vision_patch_teacher_weight']==.2 and b['vision_patch_teacher_every']==4
    assert b.get('retrieval_head','linear64')=='linear64' and b['sam_rho']==0


def test_existing_v_only_helper_preserves_gradient_and_requires_matching_grid():
    from types import SimpleNamespace
    pixels=torch.randn(2,49,2048,requires_grad=True)
    frontend=SimpleNamespace(patches=lambda x,n:x,merge=lambda x:x.reshape(-1,2048))
    model=SimpleNamespace(metadata={'input_preprocessing':'contain_white'},frontend=frontend,vision=lambda x:x)
    batch=dict(input_ids=torch.zeros(2,77,dtype=torch.long),image_grid_thw=torch.tensor([[1,14,14]]*2),pixel_values=pixels)
    result=merged_vision(model,batch);assert result.shape==(2,49,2048)
    result.square().mean().backward();assert pixels.grad is not None and pixels.grad.abs().sum()>0
    batch['image_grid_thw']=torch.tensor([[1,14,16]]*2)
    with pytest.raises(ValueError):merged_vision(model,batch)


@pytest.mark.parametrize('profiles',[
    ['domain_distill_vision_patch'],['build_vision_teacher_cache','domain_distill_vision_patch'],
])
def test_queue_rejects_missing_visual_cache_or_teacher_before_creating_run(tmp_path,monkeypatch,profiles):
    import sys
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone import generalization_queue
    output=tmp_path/'run'
    argv=['queue','--gpu','fake','--assets','a','--checkpoint','c','--target','t','--abo','b','--pool','p',
          '--output',str(output),'--teacher-cache','global','--teacher-alignment','alignment','--profiles',*profiles]
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(SystemExit) as exc:generalization_queue.main()
    assert exc.value.code==2 and not output.exists()
