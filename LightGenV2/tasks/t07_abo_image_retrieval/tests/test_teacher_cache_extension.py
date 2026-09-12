import copy
from dataclasses import replace
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.data import Sample
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import sha256
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.teacher_relations import reusable_training_vectors
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config, PROFILES


def fixture(tmp_path):
    samples=[Sample(f's{i}',f'p{i}',0,'chair','train',tmp_path/f'{i}.png') for i in range(3)]
    for s in samples:s.image_path.write_bytes(s.sample_id.encode())
    cache=dict(schema=1,frozen_teacher=True,teacher_trainable_parameters=0,
        target_manifest_sha256='a'*64,pool_manifest_sha256='b'*64,model='/teacher',prompt='fixed',
        preprocessing='EXIF RGB; native aspect; processor min=max pixels 50176',
        excluded_splits=['val','test'],student_inference_teacher_required=False,
        ids=['s0','old_unused'],image_sha256=[sha256(samples[0].image_path),'c'*64],
        vectors=torch.ones(2,2048,dtype=torch.float16))
    path=tmp_path/'old_cache.pt';torch.save(cache,path)
    return samples,cache,path


def reuse(path,samples):
    return reusable_training_vectors(path,sha256(path),samples,'a'*64,'/teacher','fixed')


def test_verified_intersection_not_blind_append(tmp_path):
    samples,cache,path=fixture(tmp_path)
    vectors,audit=reuse(path,samples)
    assert list(vectors)==['s0'] and not vectors['s0'].requires_grad
    torch.testing.assert_close(vectors['s0'],cache['vectors'][0],atol=0,rtol=0)
    assert audit['reused_images']==1 and audit['new_teacher_forwards']==2
    assert audit['source_images_not_in_new_pool']==1
    with pytest.raises(ValueError,match='SHA'):
        reusable_training_vectors(path,'0'*64,samples,'a'*64,'/teacher','fixed')
    with pytest.raises(ValueError,match='training-only'):
        reuse(path,[replace(samples[0],split='test')]+samples[1:])
    with pytest.raises(ValueError,match='training-only'):reuse(path,samples+samples[:1])
    samples[0].image_path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='content changed'):reuse(path,samples)


@pytest.mark.parametrize('key,value',[
    ('target_manifest_sha256','z'*64),('model','/other'),('prompt','different'),
    ('excluded_splits',[]),('teacher_trainable_parameters',1),('frozen_teacher',False),
    ('student_inference_teacher_required',True),('preprocessing','crop'),
    ('ids',['s0','s0']),('vectors',torch.ones(2,64,dtype=torch.float16)),
    ('vectors',torch.full((2,2048),float('nan'),dtype=torch.float16)),
    ('vectors',torch.zeros(2,2048,dtype=torch.float16)),
])
def test_reuse_rejects_incompatible_identity(tmp_path,key,value):
    samples,cache,path=fixture(tmp_path);bad=copy.deepcopy(cache);bad[key]=value;torch.save(bad,path)
    with pytest.raises(ValueError):reuse(path,samples)


def test_refit_data_pair_has_same_training_recipe():
    a=overlay_config({},'domain_distill_refit250');b=overlay_config({},'domain_distill_refit500')
    assert all(p in PROFILES for p in ('domain_distill_refit250','domain_distill_refit500'))
    assert a['expected_pool_products_per_category']==250 and b['expected_pool_products_per_category']==500
    ignored={'expected_pool_products_per_category','protocol'}
    assert {k:v for k,v in a.items() if k not in ignored}=={k:v for k,v in b.items() if k not in ignored}
    assert 'teacher_alignment_sha256' not in a and 'teacher_alignment_origin_checkpoint_sha256' not in a
    assert a['teacher_feature_weight']==2 and a['learning_rate_multiplier']==.25
    assert a['adapt']['steps']==250
    assert a['preserve_restored_category_proxies'] and len(a['restore_auxiliary_source_sha256'])==64
