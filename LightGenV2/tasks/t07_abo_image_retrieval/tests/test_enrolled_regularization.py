import csv
import json
import random

import pytest
import torch
from PIL import Image

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.enrolled_regularization import (
    load_external_pool, load_external_relations, augment_whole_object, curriculum_epoch, curriculum_loss_weights)
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import sha256
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import training_pairs
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.enrolled_regularization import fitting_bank_diagnostics


def test_bank_metric_excludes_self_and_is_read_only_chunk_independent():
    z = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    ids = ['a', 'b', 'c', 'd']
    before = z.detach().clone(); rng = torch.get_rng_state().clone()
    a = fitting_bank_diagnostics(z, labels, ids, 1)
    b = fitting_bank_diagnostics(z, labels, ids, 4)
    assert a == b and a['hit_at_1'] == 0.
    assert a['median_best_positive_minus_negative_cosine'] == -1.
    assert a['candidates_per_query'] == 3 and a['query_count'] == 4
    assert a['used_for_checkpoint_selection'] is False
    assert torch.equal(z, before) and z.grad is None and torch.equal(rng, torch.get_rng_state())
    good = fitting_bank_diagnostics(torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]]), labels, ids, 3)
    assert good['hit_at_1'] == 1. and good['product_count'] == 2


@pytest.mark.parametrize('bad', ['duplicate', 'zero', 'nan', 'singleton', 'single_sku', 'chunk'])
def test_bank_metric_rejects_invalid_self_retrieval_inputs(bad):
    z = torch.ones(4, 2); labels = torch.tensor([0, 0, 1, 1]); ids = list('abcd'); chunk = 2
    if bad == 'duplicate': ids[-1] = ids[0]
    if bad == 'zero': z[0] = 0
    if bad == 'nan': z[0, 0] = float('nan')
    if bad == 'singleton': labels[-1] = 2
    if bad == 'single_sku': labels[:] = 0
    if bad == 'chunk': chunk = 0
    with pytest.raises(ValueError): fitting_bank_diagnostics(z, labels, ids, chunk)


def external_fixture(tmp_path):
    rows=[]
    for i in range(4):
        p=tmp_path/f'{i}.png'; Image.new('RGB',(16,16),(i*50,30,80)).save(p)
        rows.append(dict(sample_id=str(i),product_id=f'ext{i//2}',category_id=0,
                         image_path=p.name,image_sha256=sha256(p)))
    path=tmp_path/'manifest.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    digest=sha256(path)
    (tmp_path/'report.json').write_text(json.dumps(dict(manifest_sha256=digest,
        target_manifest_sha256='parent',selected_images=4,selected_products=2)))
    protocol=dict(parent_manifest_sha256='parent')
    groups=dict(train=[dict(product_id='target',image_sha256='a')],query=[dict(product_id='target',image_sha256='b')])
    return digest,protocol,groups,rows


def test_external_instance_pool_and_distinct_photo_pairs(tmp_path):
    digest,protocol,groups,_=external_fixture(tmp_path)
    fit,audit=load_external_pool(tmp_path,tmp_path,protocol,groups,digest)
    assert audit['products']==2 and audit['target_product_overlap']==0
    rows,labels=training_pairs(fit,random.Random(3),2,1.)
    assert len(set(labels.tolist()))==2
    assert all(rows[i]['product_id']==rows[i+2]['product_id'] and rows[i]['sample_id']!=rows[i+2]['sample_id'] for i in range(2))


@pytest.mark.parametrize('collision',['product','hash','changed','manifest'])
def test_external_rejects_overlap_or_changed_data(tmp_path,collision):
    digest,protocol,groups,rows=external_fixture(tmp_path)
    if collision=='product':groups['query'][0]['product_id']='ext0'
    if collision=='hash':groups['query'][0]['image_sha256']=rows[0]['image_sha256']
    if collision=='changed':Image.new('RGB',(16,16),'red').save(tmp_path/'0.png')
    if collision=='manifest':digest='wrong'
    with pytest.raises(ValueError):load_external_pool(tmp_path,tmp_path,protocol,groups,digest)


def test_schedule_external_then_target_no_boundary_overlap():
    assert [curriculum_epoch(e,2,3) for e in range(1,6)]==[(True,1,2),(True,2,2),(False,1,3),(False,2,3),(False,3,3)]
    assert curriculum_epoch(1,0,3)==(False,1,3)
    with pytest.raises(ValueError):curriculum_epoch(6,2,3)


def test_whole_object_augmentation_is_deterministic_and_keeps_corners():
    image=Image.new('RGB',(224,224),'black')
    a=augment_whole_object(image,random.Random(42));b=augment_whole_object(image,random.Random(42))
    assert a.size==(224,224) and a.tobytes()==b.tobytes()
    # Uncropped complete black rectangle is contained strictly inside white canvas.
    assert a.getpixel((0,0))[0]>200
    assert a.getpixel((112,112))[0]<30


def test_mild_ablation_only_differs_by_sam_and_has_no_geometry():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import PROFILES
    a=dict(PROFILES['sku_mild_adamw']);b=dict(PROFILES['sku_mild_sam'])
    assert a.pop('sam_rho')==0 and b.pop('sam_rho')==.002 and a==b
    image=Image.new('RGB',(224,224),'black')
    image.putpixel((0,0),(255,255,255))
    out=augment_whole_object(image,random.Random(42),mild=True)
    assert out.getpixel((0,0))[0]>200 and out.getpixel((1,1))[0]<30


@pytest.mark.parametrize('name,key,value', [
    ('sku_augmentation_only', 'mild_augmentation', False),
    ('sku_phase_dropout_only', 'phase_dropout', .03),
])
def test_isolated_regularization_profiles_change_one_setting(name,key,value):
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import PROFILES
    base=PROFILES['sku_capacity_control']
    candidate=PROFILES[name]
    assert candidate is not base
    assert {k for k in base.keys() | candidate.keys() if base.get(k)!=candidate.get(k)}=={key}
    assert candidate[key]==value
    assert candidate['sam_rho']==0 and candidate['teacher_weight']==0
    assert 'electronic_expansion' not in candidate and 'head_expansion' not in candidate


def teacher_fixture(tmp_path):
    digest,protocol,groups,rows=external_fixture(tmp_path)
    fit,_=load_external_pool(tmp_path,tmp_path,protocol,groups,digest)
    protocol['rows']=[dict(sample_id='old_target_now_query',product_id='target',image_sha256='protected')]
    cache=dict(schema=1,frozen_teacher=True,teacher_trainable_parameters=0,
        target_manifest_sha256='parent',pool_manifest_sha256=digest,
        model='/models/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda',
        prompt='Represent this catalog product image for category-aware visual similarity retrieval.',
        preprocessing='EXIF RGB; native aspect; processor min=max pixels 50176',
        ids=['old_target_now_query']+[r['sample_id'] for r in rows],
        image_sha256=['protected']+[r['image_sha256'] for r in rows],
        vectors=torch.randn(5,2048,dtype=torch.float16))
    path=tmp_path/'teacher.pt';torch.save(cache,path)
    return path,cache,fit,protocol,digest


def test_external_teacher_drops_all_target_rows_and_uses_first64(tmp_path):
    path,cache,fit,protocol,pool_sha=teacher_fixture(tmp_path)
    kept,audit=load_external_relations(path,sha256(path),fit,protocol,pool_sha)
    assert set(kept)=={'0','1','2','3'} and audit['discarded_other_cache_rows']==1
    assert torch.allclose(kept['0'],torch.nn.functional.normalize(cache['vectors'][1,:64].float(),dim=0))
    assert all(not v.requires_grad and v.shape==(64,) for v in kept.values())


@pytest.mark.parametrize('bad',['sha','pool','trainable','prompt','hash','zero','duplicate','target'])
def test_external_teacher_rejects_bad_identity_or_target_fitting(tmp_path,bad):
    path,cache,fit,protocol,pool_sha=teacher_fixture(tmp_path)
    if bad=='pool':cache['pool_manifest_sha256']='wrong'
    if bad=='trainable':cache['teacher_trainable_parameters']=1
    if bad=='prompt':cache['prompt']='other task'
    if bad=='hash':cache['image_sha256'][1]='wrong'
    if bad=='zero':cache['vectors'][1,:64]=0
    if bad=='duplicate':cache['ids'][0]=cache['ids'][1]
    if bad=='target':fit['train'][0]['product_id']='target'
    torch.save(cache,path)
    with pytest.raises(ValueError):
        load_external_relations(path,'wrong' if bad=='sha' else sha256(path),fit,protocol,pool_sha)


def test_external_soft_pretraining_never_replaces_target_supervision():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import PROFILES
    p=PROFILES['sku_external_relations']
    for taper in (0.,.5,1.):
        assert curriculum_loss_weights(p,True,taper)==(0.,1.)
        assert curriculum_loss_weights(p,False,taper)==(1.,0.)
    assert p['sam_rho']==0 and p['phase_dropout']==0
