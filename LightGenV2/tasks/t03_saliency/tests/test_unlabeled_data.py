import hashlib
import json

import pytest
import torch
from PIL import Image

from LightGenV2.tasks.t03_saliency.prepare_unlabeled_pool import curate, file_sha
from LightGenV2.tasks.t03_saliency.unlabeled_data import AuditedUnlabeledImages, UnlabeledTeacherMaps, collate_unlabeled, PREPROCESSING
from LightGenV2.tasks.t03_saliency.export_unlabeled_teacher import checked_cpu_logits


def fixture(tmp_path):
    coco, salicon = tmp_path/'coco', tmp_path/'salicon'
    coco.mkdir()
    for split, image_id in [('train',1),('val',2)]:
        p = salicon/'images'/split
        p.mkdir(parents=True)
        Image.new('RGB',(8,6),(image_id*20,30,40)).save(p/f'COCO_{image_id:012}.jpg')
    for image_id in range(1,5):
        Image.new('RGB',(8,6),(image_id*20,30,40)).save(coco/f'{image_id:012}.jpg')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(curate(coco,salicon,count=2,expected_counts=(1,1))))
    dataset = AuditedUnlabeledImages(manifest,file_sha(manifest),salicon)
    return dataset, manifest, salicon


def test_decode_resize_and_content_validation(tmp_path):
    dataset, manifest, salicon = fixture(tmp_path)
    row = dataset[0]
    assert row['image'].size == (224,224) and row['image'].mode == 'RGB'
    assert row['sample_id'] == 'unlabeled/coco2017/000000000003'
    assert 'density' not in row and 'fixation' not in row
    assert collate_unlabeled([row])['sample_ids'] == [row['sample_id']]
    with pytest.raises(ValueError): AuditedUnlabeledImages(manifest,'0'*64,salicon)
    path = dataset.root/dataset.records[0]['image_file']
    path.write_bytes(b'changed')
    with pytest.raises(ValueError): dataset[0]


@pytest.mark.parametrize('kind',['traversal','split_drift','id_overlap','count'])
def test_rejects_manifest_and_exclusion_drift(tmp_path,kind):
    dataset,manifest,salicon = fixture(tmp_path)
    m = dataset.manifest
    if kind == 'traversal': m['images'][0]['image_file'] = '../other/000000000003.jpg'
    if kind == 'count': m['selected_count'] = 99
    if kind == 'split_drift': (salicon/'images/train/COCO_000000000001.jpg').rename(salicon/'images/train/COCO_000000000009.jpg')
    if kind == 'id_overlap':
        m['images'][0]['image_id'] = 1
        from LightGenV2.tasks.t03_saliency.prepare_unlabeled_pool import ids_sha
        m['selected_ids_sha256'] = ids_sha([1,4])
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError): AuditedUnlabeledImages(manifest,file_sha(manifest),salicon)


def cached(tmp_path,dataset,**overrides):
    manifest = {'split':'extra_unlabeled_excluding_salicon_train_and_test',
                'checkpoint_sha256':'a'*64,'image_manifest_sha256':dataset.manifest_sha256,
                'preprocessing':PREPROCESSING,'ground_truth_available':False,'augmentation':False}
    manifest.update(overrides)
    path = tmp_path/'maps.pt'
    torch.save({'manifest':manifest,'sample_ids':dataset.sample_ids,
                'logits':torch.randn(len(dataset),1,224,224).half()},path)
    return path


def test_teacher_cache_requires_full_provenance_and_exact_ids(tmp_path):
    dataset,_,_ = fixture(tmp_path)
    path = cached(tmp_path,dataset)
    cache = UnlabeledTeacherMaps(path,file_sha(path),dataset,'a'*64)
    ids = dataset.sample_ids[::-1]
    actual = cache.get(ids,'cpu')
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual,cache.values.flip(0).float())
    with pytest.raises(KeyError): cache.get(['validation/3'],'cpu')
    with pytest.raises(ValueError): cache.get([ids[0],ids[0]],'cpu')
    with pytest.raises(ValueError): UnlabeledTeacherMaps(path,'0'*64,dataset,'a'*64)
    with pytest.raises(ValueError): UnlabeledTeacherMaps(path,file_sha(path),dataset,'b'*64)


@pytest.mark.parametrize('change', ['ids','shape','nan','split','gt','preprocessing'])
def test_rejects_bad_teacher_cache(tmp_path,change):
    dataset,_,_ = fixture(tmp_path)
    path = cached(tmp_path,dataset)
    payload = torch.load(path,weights_only=False)
    if change == 'ids': payload['sample_ids'] = payload['sample_ids'][::-1]
    if change == 'shape': payload['logits'] = payload['logits'][:1]
    if change == 'nan': payload['logits'][0,0,0,0] = float('nan')
    if change == 'split': payload['manifest']['split'] = 'train_only'
    if change == 'gt': payload['manifest']['ground_truth_available'] = True
    if change == 'preprocessing': payload['manifest']['preprocessing'] = 'different'
    torch.save(payload,path)
    with pytest.raises(ValueError): UnlabeledTeacherMaps(path,file_sha(path),dataset,'a'*64)


def test_half_precision_export_is_finite_and_detached():
    value = torch.randn(2,1,224,224,requires_grad=True)
    output = checked_cpu_logits(value,2)
    assert output.dtype == torch.float16 and not output.requires_grad
    with pytest.raises(ValueError): checked_cpu_logits(value,1)
    with pytest.raises(ValueError): checked_cpu_logits(torch.full_like(value,1e10),2)
    with pytest.raises(ValueError): checked_cpu_logits(torch.full_like(value,float('nan')),2)
