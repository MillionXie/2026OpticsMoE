from copy import deepcopy
import json

import pytest

from LightGenV2.tasks.t03_saliency.prepare_semantic_targets import prepare, semantic_records
from LightGenV2.tasks.t03_saliency.prepare_unlabeled_pool import file_sha, ids_sha


def fixture():
    categories = [{'id': 2*i+1, 'name': f'category_{i}'} for i in range(80)]
    annotations = {'categories': categories, 'images': [{'id': i} for i in (10, 20, 30)],
        'annotations': [
            {'id': 1, 'image_id': 10, 'category_id': 3, 'iscrowd': 1},
            {'id': 2, 'image_id': 10, 'category_id': 3, 'iscrowd': 0},
            {'id': 3, 'image_id': 20, 'category_id': 1, 'iscrowd': 0}]}
    pool = {'schema_version': 1, 'status': 'audited_images_only_not_teacher_predictions',
            'images': [{'image_id': 10}, {'image_id': 30}],
            'selected_count': 2, 'selected_ids_sha256': ids_sha([10,30])}
    return annotations, pool


def test_semantic_targets_respect_sparse_categories_order_crowd_and_empty_images():
    a, p = fixture()
    result = semantic_records(a, p)
    assert result['records'] == [
        {'sample_id': 'unlabeled/coco2017/000000000010', 'image_id': 10, 'positive_category_ids': [3]},
        {'sample_id': 'unlabeled/coco2017/000000000030', 'image_id': 30, 'positive_category_ids': []}]
    assert result['category_positive_image_counts'][:2] == [0,1]
    assert result['images_without_annotated_instances'] == 1
    assert len(result['categories']) == 80


@pytest.mark.parametrize('failure', ['missing_image','category','duplicate_annotation','duplicate_selected'])
def test_invalid_annotation_or_pool_identity_is_rejected(failure):
    a, p = fixture()
    if failure == 'missing_image': a['images'].pop()
    if failure == 'category': a['annotations'][0]['category_id'] = 2
    if failure == 'duplicate_annotation': a['annotations'].append(deepcopy(a['annotations'][0]))
    if failure == 'duplicate_selected': p['images'][1]['image_id'] = 10
    with pytest.raises(ValueError): semantic_records(a,p)


def test_preparer_checks_hashes_excludes_salicon_and_refuses_overwrite(tmp_path):
    a,p = fixture()
    for split,folder,image_id in [('train','train',100),('test','val',200)]:
        root=tmp_path/'SALICON/images'/folder;root.mkdir(parents=True)
        (root/f'{image_id:012d}.jpg').write_bytes(b'filename-only exclusion fixture')
        p[f'salicon_{split}_count']=1;p[f'salicon_{split}_ids_sha256']=ids_sha([image_id])
    ann,manifest,out=tmp_path/'annotations.json',tmp_path/'pool.json',tmp_path/'targets.json'
    ann.write_text(json.dumps(a));manifest.write_text(json.dumps(p))
    def run(expected=None):
        return prepare(ann,expected or file_sha(ann),manifest,file_sha(manifest),tmp_path/'SALICON',out)
    with pytest.raises(ValueError,match='annotation SHA'):run('0'*64)
    report=run();assert report['samples']==2 and report['sha256']==file_sha(out)
    saved=json.loads(out.read_text())
    assert saved['additional_human_semantic_supervision'] and saved['no_saliency_or_fixation_targets_created']
    with pytest.raises(FileExistsError):run()
    # A fresh output must still reject a target-pool ID in either SALICON split.
    p['images'][0]['image_id']=100;manifest.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='overlaps SALICON'):
        prepare(ann,file_sha(ann),manifest,file_sha(manifest),tmp_path/'SALICON',tmp_path/'rejected.json')
