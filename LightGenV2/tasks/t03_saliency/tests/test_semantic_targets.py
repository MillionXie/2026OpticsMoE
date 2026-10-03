from copy import deepcopy
import json

import pytest

from LightGenV2.tasks.t03_saliency.prepare_semantic_targets import prepare, semantic_records, add_spatial_boxes
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


def spatial_fixture():
    a,p=fixture()
    for image in a['images']:image.update(width=100,height=50)
    a['annotations'][0]['bbox']=[-10,5,40,20]
    a['annotations'][1]['bbox']=[20,20,0,10]
    # Outside selected pool: it must not contribute any regions.
    a['annotations'][2]['bbox']=[0,0,100,50]
    return a,p


def test_regions_are_clipped_normalized_ordered_and_do_not_leak_unselected_images():
    a,p=spatial_fixture();result=add_spatial_boxes(a,semantic_records(a,p))
    assert result['spatial_box_count']==1 and result['spatial_boxes_clipped']==1
    assert result['spatial_boxes_skipped_nonpositive_or_outside']==1
    assert result['records'][0]['boxes_xyxy_unit']==[
        {'annotation_id':1,'category_id':3,'xyxy':[0.,.1,.3,.5]}]
    assert result['records'][1]['boxes_xyxy_unit']==[]
    assert result['records'][0]['source_size_wh']==[100,50]
    assert result['additional_human_box_supervision']


@pytest.mark.parametrize('bad',[None,[0,1,2],[0,0,float('nan'),1],[0,0,float('inf'),1]])
def test_invalid_boxes_rejected(bad):
    a,p=spatial_fixture();a['annotations'][0]['bbox']=bad
    with pytest.raises(ValueError,match='bbox'):add_spatial_boxes(a,semantic_records(a,p))


def test_image_annotation_geometry_checked_before_writing_regions(tmp_path):
    from PIL import Image
    a,p=spatial_fixture();root=tmp_path/'coco';root.mkdir();p['coco_root']=str(root)
    for item in p['images']:
        item['image_file']=f"{item['image_id']:012d}.jpg"
        Image.new('RGB',(100,50)).save(root/item['image_file'])
    for split,folder,image_id in [('train','train',100),('test','val',200)]:
        d=tmp_path/'SALICON/images'/folder;d.mkdir(parents=True)
        (d/f'{image_id:012d}.jpg').write_bytes(b'exclusion fixture')
        p[f'salicon_{split}_count']=1;p[f'salicon_{split}_ids_sha256']=ids_sha([image_id])
    ann=tmp_path/'ann.json';manifest=tmp_path/'pool.json'
    ann.write_text(json.dumps(a));manifest.write_text(json.dumps(p))
    def run(out):return prepare(ann,file_sha(ann),manifest,file_sha(manifest),tmp_path/'SALICON',out,include_boxes=True)
    out=tmp_path/'regions.json';report=run(out)
    assert report['spatial_box_count']==1
    assert json.loads(out.read_text())['purpose']=='training_only_auxiliary_object_regions_not_saliency_ground_truth'
    Image.new('RGB',(99,50)).save(root/p['images'][0]['image_file'])
    with pytest.raises(ValueError,match='dimensions mismatch'):run(tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()
