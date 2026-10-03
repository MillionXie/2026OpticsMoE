import pytest
from LightGenV2.tasks.t03_saliency.prepare_unlabeled_pool import curate, image_index, ids_sha


def pool(tmp_path):
    coco, salicon = tmp_path/'coco', tmp_path/'salicon'
    coco.mkdir()
    for split, image_id in [('train', 1), ('val', 2)]:
        path = salicon/'images'/split
        path.mkdir(parents=True)
        (path/f'COCO_{image_id:012}.jpg').write_bytes(f'image{image_id}'.encode())
    for image_id in range(1, 7):
        (coco/f'{image_id:012}.jpg').write_bytes(f'image{image_id}'.encode())
    return coco, salicon


def test_excludes_test_train_ids_and_content_aliases(tmp_path):
    coco, salicon = pool(tmp_path)
    (coco/'000000000003.jpg').write_bytes(b'image2')  # renamed test file
    (coco/'000000000006.jpg').write_bytes(b'image5')  # duplicate selected image
    report = curate(coco, salicon, count=4, expected_counts=(1,1))
    assert [r['image_id'] for r in report['images']] == [4,5]
    assert report['selected_count'] == 2
    assert report['overlap_with_salicon_test_before_exclusion'] == 1
    assert report['selected_ids_sha256'] == ids_sha([5,4])
    assert len(report['rejected_content_duplicates']) == 2
    assert not report['teacher_predictions_generated']
    assert report['new_image_bytes_written'] == 0


def test_sampling_is_reproducible_and_full_split_is_required(tmp_path):
    coco, salicon = pool(tmp_path)
    a = curate(coco, salicon, count=2, seed=7, expected_counts=(1,1))
    b = curate(coco, salicon, count=2, seed=7, expected_counts=(1,1))
    assert a == b
    with pytest.raises(ValueError): curate(coco, salicon, count=2)
    with pytest.raises(ValueError): curate(coco, salicon, count=5, expected_counts=(1,1))
    with pytest.raises(ValueError): curate(coco, salicon, count=0, expected_counts=(1,1))


def test_duplicate_ids_and_empty_roots_rejected(tmp_path):
    coco, salicon = pool(tmp_path)
    (coco/'COCO_000000000001.jpg').write_bytes(b'alias')
    with pytest.raises(ValueError): image_index(coco)
    with pytest.raises(FileNotFoundError): image_index(tmp_path/'missing')
    empty = tmp_path/'empty'
    empty.mkdir()
    with pytest.raises(ValueError): image_index(empty)
