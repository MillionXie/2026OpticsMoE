import copy
import random
import pytest
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retail_sources import image_members, selected_categories, shape_groups, off_inventory, SHAPE_PROTOCOL
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import fitting_groups, training_pairs
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config


def fixture_rows():
    return [dict(sample_id=f'{role}/{sku}/{i}', image_path=f'{role}/{sku}/{i}',
                 image_sha256=f'{role}-{sku}-{i}', product_id=sku, split=role)
            for role, count in [('train', 3), ('query', 1)] for sku in ('1/1', '1/2') for i in range(count)]


def test_shape_sku_is_category_scoped_and_extraction_safe():
    rows = image_members(['training_set/1/2/a.jpg', 'training_set/2/2/b.jpg', 'setup.py'], 'training_set')
    assert [r['product_id'] for r in rows] == ['1/2', '2/2']
    with pytest.raises(ValueError):
        image_members(['../1/2/a.jpg'], 'training_set')


def test_shape_subset_fixed_independent_of_order():
    values = [str(i) for i in range(62)]
    assert selected_categories(values) == selected_categories(reversed(values))
    assert len(selected_categories(values)) == 8


def test_shape_shared_gallery_not_test_training():
    groups = shape_groups(fixture_rows())
    fit = fitting_groups({'protocol': SHAPE_PROTOCOL}, groups)
    assert len(groups['gallery']) == 6 and len(fit['gallery']) == 2
    assert not {r['sample_id'] for r in fit['train']} & {r['sample_id'] for r in fit['gallery']}
    assert all(r['source_split'] == 'train' for r in fit['gallery'])
    batch, labels = training_pairs(fit, random.Random(42), 2)
    assert len(batch) == len(labels) == 4
    assert not {r['sample_id'] for r in batch} & {r['sample_id'] for r in groups['query']}


def test_shape_refuses_leaked_or_missing_positives():
    rows = fixture_rows()
    rows[-1]['image_sha256'] = rows[0]['image_sha256']
    with pytest.raises(ValueError, match='duplicate'):
        shape_groups(rows)
    rows = fixture_rows()
    rows[-1]['product_id'] = 'unseen_without_reference'
    with pytest.raises(ValueError, match='positives'):
        shape_groups(rows)


def test_off_front_languages_do_not_create_independent_photos():
    rows = off_inventory([dict(code='123', images={
        '1': {}, '2': {}, 'front_en': {'imgid': '1'}, 'front_fr': {'imgid': 1},
        'nutrition_en': {'imgid': '2'}})])
    assert rows[0]['independent_selected_front_ids'] == ['1']
    assert not rows[0]['potential_front_pair']


def test_recovery_pair_only_changes_sam_and_description():
    base = {'adapt': {}, 'augmentation': {}}
    a = overlay_config(copy.deepcopy(base), 'recovery_phase05_adam')
    b = overlay_config(copy.deepcopy(base), 'recovery_phase05_sam')
    assert a.pop('sam_rho') == 0 and b.pop('sam_rho') == .03
    assert a == b and a['track_clean_train'] and a['independent_phase_dropout']
