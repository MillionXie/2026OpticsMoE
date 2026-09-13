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


def test_enrollment_view_counts_nested_without_dropping_products():
    from types import SimpleNamespace
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.catalog_view_audit import view_indices
    samples = [SimpleNamespace(product_id=p, sample_id=f'{p}-{i}') for p in ('a', 'b') for i in range(12)]
    selected = [view_indices(samples, k) for k in (1, 3, 12)]
    assert set(selected[0]) <= set(selected[1]) <= set(selected[2])
    assert [len(x) for x in selected] == [2, 6, 24]
    assert all({samples[i].product_id for i in indices} == {'a', 'b'} for indices in selected)


def test_budget_centroids_matches_original12_but_supports1_and3():
    import torch
    from types import SimpleNamespace
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.data import _gallery_centroids
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.catalog_view_audit import budget_centroids
    samples = [SimpleNamespace(product_id='a', category_id=0, category_name='a') for _ in range(12)]
    z = torch.nn.functional.normalize(torch.arange(1, 769).reshape(12, 64).float(), dim=1)
    a, _ = budget_centroids(samples, z, 12)
    b, _ = _gallery_centroids(samples, z)
    assert torch.equal(a, b)
    for count in (1, 3):
        result, meta = budget_centroids(samples[:count], z[:count], count)
        assert result.shape == (1, 64) and meta[0].view_count == count
    with pytest.raises(RuntimeError):
        _gallery_centroids(samples[:1], z[:1])


def test_instance_protocol_retains_all_products_but_no_same_photo():
    from types import SimpleNamespace
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.catalog_view_audit import instance_rows
    samples = [SimpleNamespace(product_id=str(p), sample_id=f'{p}-{i}') for p in range(40) for i in range(12)]
    rows, indices = instance_rows(samples)
    assert sorted(indices) == list(range(480))
    gallery = [r for r in rows if r['split'] == 'gallery']
    query = [r for r in rows if r['split'] == 'query']
    assert len(gallery) == 160 and len(query) == 320
    assert {r['product_id'] for r in gallery} == {r['product_id'] for r in query}
    assert not {r['sample_id'] for r in gallery} & {r['sample_id'] for r in query}
