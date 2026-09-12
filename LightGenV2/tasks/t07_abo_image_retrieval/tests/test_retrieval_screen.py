from collections import Counter
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import coil_records, validate_rows, rank_instances


def test_coil_fixed_complete_split_and_angular_gap():
    rows = coil_records()
    assert rows == coil_records()
    assert Counter(r['split'] for r in rows) == dict(train=4320, gallery=160, query=320, unused_test_view=2400)
    g = validate_rows(rows)
    assert len({r['product_id'] for r in g['train']}) == 60
    assert len({r['product_id'] for r in g['query']}) == 40
    for query in g['query']:
        angles = [r['angle'] for r in g['gallery'] if r['product_id'] == query['product_id']]
        assert min(min(abs(query['angle']-a), 360-abs(query['angle']-a)) for a in angles) == 30


def test_identity_leakage_is_rejected():
    rows = coil_records()
    train = next(r for r in rows if r['split'] == 'train')
    query = next(r for r in rows if r['split'] == 'query')
    query['product_id'] = train['product_id']
    with pytest.raises(ValueError, match='leakage'):
        validate_rows(rows)


def test_ranking_uses_identity_not_nearby_category_or_query_itself():
    rows = [dict(sample_id='ga', product_id='a', split='gallery'),
            dict(sample_id='gb', product_id='b', split='gallery'),
            dict(sample_id='qa', product_id='a', split='query')]
    z = torch.zeros(3, 64)
    z[0, 0] = 1; z[1:, 1] = 1
    metrics, predictions = rank_instances(z, rows)
    assert metrics['hit_at_1'] == 0
    assert metrics['hit_at_5'] == 1
    assert predictions[0]['top1_sample_id'] == 'gb'
    z[2] = z[0]
    assert rank_instances(z, rows)[0]['hit_at_1'] == 1


@pytest.mark.parametrize('bad', ['zero', 'nan', 'shape'])
def test_invalid_vectors_fail(bad):
    rows = [dict(sample_id='g', product_id='a', split='gallery'), dict(sample_id='q', product_id='a', split='query')]
    z = torch.ones(2, 64)
    if bad == 'zero': z.zero_()
    elif bad == 'nan': z[0, 0] = float('nan')
    else: z = z[:, :32]
    with pytest.raises(ValueError): rank_instances(z, rows)
