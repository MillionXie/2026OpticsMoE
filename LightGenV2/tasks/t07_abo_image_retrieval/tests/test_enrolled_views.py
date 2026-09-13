import copy

import pytest
import torch

from LightGenV2.tasks.t07_abo_image_retrieval.analysis.enrolled_views import rank_cache, summarize, validate_protocol


def rows():
    return [dict(sample_id='ga', product_id='a', split='train', category_id=0, azimuth='0'),
            dict(sample_id='qa', product_id='a', split='query', category_id=0, azimuth='66'),
            dict(sample_id='gb', product_id='b', split='train', category_id=1, azimuth='12'),
            dict(sample_id='qb', product_id='b', split='query', category_id=1, azimuth='24')]


def test_circular_gap_and_paired_counts():
    a = {key: dict(hit_at_1=v) for key, v in [('qa', 0), ('qb', 1)]}
    b = {key: dict(hit_at_1=v) for key, v in [('qa', 1), ('qb', 1)]}
    r = summarize(rows(), a, b)
    total = next(g for g in r['groups'] if g['dimension'] == 'all')
    assert total['n'] == 2 and total['optical_hits'] == 1 and total['qwen_only'] == 1
    assert [x['nearest_train_gap_index'] for x in r['predictions']] == [6, 12]
    with pytest.raises(ValueError, match='ALL fixed queries'):
        summarize(rows(), {'qa': a['qa']}, b)


def test_cache_order_does_not_change_stable_ties_or_mutate_inputs():
    r = rows()
    z = torch.ones(4, 64)
    cache = dict(ids=[x['sample_id'] for x in r], vectors=z, manifest_sha256='sha')
    before = copy.deepcopy(cache)
    first = rank_cache(r, cache, 'sha')
    order = [3, 2, 1, 0]
    reordered = dict(cache, ids=[cache['ids'][i] for i in order], vectors=z[order])
    assert rank_cache(r, reordered, 'sha') == first
    assert first[1]['qb']['top1_sample_id'] == 'ga'
    assert torch.equal(z, before['vectors']) and cache['ids'] == before['ids']
    with pytest.raises(ValueError, match='SHA'):
        rank_cache(r, cache, 'wrong')
    with pytest.raises(ValueError, match='identity'):
        rank_cache(r, dict(cache, ids=['ga'] * 4), 'sha')


@pytest.mark.parametrize('field,value', [('azimuth', '72'), ('azimuth', '-1'), ('split', 'test')])
def test_protocol_rejects_invalid_views(field, value):
    r = rows()
    r[0][field] = value
    with pytest.raises(ValueError):
        validate_protocol(r)


def test_duplicate_view_or_missing_training_sku_rejected():
    r = rows()
    r[1]['azimuth'] = '0'
    with pytest.raises(ValueError, match='Duplicate view'):
        validate_protocol(r)
    with pytest.raises(ValueError, match='TRAIN gallery'):
        validate_protocol(rows()[1:])
