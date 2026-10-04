import csv
from pathlib import Path

import pytest
import torch

from LightGenV2.tasks.t08_abo_image_text_retrieval.cache_preflight import (
    expected_identity, validate_teacher_cache,
)


@pytest.fixture
def cache_fixture(tmp_path):
    for stem, count in [('train', 4800), ('test', 2400), ('titles', 100), ('manifest', 7200)]:
        with (tmp_path / (stem + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['sample_id', 'product_id'])
            writer.writeheader()
            writer.writerows({'sample_id': f'{stem}_{i}', 'product_id': f'p{i % 100}'}
                             for i in range(count))
    raw = {'dataset': {'dataset_root': str(tmp_path)},
           'qwen': {'model_id': str(tmp_path / 'model')}}
    payload = {'identity': expected_identity(raw),
               'train': torch.ones(4800, 64), 'test': torch.ones(2400, 64),
               'titles': torch.ones(100, 64)}
    return raw, payload, tmp_path / 'cache.pt'


def test_valid_cache_is_read_only(cache_fixture):
    raw, payload, path = cache_fixture
    torch.save(payload, path)
    before = path.read_bytes()
    assert validate_teacher_cache(raw, path) == payload['identity']
    assert path.read_bytes() == before


def test_missing_cache_stops_before_torch_load(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('must not load'))
    with pytest.raises(FileNotFoundError, match='no model load'):
        validate_teacher_cache({}, tmp_path / 'missing.pt')


@pytest.mark.parametrize('field,value', [
    ('model_id', 'Qwen/Qwen3-VL-Embedding-2B'),
    ('retrieval_direction', 'image_to_text'),
    ('image_instruction', 'a different prompt'),
])
def test_incompatible_identity_rejected(cache_fixture, field, value):
    raw, payload, path = cache_fixture
    payload['identity'][field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match='identity differs'):
        validate_teacher_cache(raw, path)


def test_changed_dataset_rejected(cache_fixture):
    raw, payload, path = cache_fixture
    torch.save(payload, path)
    with (Path(raw['dataset']['dataset_root']) / 'manifest.csv').open('a') as stream:
        stream.write('changed,p0\n')
    with pytest.raises(ValueError, match='identity differs'):
        validate_teacher_cache(raw, path)


def test_changed_order_rejected(cache_fixture):
    raw, payload, path = cache_fixture
    payload['identity']['test_ids'].reverse()
    torch.save(payload, path)
    with pytest.raises(ValueError, match='identity differs'):
        validate_teacher_cache(raw, path)


@pytest.mark.parametrize('value', [torch.ones(1, 64), torch.ones(100, 64, dtype=torch.int64),
                                  torch.full((100, 64), float('nan'))])
def test_invalid_tensor_rejected(cache_fixture, value):
    raw, payload, path = cache_fixture
    payload['titles'] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match='shape mismatch|finite floating'):
        validate_teacher_cache(raw, path)
