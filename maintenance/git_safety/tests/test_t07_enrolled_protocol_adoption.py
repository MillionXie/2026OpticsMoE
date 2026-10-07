"""Original enrolled split helper: synthetic files only, no images/model/CCD evaluation."""
import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from LightGenV2.tasks.t07_abo_image_retrieval.standalone import enrolled_abo as module
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retail_sources import shape_groups

ROOT = Path(__file__).resolve().parents[3]


def samples(root, products=('a', 'b'), views=12):
    result = []
    for product in products:
        for index in range(views):
            path = root / f'{product}_{index}.jpg'
            path.write_bytes(f'synthetic-{product}-{index}'.encode())
            result.append(SimpleNamespace(sample_id=f'{product}_{index}', product_id=product,
                          category_id=0, image_path=path, split='train' if product == 'a' else 'test'))
    return result


def test_helper_remains_exact_original_protected_server_source():
    path = ROOT / 'LightGenV2/tasks/t07_abo_image_retrieval/standalone/enrolled_abo.py'
    payload = path.read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(payload).hexdigest() == '0344436032f2e492f6af602ad95fe6afff1d85ce74d740896bacfc4185137f4e'


def test_all_products_retained_eight_four_split_independent_of_input_order(tmp_path):
    source = samples(tmp_path)
    rows = module.split_products(source, tmp_path)
    assert rows == module.split_products(list(reversed(source)), tmp_path)
    groups = shape_groups(rows)
    assert Counter(row['product_id'] for row in groups['train']) == {'a': 8, 'b': 8}
    assert Counter(row['product_id'] for row in groups['query']) == {'a': 4, 'b': 4}
    assert len(groups['gallery']) == len(groups['train']) == 16
    assert not {row['sample_id'] for row in groups['train']} & {row['sample_id'] for row in groups['query']}
    assert not {row['image_sha256'] for row in groups['train']} & {row['image_sha256'] for row in groups['query']}
    assert {row['original_split'] for row in rows} == {'train', 'test'}


def test_split_is_hash_order_not_a_model_score(tmp_path):
    source = samples(tmp_path, products=('a',))
    ordered = sorted(source, key=lambda s: hashlib.sha256(('abo-enrolled42:' + s.sample_id).encode()).hexdigest())
    rows = module.split_products(source, tmp_path)
    assert [row['sample_id'] for row in rows] == [s.sample_id for s in ordered]
    assert [row['split'] for row in rows] == ['train'] * 8 + ['query'] * 4


@pytest.mark.parametrize('views', [11, 13])
def test_invalid_photo_count_is_not_silently_dropped(tmp_path, views):
    with pytest.raises(ValueError, match='Require12'):
        module.split_products(samples(tmp_path, products=('a',), views=views), tmp_path)


def test_original_image_duplicates_across_roles_are_rejected(tmp_path):
    source = samples(tmp_path, products=('a',))
    for sample in source:
        sample.image_path.write_bytes(b'exact-duplicate')
    with pytest.raises(ValueError, match='duplicate'):
        shape_groups(module.split_products(source, tmp_path))


def test_existing_protocol_is_preserved_before_reading_original_dataset(tmp_path):
    output = tmp_path / 'protocol'
    output.mkdir()
    marker = output / 'protocol.json'
    marker.write_bytes(b'preserved')
    with patch.object(module, '_load_contract', side_effect=AssertionError('must not read data')):
        with pytest.raises(FileExistsError):
            module.prepare(tmp_path / 'missing_data', output)
    assert marker.read_bytes() == b'preserved'
