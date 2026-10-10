"""Data leakage and record-preservation contracts for the new dataset."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_fishnet import grouped_split


def examples():
    return [dict(label=k, original_hash=f'original-{k}-{i}', cache_hash=f'cache-{k}-{i}')
            for k in range(8) for i in range(20)]


def test_both_pixel_spaces_group_transitively_without_dropping_records():
    rows = examples()
    rows[1]['original_hash'] = rows[0]['original_hash']
    rows[2]['cache_hash'] = rows[1]['cache_hash']
    split, count = grouped_split(rows, 20261011)
    owner = {i: key for key, indices in split.items() for i in indices}
    assert len(owner) == len(rows) == 160
    assert owner[0] == owner[1] == owner[2] and count == 158
    assert all({rows[i]['label'] for i in indices} == set(range(8)) for indices in split.values())
    assert grouped_split(rows, 20261011)[0] == split


@pytest.mark.parametrize('kind', ['original_hash', 'cache_hash'])
def test_conflicting_class_for_identical_pixels_is_rejected(kind):
    rows = examples()
    rows[20][kind] = rows[0][kind]
    with pytest.raises(AssertionError, match='conflicting labels'):
        grouped_split(rows, 20261011)
