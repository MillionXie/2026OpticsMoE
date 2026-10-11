import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'prepare_butterfly.py'
spec = importlib.util.spec_from_file_location('butterfly_preparation', path)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_view_and_alias_grouping():
    prefix = 'Butterfly_imbalanced_train_Heliconius_erato_42_'
    assert p.specimen(prefix + 'CAM017773_d_whitestandard.JPG') == p.specimen(prefix + 'CAM017773_v.JPG')
    assert p.specimen(prefix + 'CS1669_Dorsal.jpeg') == p.specimen(prefix + 'CS001669_v.JPG')
    assert p.specimen(prefix + 'CS-CM_4_dorsal.jpeg') == p.specimen(prefix + 'CS-CM4_ventral.jpeg')
    assert p.specimen(prefix + '16426d.JPG') == p.specimen(prefix + '16426v.JPG')
    assert p.specimen(prefix + '16042_H_m_hybrid_V.JPG.jpg') == '16042'


def test_transitive_groups_keep_every_record_and_every_class():
    rows = []
    for label in range(3):
        for i in range(20):
            rows.append(dict(label=label, specimen=f'{label}s{i}', original_hash=f'{label}o{i}', cache_hash=f'{label}c{i}'))
    rows.extend([dict(rows[0], cache_hash='extra'), dict(rows[1], specimen='extra', cache_hash='extra')])
    split, groups = p.grouped_split(rows, 20261011)
    assert groups == 59
    assert sorted(i for v in split.values() for i in v) == list(range(62))
    assert len({key for key, v in split.items() if any(i in v for i in [0, 1, 60, 61])}) == 1
    assert all({rows[i]['label'] for i in v} == {0, 1, 2} for v in split.values())
    assert split == p.grouped_split(rows, 20261011)[0]


def test_conflicting_specimen_labels_fail():
    with pytest.raises(AssertionError, match='Conflicting grouped labels'):
        p.grouped_split([dict(label=i, specimen='same', original_hash=str(i), cache_hash=str(i)) for i in range(2)], 17)
