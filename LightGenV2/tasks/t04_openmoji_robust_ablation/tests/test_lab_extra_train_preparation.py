import hashlib
import json
import sys

import pytest

from LightGenV2.tasks.t04_openmoji_robust_ablation import lab_prepare_extra_train1000 as first
from LightGenV2.tasks.t04_openmoji_robust_ablation import lab_prepare_extra2_train1000 as second


def test_original_two_additions_are_disjoint_and_never_overwrite(tmp_path, monkeypatch):
    project = tmp_path / 'bench'
    data = tmp_path / 'OpenMoji_Lab_SHS_8um' / 'data'
    data.mkdir(parents=True)
    project.mkdir()
    def rows(prefix, count):
        result = []
        for i in range(count):
            name = f'{prefix}_{i:04d}'
            folder = data / name
            folder.mkdir()
            (folder / 'source.bin').write_bytes(name.encode())
            result.append(dict(sample_id=name, relative_dir=name,
                               task=('add', 'replace', 'move', 'remove')[i % 4],
                               files={'source': 'source.bin'}))
        return result
    def manifest(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in values), encoding='utf-8')
    train = rows('train', 3000)
    test = rows('test', 1000)
    manifest(data / 'train.jsonl', train)
    manifest(data / 'test.jsonl', test)
    manifest(project / 'data_train_adapt1000' / 'capture_train.jsonl', train[:1000])
    (data / 'token_embeddings_v1.pt').write_bytes(b'test fixture only')
    monkeypatch.setattr(sys, 'argv', ['prepare', '--project', str(project)])
    first.main()
    second.main()
    identities = []
    for name, seed in [('data_train_extra1000', 1002), ('data_train_extra2_1000', 1003)]:
        audit = json.loads((project / name / 'split_audit.json').read_text())
        assert audit['seed'] == seed and audit['capture_count'] == 1000
        assert audit['tasks'] == dict(add=250, replace=250, move=250, remove=250)
        assert audit['hardware_started'] is False
        assert audit['test_source_overlap'] == audit['prior_train_source_overlap'] == 0
        identities.append(set(audit['fit_ids']))
        for row in map(json.loads, (project / name / 'capture_train.jsonl').read_text().splitlines()):
            original = data / row['relative_dir'] / 'source.bin'
            copied = project / name / row['relative_dir'] / 'source.bin'
            assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(copied.read_bytes()).digest()
    assert not identities[0] & identities[1]
    before = (project / 'data_train_extra1000' / 'capture_train.jsonl').read_bytes()
    with pytest.raises(AssertionError, match='overwrite'):
        first.main()
    assert (project / 'data_train_extra1000' / 'capture_train.jsonl').read_bytes() == before
