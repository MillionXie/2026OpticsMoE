from pathlib import Path
import sys
from types import SimpleNamespace
import copy
import json

import pytest
import yaml

from LightGenV2.tasks.t08_abo_image_text_retrieval import text_to_image as entry


def args(tmp_path):
    return ['--model', str(tmp_path/'model'), '--data-root', str(tmp_path/'data'),
            '--checkpoint', str(tmp_path/'body.pt'), '--teacher-cache', str(tmp_path/'cache.pt'),
            '--run-dir', str(tmp_path/'new_run')]


def test_inspect_is_read_only(tmp_path, capsys):
    assert entry.main(args(tmp_path)+['--inspect']) == 0
    assert 'text_to_image' in capsys.readouterr().out
    assert not (tmp_path/'new_run').exists()


def test_profile_architecture_and_prompt_values_are_preserved(tmp_path):
    original = yaml.safe_load(entry.PROFILE.read_text())
    resolved = entry.configuration(entry.PROFILE, model=tmp_path/'model', data_root=tmp_path/'data',
                                   checkpoint=tmp_path/'body.pt', teacher_cache=tmp_path/'cache.pt', run_dir=tmp_path/'new_run')
    for key in ('language_optical', 'optical', 'electronic', 'balanced_fusion', 'router_hardware_measurement'):
        assert resolved.get(key) == original.get(key)
    assert resolved['abo_image_text']['retrieval_direction'] == 'text_to_image'
    assert resolved['abo_image_text']['selection_direction'] == 'text_to_image'
    assert not resolved['abo_image_text']['allow_propagation_distance_transition']
    assert not resolved['abo_image_text']['allow_electronic_compaction_transition']


def test_modified_profile_is_rejected_before_run(tmp_path):
    changed = tmp_path/'changed.yaml'
    changed.write_text(entry.PROFILE.read_text()+'\n# altered\n')
    with pytest.raises(ValueError, match='audited'):
        entry.main(args(tmp_path)+['--profile', str(changed), '--inspect'])
    assert not (tmp_path/'new_run').exists()


def test_missing_local_model_does_not_download_or_create_run(tmp_path):
    with pytest.raises(FileNotFoundError, match='Local Qwen'):
        entry.main(args(tmp_path))
    assert not (tmp_path/'new_run').exists()


def test_wrong_body_is_rejected_without_outputs(tmp_path):
    (tmp_path/'model').mkdir()
    (tmp_path/'data').mkdir()
    for name in ('train.csv', 'test.csv', 'titles.csv', 'manifest.csv'):
        (tmp_path/'data'/name).write_text('synthetic\n')
    (tmp_path/'body.pt').write_text('wrong\n')
    with pytest.raises(RuntimeError, match='SHA mismatch'):
        entry.main(args(tmp_path))
    assert not (tmp_path/'new_run').exists()


def test_dispatch_is_evaluation_only_and_refuses_existing_run(tmp_path, monkeypatch):
    (tmp_path/'model').mkdir()
    (tmp_path/'data').mkdir()
    for name in ('train.csv', 'test.csv', 'titles.csv', 'manifest.csv'):
        (tmp_path/'data'/name).write_text('synthetic\n')
    (tmp_path/'body.pt').write_bytes(b'synthetic weight fixture\n')
    real_hash = entry.hashlib.sha256
    def checksum(data):
        return SimpleNamespace(hexdigest=lambda: entry.BODY_SHA256) if data == b'synthetic weight fixture\n' else real_hash(data)
    monkeypatch.setattr(entry.hashlib, 'sha256', checksum)
    from LightGenV2.tasks.t08_abo_image_text_retrieval import cache_preflight
    monkeypatch.setattr(cache_preflight, 'validate_teacher_cache', lambda raw, path: None)
    calls = []
    mock = SimpleNamespace(run=lambda options: calls.append(options) or {'status': 'synthetic dispatch only'})
    monkeypatch.setitem(sys.modules, 'LightGenV2.tasks.t08_abo_image_text_retrieval.reverse_runtime', mock)
    assert entry.main(args(tmp_path)) == 0
    assert calls[0].evaluate_only and calls[0].epochs is None
    assert not calls[0].force_teacher_cache
    assert calls[0].expected_resume_sha256 == entry.BODY_SHA256
    saved = yaml.safe_load((tmp_path/'new_run'/'input_config.yaml').read_text())
    assert saved['abo_image_text']['resume_checkpoint_sha256'] == entry.BODY_SHA256
    with pytest.raises(FileExistsError):
        entry.main(args(tmp_path))
    assert len(calls) == 1


def test_flattened_profile_changes_only_audited_fields():
    from maintenance.git_safety import materialize_t08_eval_profile as generator
    payload, identity = generator.build()
    assert payload.replace('\r\n', '\n') == entry.PROFILE.read_text().replace('\r\n', '\n')
    rows = []
    original = generator.read_chain(identity['original_profile'], set(), rows)
    expected = copy.deepcopy(original)
    for change in identity['reviewed_changes']:
        keys = change['key'].split('.')
        node = expected
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        assert node.get(keys[-1]) == change['original']
        node[keys[-1]] = change['value']
    assert expected == yaml.safe_load(payload)
    saved = json.loads((generator.ROOT/'maintenance/storage/T08_EVAL_PROFILE_IDENTITY_20261004.json').read_text())
    assert saved == identity and len(rows) == 21
