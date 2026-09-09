import json
from pathlib import Path

import pytest

from LightGenV2.tasks.t04_semantic_interaction import standalone as release
from LightGenV2.tasks.t04_semantic_interaction.build_lab_package import REPO, source_closure


def test_import_closure_is_local_and_has_ours_not_baseline():
    paths = source_closure()
    relatives = {p.relative_to(REPO).as_posix() for p in paths}
    for suffix in ['embedding_model.py', 'router_repair.py', 'shared_readout.py', 'training.py']:
        assert 'LightGenV2/tasks/t04_semantic_interaction/' + suffix in relatives
    assert 'LightGenV2/tasks/t04_semantic_interaction/qwen_shared.py' not in relatives
    assert len(paths) < 100


def test_integrity_check_rejects_modified_and_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    content = tmp_path / 'weights.pt'
    content.write_bytes(b'test artifact')
    (tmp_path / 'MANIFEST.json').write_text(json.dumps({
        'package_git_commit': 'test',
        'files': [{'path': 'weights.pt', 'sha256': release.digest(content)}],
    }))
    assert release.verify()['package_git_commit'] == 'test'
    content.write_bytes(b'modified')
    with pytest.raises(RuntimeError, match='Missing/modified'):
        release.verify()
    content.unlink()
    with pytest.raises(RuntimeError, match='Missing/modified'):
        release.verify()


def test_settings_relocate_all_runtime_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    (tmp_path / 'settings.json').write_text(json.dumps({'seed': 73, 'shared_readout_variant': 'slim'}))
    cfg = release.settings(tmp_path / 'outputs')
    for key in ['config_path', 'data_dir', 'asset_dir', 'output_dir', 'qwen_checkpoint',
                'prompt_cache_path', 'optical_base_config', 'legacy_warmstart_checkpoint']:
        assert Path(getattr(cfg, key)).is_relative_to(tmp_path)
    assert cfg.shared_readout_variant == 'standard'
    assert cfg.num_workers == 0


def test_outputs_never_overwrite_existing_results(tmp_path):
    out = release.output_directory(str(tmp_path / 'new'))
    (out / 'result.json').write_text('{}')
    with pytest.raises(FileExistsError):
        release.output_directory(str(out))
