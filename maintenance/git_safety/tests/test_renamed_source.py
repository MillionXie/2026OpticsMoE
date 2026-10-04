import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_renamed_source import prepare
from test_main_preparation import git, synthetic_repo


def fixture(tmp_path):
    main = synthetic_repo(tmp_path)
    data = b'original = 1\n'
    (tmp_path/'old.py').write_bytes(data)
    git(tmp_path, 'add', 'old.py')
    git(tmp_path, 'commit', '-m', 'original direction')
    return main, dict(source_commit=git(tmp_path, 'rev-parse', 'HEAD').decode().strip(),
                     task_prefix='task/', paths=[dict(source_path='old.py', target_path='task/reverse.py',
                                                      sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))])


def test_exact_blob_preserves_index_main_and_user_changes(tmp_path):
    main, manifest = fixture(tmp_path)
    (tmp_path/'old.py').write_text('user = 2\n')
    (tmp_path/'user.py').write_text('staged = 3\n')
    git(tmp_path, 'add', 'user.py')
    before = git(tmp_path, 'status', '--porcelain')
    result = prepare(tmp_path, manifest, 'direction-specific entry')
    assert git(tmp_path, 'show', result['candidate_commit']+':task/reverse.py') == b'original = 1\n'
    assert git(tmp_path, 'status', '--porcelain') == before
    assert git(tmp_path, 'rev-parse', 'main').decode().strip() == main


@pytest.mark.parametrize('bad', ['sha', 'existing', 'outside', 'duplicate'])
def test_refuses_unsafe_import(tmp_path, bad):
    _, manifest = fixture(tmp_path)
    if bad == 'sha':
        manifest['paths'][0]['sha256'] = '0'*64
    elif bad == 'existing':
        manifest['task_prefix'] = ''
        manifest['paths'][0]['target_path'] = 'README.md'
    elif bad == 'outside':
        manifest['paths'][0]['target_path'] = '../escape.py'
    else:
        manifest['paths'] *= 2
    with pytest.raises(RuntimeError):
        prepare(tmp_path, manifest, 'refused')


def test_composes_on_descendant_without_switching_main(tmp_path):
    main, manifest = fixture(tmp_path)
    tree = git(tmp_path, 'rev-parse', main+'^{tree}').decode().strip()
    parent = git(tmp_path, 'commit-tree', tree, '-p', main, '-m', 'metadata').decode().strip()
    result = prepare(tmp_path, manifest, 'composed entry', parent)
    assert result['candidate_parent'] == parent
    assert git(tmp_path, 'rev-parse', result['candidate_commit']+'^').decode().strip() == parent
    assert git(tmp_path, 'rev-parse', 'main').decode().strip() == main
