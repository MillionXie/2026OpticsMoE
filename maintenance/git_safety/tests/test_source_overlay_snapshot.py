import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapshot_source_overlay import snapshot


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args],
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')).decode().strip()


@pytest.fixture
def repo(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    git(tmp_path, 'config', 'user.name', 'test')
    git(tmp_path, 'config', 'user.email', 'test@example.invalid')
    path = 'LightGenV2/tasks/t09_multimodal_matching/vision.py'
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text('value = 1\n')
    git(tmp_path, 'add', path)
    git(tmp_path, 'commit', '-qm', 'base')
    file.write_text('value = 2\n')
    manifest = dict(expected_head=git(tmp_path, 'rev-parse', 'HEAD'),
                    archive_ref='refs/archive/test-overlay',
                    paths=[dict(path=path, sha256=hashlib.sha256(file.read_bytes()).hexdigest())])
    return tmp_path, file, manifest


def test_snapshot_preserves_working_head_index_and_dirty_source(repo):
    root, file, manifest = repo
    index = (root / '.git/index').read_bytes()
    before = git(root, 'status', '--porcelain')
    result = snapshot(root, manifest)
    assert git(root, 'rev-parse', 'HEAD') == manifest['expected_head']
    assert (root / '.git/index').read_bytes() == index
    assert git(root, 'status', '--porcelain') == before
    assert file.read_text() == 'value = 2\n'
    assert git(root, 'show', result['archive_commit']+':'+manifest['paths'][0]['path']) == 'value = 2'
    assert git(root, 'for-each-ref', '--format=%(refname)', 'refs/heads').count('\n') == 0


@pytest.mark.parametrize('change', ['sha', 'head', 'branch', 'duplicate', 'syntax'])
def test_snapshot_rejects_unreviewed_inputs(repo, change):
    root, file, manifest = repo
    if change == 'sha':
        manifest['paths'][0]['sha256'] = '0'*64
    elif change == 'head':
        manifest['expected_head'] = '0'*40
    elif change == 'branch':
        manifest['archive_ref'] = 'refs/heads/new-branch'
    elif change == 'duplicate':
        manifest['paths'].append(dict(manifest['paths'][0]))
    else:
        file.write_text('invalid syntax +\n')
        manifest['paths'][0]['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    with pytest.raises((RuntimeError, SyntaxError)):
        snapshot(root, manifest)
    assert not git(root, 'for-each-ref', '--format=%(refname)', 'refs/archive')


def test_existing_archive_never_overwritten(repo):
    root, _, manifest = repo
    first = snapshot(root, manifest)
    with pytest.raises(RuntimeError):
        snapshot(root, manifest)
    assert git(root, 'rev-parse', manifest['archive_ref']) == first['archive_commit']


def test_documentation_requires_explicit_opt_in_and_preserves_bytes(repo):
    root, _, manifest = repo
    doc = root/'LightGenPublic/README.md'
    doc.parent.mkdir()
    doc.write_bytes(b'# Historical method\n')
    git(root, 'add', 'LightGenPublic/README.md')
    git(root, 'commit', '-qm', 'tracked documentation')
    doc.write_bytes(b'# Historical method\r\nRetain baseline.\r\n')
    manifest['expected_head'] = git(root, 'rev-parse', 'HEAD')
    manifest['paths'] = [dict(path='LightGenPublic/README.md', sha256=hashlib.sha256(doc.read_bytes()).hexdigest())]
    with pytest.raises(RuntimeError):
        snapshot(root, manifest)
    result = snapshot(root, manifest, include_documentation=True)
    stored = subprocess.check_output(['git', '-C', str(root), 'show', result['archive_commit']+':LightGenPublic/README.md'])
    assert stored == doc.read_bytes()


@pytest.mark.parametrize('path', ['LightGenV2/results.pt', 'LightGenV2/unknown.py', '../escape.md'])
def test_documentation_mode_still_refuses_artifacts_untracked_and_escape(repo, path):
    root, _, manifest = repo
    manifest['paths'] = [dict(path=path, sha256='0'*64)]
    with pytest.raises(RuntimeError):
        snapshot(root, manifest, include_documentation=True)
