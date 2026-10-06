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


def test_named_user_source_requires_opt_in_and_preserves_bytes(repo):
    root, _, manifest = repo
    path = 'TransferFromElectricity/d2nn_pack/model_adapt.py'
    file = root/path
    file.parent.mkdir(parents=True)
    file.write_bytes(b'user_value = 42\n')
    manifest['paths'] = [dict(path=path, sha256=hashlib.sha256(file.read_bytes()).hexdigest(), entry_kind='untracked')]
    with pytest.raises(RuntimeError):
        snapshot(root, manifest, include_untracked_source=True)
    result = snapshot(root, manifest, include_untracked_source=True, include_user_d2nn_source=True)
    assert subprocess.check_output(['git','-C',str(root),'show',result['archive_commit']+':'+path]) == file.read_bytes()
    assert result['working_head_unchanged'] and result['user_index_unchanged']


def test_launcher_opt_in_archives_without_running_or_changing_checkout(repo):
    root, _, manifest = repo
    path = 'LightGenV2/tasks/t07_abo_image_retrieval/run_old.cmd'
    file = root/path
    file.parent.mkdir(parents=True)
    file.write_bytes(b'@echo off\r\necho should_not_run > marker.txt\r\n')
    manifest['paths'] = [dict(path=path, sha256=hashlib.sha256(file.read_bytes()).hexdigest(), entry_kind='untracked')]
    before = git(root, 'status', '--porcelain')
    with pytest.raises(RuntimeError):
        snapshot(root, manifest, include_untracked_source=True)
    result = snapshot(root, manifest, include_untracked_source=True, include_reviewed_launchers=True)
    assert git(root, 'status', '--porcelain') == before
    assert not (root/'marker.txt').exists()
    assert subprocess.check_output(['git','-C',str(root),'show',result['archive_commit']+':'+path]) == file.read_bytes()


def test_launcher_mode_does_not_allow_private_handoff_paths(repo):
    root, _, manifest = repo
    manifest['paths'] = [dict(path='handoffs/connect.cmd', sha256='0'*64, entry_kind='untracked')]
    with pytest.raises(RuntimeError):
        snapshot(root, manifest, include_untracked_source=True, include_reviewed_launchers=True)


def test_named_t12_delivery_tool_only_opt_in(repo):
    root, _, manifest = repo
    path = 'handoffs/t12_small_baseline_share_20260928/package/export_pairs.py'
    file = root/path
    file.parent.mkdir(parents=True)
    file.write_bytes(b'print("not executed")\n')
    manifest['paths'] = [dict(path=path, sha256=hashlib.sha256(file.read_bytes()).hexdigest(), entry_kind='untracked')]
    with pytest.raises(RuntimeError):
        snapshot(root, manifest, include_untracked_source=True)
    result = snapshot(root, manifest, include_untracked_source=True, include_t12_delivery_tools=True)
    assert result['working_head_unchanged'] and result['user_index_unchanged']
    assert subprocess.check_output(['git','-C',str(root),'show',result['archive_commit']+':'+path]) == file.read_bytes()


def test_t12_opt_in_rejects_arbitrary_connection_script(repo):
    root, _, manifest = repo
    manifest['paths'] = [dict(path='handoffs/t12_small_baseline_share_20260928/package/connect.py', sha256='0'*64, entry_kind='untracked')]
    with pytest.raises(RuntimeError):
        snapshot(root, manifest, include_untracked_source=True, include_t12_delivery_tools=True)


def test_user_source_mode_does_not_enable_other_directories(repo):
    root, _, manifest = repo
    manifest['paths'] = [dict(path='TransferFromElectricity/unknown.py', sha256='0'*64, entry_kind='untracked')]
    with pytest.raises(RuntimeError):
        snapshot(root, manifest, include_untracked_source=True, include_user_d2nn_source=True)


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


@pytest.mark.parametrize('suffix,content', [('.py', b'value = 3\n'), ('.yaml', b'alpha: 0.3\n')])
def test_explicit_untracked_recovery_preserves_original_checkout(repo, suffix, content):
    root, _, manifest = repo
    path = 'LightGenV2/tasks/t09_multimodal_matching/recovery' + suffix
    file = root/path
    file.write_bytes(content)
    manifest['paths'] = [dict(path=path, sha256=hashlib.sha256(content).hexdigest(), entry_kind='untracked')]
    before = git(root, 'status', '--porcelain')
    index = (root/'.git/index').read_bytes()
    with pytest.raises(RuntimeError):
        snapshot(root, manifest)
    result = snapshot(root, manifest, include_untracked_source=True)
    assert git(root, 'status', '--porcelain') == before
    assert (root/'.git/index').read_bytes() == index
    assert file.read_bytes() == content
    assert git(root, 'rev-parse', 'HEAD') == manifest['expected_head']
    stored = subprocess.check_output(['git', '-C', str(root), 'show', result['archive_commit']+':'+path])
    assert stored == content


@pytest.mark.parametrize('case', ['missing_kind', 'ignored', 'staged', 'artifact', 'tracked_kind'])
def test_untracked_opt_in_is_not_blanket_permission(repo, case):
    root, original, manifest = repo
    path = 'LightGenV2/tasks/t09_multimodal_matching/recovery.py'
    file = root/path
    file.write_bytes(b'value = 3\n')
    kind = 'untracked'
    if case == 'missing_kind':
        kind = 'tracked'
    elif case == 'ignored':
        (root/'.git/info/exclude').write_text('recovery.py\n')
    elif case == 'staged':
        git(root, 'add', path)
    elif case == 'artifact':
        path = 'LightGenV2/results.pt'
        file = root/path
        file.write_bytes(b'not a source file')
    elif case == 'tracked_kind':
        file = original
        path = manifest['paths'][0]['path']
    manifest['paths'] = [dict(path=path, sha256=hashlib.sha256(file.read_bytes()).hexdigest(), entry_kind=kind)]
    with pytest.raises(RuntimeError):
        snapshot(root, manifest, include_untracked_source=True)
    assert not git(root, 'for-each-ref', '--format=%(refname)', 'refs/archive')
