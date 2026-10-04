import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapshot_timing_sources import PATHS, snapshot


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode().strip()


@pytest.fixture
def repo(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    git(tmp_path, 'config', 'user.name', 'test')
    git(tmp_path, 'config', 'user.email', 'test@example.invalid')
    (tmp_path / 'README.md').write_text('protected\n')
    git(tmp_path, 'add', 'README.md')
    git(tmp_path, 'commit', '-qm', 'initial')
    hashes = {}
    for path in PATHS:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'value = 1\r\n')
        hashes[path] = hashlib.sha256(file.read_bytes()).hexdigest()
    return tmp_path, git(tmp_path, 'rev-parse', 'HEAD'), hashes


def test_exact_untracked_recovery_preserves_checkout_and_index(repo):
    root, head, hashes = repo
    before = (root / '.git/index').read_bytes()
    result = snapshot(root, head, 'refs/archive/timing-test', hashes)
    assert git(root, 'rev-parse', 'HEAD') == head
    assert (root / '.git/index').read_bytes() == before
    for path in PATHS:
        saved = subprocess.check_output(['git', '-C', str(root), 'show', result['archive_commit'] + ':' + path])
        assert saved == (root / path).read_bytes()
    assert git(root, 'branch', '--format=%(refname)').count('\n') == 0
    with pytest.raises(RuntimeError):
        snapshot(root, head, 'refs/archive/timing-test', hashes)


@pytest.mark.parametrize('change', ['sha', 'head', 'branch', 'tracked', 'extra'])
def test_rejects_unreviewed_or_unsafe_identity(repo, change):
    root, head, hashes = repo
    ref = 'refs/archive/timing-test'
    if change == 'sha': hashes[PATHS[0]] = '0' * 64
    if change == 'head': head = '0' * 40
    if change == 'branch': ref = 'refs/heads/unwanted'
    if change == 'tracked': git(root, 'add', PATHS[0])
    if change == 'extra': hashes['private_connection.py'] = '0' * 64
    with pytest.raises(RuntimeError):
        snapshot(root, head, ref, hashes)
    assert not git(root, 'for-each-ref', '--format=%(refname)', 'refs/archive')
