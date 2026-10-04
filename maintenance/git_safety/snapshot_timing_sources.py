"""Archive two explicitly reviewed, untracked timing scripts without a checkout.

Only Git objects and a new refs/archive reference are written. Never executes
the profilers, modifies user index/source, creates a branch, or publishes main.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

PATHS = (
    'LightGenV2/scripts/profile_narrow_optical_electronics_a100.py',
    'LightGenV2/scripts/profile_latest_optical_electronics_a100.py',
)


def snapshot(root, expected_head, archive_ref, hashes):
    root = Path(root).resolve()
    if set(hashes) != set(PATHS):
        raise RuntimeError('Only the two reviewed timing sources are permitted')
    if not archive_ref.startswith('refs/archive/'):
        raise RuntimeError('Only a recovery reference is permitted')

    def git(*args, data=None, env=None):
        return subprocess.check_output(['git', '-C', str(root), *args], input=data,
                                       env=dict(os.environ if env is None else env,
                                                GIT_OPTIONAL_LOCKS='0'))

    git('check-ref-format', archive_ref)
    found = subprocess.run(['git', '-C', str(root), 'show-ref', '--verify', '--quiet', archive_ref])
    if found.returncode != 1:
        raise RuntimeError('Existing recovery reference protected')
    if git('rev-parse', 'HEAD').decode().strip() != expected_head:
        raise RuntimeError('HEAD changed')
    index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not index.is_absolute():
        index = root / index
    original_index = index.read_bytes() if index.exists() else None
    original_status = git('status', '--porcelain', '-z', '-uno')
    contents, objects = {}, {}
    for path in PATHS:
        if git('ls-tree', expected_head, '--', path) or git('ls-files', '--', path):
            raise RuntimeError('Tracked paths require separate overlay review')
        source = root / path
        if source.is_symlink() or root not in source.resolve().parents:
            raise RuntimeError('Source outside root')
        content = source.read_bytes()
        if len(content) > 1_000_000 or hashlib.sha256(content).hexdigest() != hashes[path]:
            raise RuntimeError('Reviewed source identity changed')
        compile(content.decode('utf-8'), path, 'exec')
        contents[path] = content
        objects[path] = git('hash-object', '-w', '--stdin', data=content).decode().strip()
    with tempfile.TemporaryDirectory(prefix='lightgen_timing_recovery_index_') as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch) / 'index'))
        git('read-tree', expected_head, env=env)
        for path, oid in objects.items():
            git('update-index', '--add', '--cacheinfo', '100644', oid, path, env=env)
        changed = set(git('diff', '--cached', '--name-only', '-z', expected_head,
                          env=env).decode().strip('\0').split('\0'))
        if changed != set(PATHS):
            raise RuntimeError('Unexpected recovery tree')
        tree = git('write-tree', env=env).decode().strip()
        commit = git('commit-tree', tree, '-p', expected_head, env=env,
                     data=b'Preserve reviewed untracked timing sources; no runtime claim\n').decode().strip()
    if (git('rev-parse', 'HEAD').decode().strip() != expected_head
            or original_status != git('status', '--porcelain', '-z', '-uno')
            or original_index != (index.read_bytes() if index.exists() else None)
            or any((root / path).read_bytes() != data for path, data in contents.items())):
        raise RuntimeError('Concurrent state change; recovery reference not created')
    git('update-ref', archive_ref, commit, '0' * 40)
    return {'archive_commit': commit, 'archive_ref': archive_ref,
            'source_head': expected_head, 'files': [
                {'path': p, 'sha256': hashes[p], 'bytes': len(contents[p])} for p in PATHS],
            'source_and_index_unchanged': True, 'benchmark_executed': False,
            'development_branch_created': False, 'main_published': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--head', required=True)
    p.add_argument('--archive-ref', required=True)
    p.add_argument('--narrow-sha256', required=True)
    p.add_argument('--latest-sha256', required=True)
    a = p.parse_args()
    print(json.dumps(snapshot(a.root, a.head, a.archive_ref,
                             dict(zip(PATHS, (a.narrow_sha256, a.latest_sha256)))), indent=2))
