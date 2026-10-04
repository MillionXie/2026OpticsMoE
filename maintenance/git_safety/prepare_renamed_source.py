"""Prepare one exact, renamed source blob on main; never modify a checkout.

Used when two actual runtimes occupied the same historical path but implement
different task directions. No text rewriting, source overwrite or ref updates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from prepare_scoped_main import git
from review_git import forbidden_artifact


def prepare(root, manifest, message, base_commit=None):
    root = Path(root).resolve()
    main = git(root, 'rev-parse', 'main').decode().strip()
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    base = git(root, 'rev-parse', '--verify', (base_commit or main)+'^{commit}').decode().strip()
    git(root, 'merge-base', '--is-ancestor', main, base)
    if 'branch refs/heads/main\n' in git(root, 'worktree', 'list', '--porcelain').decode():
        raise RuntimeError('main is checked out')
    index = Path(git(root, 'rev-parse', '--path-format=absolute', '--git-path', 'index').decode().strip())
    before = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    source = git(root, 'rev-parse', '--verify', manifest['source_commit']+'^{commit}').decode().strip()
    rows, used = [], set()
    for row in manifest['paths']:
        original, target = row['source_path'], row['target_path']
        for path in (original, target):
            pure = PurePosixPath(path)
            if pure.is_absolute() or '..' in pure.parts or str(pure) != path or '\\' in path or ':' in path:
                raise RuntimeError('Unsafe source path')
        if target in used or not target.startswith(manifest['task_prefix']):
            raise RuntimeError('Duplicate or out-of-task target')
        used.add(target)
        if not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise RuntimeError('Missing reviewed source SHA')
        entry = git(root, 'ls-tree', '-z', source, '--', original)
        if not entry:
            raise RuntimeError('Missing source')
        mode, kind, oid = entry.split(b'\t', 1)[0].split()
        if mode not in (b'100644', b'100755') or kind != b'blob':
            raise RuntimeError('Special source entry')
        data = git(root, 'cat-file', 'blob', oid.decode())
        if (b'\0' in data or forbidden_artifact(target, len(data))
                or PurePosixPath(target).name == 'server_sync.py'):
            raise RuntimeError('Unsafe source content')
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise RuntimeError('Source SHA/size mismatch')
        if git(root, 'ls-tree', base, '--', target):
            raise RuntimeError('Target already exists; never overwrite')
        if target.endswith('.py'):
            compile(data.decode('utf-8'), target, 'exec')
        else:
            data.decode('utf-8')
        rows.append((target, mode.decode(), oid.decode()))
    if not rows:
        raise RuntimeError('No source additions')
    with tempfile.TemporaryDirectory(prefix='lightgen_renamed_index_') as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch)/'index'))
        git(root, 'read-tree', base, env=env)
        for path, mode, oid in rows:
            git(root, 'update-index', '--add', '--cacheinfo', mode, oid, path, env=env)
        tree = git(root, 'write-tree', env=env).decode().strip()
        candidate = git(root, 'commit-tree', tree, '-p', base, env=env, data=(message+'\n').encode()).decode().strip()
    after = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    if before != after or head != git(root, 'rev-parse', 'HEAD').decode().strip() or main != git(root, 'rev-parse', 'main').decode().strip():
        raise RuntimeError('Concurrent index/HEAD/main change')
    return dict(base_main=main, candidate_parent=base, candidate_commit=candidate, source_commit=source,
                renamed_paths=[r[0] for r in rows], working_files_unchanged=True,
                user_index_unchanged=True, refs_updated=False, pushed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--message', required=True)
    parser.add_argument('--base-commit')
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[2], json.loads(args.manifest.read_text()), args.message, args.base_commit), indent=2))
