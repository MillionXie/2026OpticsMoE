"""Prepare explicitly reviewed, old/new-SHA-pinned task replacements on main.

No checkout, branch, ref update, push, source copy or user-index mutation.
Missing paths must use the separate missing-only importer. Shared paths forbidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from prepare_scoped_main import git
from review_git import forbidden_artifact


def prepare(root: Path, manifest: dict, message: str, base_commit: str | None = None) -> dict:
    root = root.resolve()
    main = git(root, 'rev-parse', 'refs/heads/main').decode().strip()
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    if 'branch refs/heads/main\n' in git(root, 'worktree', 'list', '--porcelain').decode():
        raise RuntimeError('main checked out; stop')
    if main != manifest['expected_main']:
        raise RuntimeError('main changed; review old identities again')
    base = git(root, 'rev-parse', '--verify', (base_commit or main)+'^{commit}').decode().strip()
    git(root, 'merge-base', '--is-ancestor', main, base)
    source = git(root, 'rev-parse', '--verify', manifest['source_commit']+'^{commit}').decode().strip()
    prefix = manifest['task_prefix']
    parts = PurePosixPath(prefix).parts
    if len(parts) != 3 or parts[:2] != ('LightGenV2', 'tasks') or not prefix.endswith('/'):
        raise RuntimeError('Only one task directory is allowed')
    index = Path(git(root, 'rev-parse', '--path-format=absolute', '--git-path', 'index').decode().strip())
    before = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    changes, seen = [], set()
    for row in manifest['paths']:
        path = row['path']
        pure = PurePosixPath(path)
        if (not path.startswith(prefix) or '..' in pure.parts or str(pure) != path
                or ':' in path or '\\' in path or path in seen
                or any(ord(c)<32 for c in path) or pure.name == 'server_sync.py'):
            raise RuntimeError('Unsafe/duplicate path: '+path)
        seen.add(path)
        source_path = row.get('source_path', path)
        source_pure = PurePosixPath(source_path)
        if source_path != path and (not source_path.startswith(prefix)
                or not source_path.endswith('.md') or not path.endswith('.md')
                or '..' in source_pure.parts or str(source_pure) != source_path
                or ':' in source_path or '\\' in source_path
                or any(ord(c)<32 for c in source_path)):
            raise RuntimeError('Unsafe documentation source mapping')
        if not row.get('review_reason', '').strip():
            raise RuntimeError('Missing review reason')
        for key in ('source_sha256', 'expected_target_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', row[key]):
                raise RuntimeError('Missing SHA')
        entries = [git(root, 'ls-tree', '-z', base, '--', path),
                   git(root, 'ls-tree', '-z', source, '--', source_path)]
        if not all(entries):
            raise RuntimeError('Replacements require existing source AND target')
        blobs = []
        for entry in entries:
            mode, kind, oid = entry.split(b'\t',1)[0].split()
            if mode not in (b'100644', b'100755') or kind != b'blob':
                raise RuntimeError('Special file')
            blobs.append(git(root, 'cat-file', 'blob', oid.decode()))
        old, new = blobs
        if hashlib.sha256(old).hexdigest() != row['expected_target_sha256']:
            raise RuntimeError('Old target SHA mismatch: '+path)
        if hashlib.sha256(new).hexdigest() != row['source_sha256']:
            raise RuntimeError('Source SHA mismatch: '+path)
        if old == new:
            raise RuntimeError('Not a replacement: '+path)
        if forbidden_artifact(path,len(new)) or b'\0' in new:
            raise RuntimeError('Forbidden artifact')
        new.decode('utf-8')
        if path.endswith('.py'):
            compile(new.decode('utf-8'), path, 'exec')
        mode, _, oid = entries[1].split(b'\t',1)[0].split()
        changes.append((path, mode.decode(), oid.decode()))
    if not changes:
        raise RuntimeError('Empty replacement list')
    with tempfile.TemporaryDirectory(prefix='lightgen_reviewed_index_') as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch)/'index'))
        git(root, 'read-tree', base, env=env)
        for path, mode, oid in changes:
            git(root, 'update-index', '--cacheinfo', mode, oid, path, env=env)
        changed = set(filter(None,git(root,'diff','--cached','--name-only','-z',base,env=env).decode().split('\0')))
        if changed != seen or git(root,'ls-files','-u',env=env):
            raise RuntimeError('Unexpected candidate changes')
        tree = git(root,'write-tree',env=env).decode().strip()
        candidate = git(root,'commit-tree',tree,'-p',base,env=env,data=(message+'\n').encode()).decode().strip()
    after = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    if before != after or head != git(root,'rev-parse','HEAD').decode().strip() or main != git(root,'rev-parse','main').decode().strip():
        raise RuntimeError('Concurrent Git change; do not publish')
    return {'base_main':main, 'candidate_parent':base, 'candidate_commit':candidate, 'source_commit':source,
            'reviewed_replacements':sorted(seen), 'working_head_unchanged':True,
            'user_index_unchanged':True, 'refs_updated':False, 'pushed':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--message', required=True)
    parser.add_argument('--base-commit')
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[2],json.loads(args.manifest.read_text(encoding='utf-8')),args.message,args.base_commit),indent=2))
