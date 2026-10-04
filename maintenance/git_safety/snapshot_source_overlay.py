"""Freeze SHA-reviewed working source in Git, without touching HEAD/user index.

The result is an archive ref, NOT a development branch or runtime claim. No
source files, experiment assets, checkout or working-tree status are changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile


def snapshot(root, manifest, *, include_documentation=False):
    root = Path(root).resolve()
    def git(*args, data=None, env=None):
        safe_env = dict(os.environ if env is None else env, GIT_OPTIONAL_LOCKS='0')
        return subprocess.run(['git', '-C', str(root), *args], input=data,
                              env=safe_env, check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE).stdout
    head = git('rev-parse', 'HEAD').decode().strip()
    if head != manifest['expected_head']:
        raise RuntimeError('HEAD changed')
    ref = manifest['archive_ref']
    if not ref.startswith('refs/archive/'):
        raise RuntimeError('Only archive refs allowed')
    git('check-ref-format', ref)
    exists = subprocess.run(['git', '-C', str(root), 'show-ref', '--verify', '--quiet', ref])
    if exists.returncode != 1:
        raise RuntimeError('Archive ref exists or lookup failed')
    index = Path(git('rev-parse', '--path-format=absolute', '--git-path', 'index').decode().strip())
    original_index = index.read_bytes() if index.exists() else None
    status = git('status', '--porcelain', '-z', '-uno')
    blobs, seen = [], set()
    for row in manifest['paths']:
        path = row['path']; pure = PurePosixPath(path)
        prefix_allowed = path.startswith('LightGenV2/') or (include_documentation and path.startswith('LightGenPublic/'))
        suffix_allowed = path.endswith('.py') or (include_documentation and path.endswith('.md'))
        if (not prefix_allowed or str(pure) != path or '..' in pure.parts
                or '\\' in path or ':' in path or path in seen or not suffix_allowed):
            raise RuntimeError('Only explicit tracked source/documentation allowed')
        seen.add(path)
        entry = git('ls-tree', '-z', head, '--', path)
        if not entry:
            raise RuntimeError('Not tracked in original HEAD')
        mode, kind, old = entry.split(b'\t', 1)[0].split()
        if mode not in (b'100644', b'100755') or kind != b'blob':
            raise RuntimeError('Special Git entry')
        source = root / path
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise RuntimeError('Source outside root')
        content = source.read_bytes()
        if len(content) > 1_000_000 or hashlib.sha256(content).hexdigest() != row['sha256']:
            raise RuntimeError('Unreviewed source SHA/size')
        decoded = content.decode('utf-8')
        if path.endswith('.py'):
            compile(decoded, path, 'exec')
        if content == git('cat-file', 'blob', old.decode()):
            raise RuntimeError('Not an overlay')
        oid = git('hash-object', '-w', '--stdin', data=content).decode().strip()
        blobs.append((path, mode.decode(), oid, content))
    if not blobs:
        raise RuntimeError('Empty overlay')
    with tempfile.TemporaryDirectory(prefix='lightgen_overlay_index_') as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp)/'index'))
        git('read-tree', head, env=env)
        for path, mode, oid, _ in blobs:
            git('update-index', '--cacheinfo', mode, oid, path, env=env)
        changed = set(filter(None, git('diff', '--cached', '--name-only', '-z', head, env=env).decode().split('\0')))
        if changed != seen:
            raise RuntimeError('Unexpected paths')
        tree = git('write-tree', env=env).decode().strip()
        commit = git('commit-tree', tree, '-p', head, data=b'Archive reviewed working source overlay; not a runtime publication\n').decode().strip()
    if (head != git('rev-parse', 'HEAD').decode().strip()
            or status != git('status', '--porcelain', '-z', '-uno')
            or original_index != (index.read_bytes() if index.exists() else None)
            or any((root/path).read_bytes() != content for path, _, _, content in blobs)):
        raise RuntimeError('Concurrent source/index change; archive not published')
    git('update-ref', ref, commit, '0'*40)
    return dict(source_head=head, archive_commit=commit, archive_ref=ref,
                files=manifest['paths'], working_head_unchanged=True,
                user_index_unchanged=True, source_files_changed=False,
                development_branch_created=False, runtime_verified=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--include-documentation', action='store_true',
                   help='Also permit reviewed tracked .md and LightGenPublic source; never data or untracked files')
    args = p.parse_args()
    print(json.dumps(snapshot(args.root, json.loads(args.manifest.read_text()),
                              include_documentation=args.include_documentation), indent=2))
