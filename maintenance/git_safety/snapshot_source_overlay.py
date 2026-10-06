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


USER_D2NN_SOURCES = frozenset('TransferFromElectricity/d2nn_pack/' + name for name in (
    'config.yaml', 'data.py', 'extract_clip_features.py', 'losses.py', 'mask_generator.py',
    'model_adapt.py', 'train_d2nn_adapt_grid.py', 'train_d2nn_mnist256.py'))
T12_DELIVERY_TOOLS = frozenset('handoffs/t12_small_baseline_share_20260928/package/' + name
                              for name in ('export_pairs.py', 'infer_ours.py')) | frozenset({
    'handoffs/t12_small_baseline_handoff_20260928/stage/materialize_pairs.py'})


def snapshot(root, manifest, *, include_documentation=False, include_untracked_source=False,
             include_user_d2nn_source=False, include_reviewed_launchers=False,
             include_t12_delivery_tools=False):
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
    index_value = git('rev-parse', '--git-path', 'index').decode().strip()
    if not index_value or '\n' in index_value or '\r' in index_value:
        raise RuntimeError('Ambiguous Git index path')
    index = Path(index_value)
    if not index.is_absolute():
        index = root / index
    original_index = index.read_bytes() if index.exists() else None
    status = git('status', '--porcelain', '-z', '-uno')
    blobs, seen = [], set()
    for row in manifest['paths']:
        path = row['path']; pure = PurePosixPath(path)
        prefix_allowed = (path.startswith('LightGenV2/')
                          or (include_documentation and path.startswith('LightGenPublic/'))
                          or (include_user_d2nn_source and path in USER_D2NN_SOURCES)
                          or (include_t12_delivery_tools and path in T12_DELIVERY_TOOLS))
        suffix_allowed = (path.endswith('.py') or (include_documentation and path.endswith('.md'))
                          or (include_untracked_source and path.endswith(('.yaml', '.yml')))
                          or (include_reviewed_launchers and path.startswith('LightGenV2/')
                              and path.endswith('.cmd')))
        if (not prefix_allowed or str(pure) != path or '..' in pure.parts
                or '\\' in path or ':' in path or path in seen or not suffix_allowed):
            raise RuntimeError('Only explicit reviewed source/documentation allowed')
        seen.add(path)
        entry = git('ls-tree', '-z', head, '--', path)
        if not entry:
            if not include_untracked_source or row.get('entry_kind') != 'untracked':
                raise RuntimeError('Not tracked in original HEAD')
            # Never silently include ignored assets, staged additions or arbitrary directories.
            observed = git('ls-files', '--others', '--exclude-standard', '-z', '--', path)
            if observed != path.encode('utf-8') + b'\0':
                raise RuntimeError('Not an explicit untracked non-ignored file')
            mode, kind, old = b'100644', b'blob', None
        else:
            if row.get('entry_kind', 'tracked') != 'tracked':
                raise RuntimeError('Tracked/untracked identity changed')
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
        if old is not None and content == git('cat-file', 'blob', old.decode()):
            raise RuntimeError('Not an overlay')
        oid = git('hash-object', '-w', '--stdin', data=content).decode().strip()
        blobs.append((path, mode.decode(), oid, content))
    if not blobs:
        raise RuntimeError('Empty overlay')
    with tempfile.TemporaryDirectory(prefix='lightgen_overlay_index_') as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp)/'index'))
        git('read-tree', head, env=env)
        for path, mode, oid, _ in blobs:
            git('update-index', '--add', '--cacheinfo', mode, oid, path, env=env)
        changed = set(filter(None, git('diff', '--cached', '--name-only', '-z', head, env=env).decode().split('\0')))
        if changed != seen:
            raise RuntimeError('Unexpected paths')
        tree = git('write-tree', env=env).decode().strip()
        commit = git('commit-tree', tree, '-p', head, data=b'Archive reviewed working source overlay; not a runtime publication\n').decode().strip()
    if (head != git('rev-parse', 'HEAD').decode().strip()
            or status != git('status', '--porcelain', '-z', '-uno')
            or original_index != (index.read_bytes() if index.exists() else None)
            or any((root/path).read_bytes() != content for path, _, _, content in blobs)
            or any(git('ls-files', '--others', '--exclude-standard', '-z', '--', row['path'])
                   != row['path'].encode('utf-8') + b'\0'
                   for row in manifest['paths'] if row.get('entry_kind') == 'untracked')):
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
    p.add_argument('--include-untracked-source', action='store_true',
                   help='Permit SHA-reviewed non-ignored .py/.yaml/.yml additions explicitly marked entry_kind=untracked; archive only')
    p.add_argument('--include-reviewed-launchers', action='store_true',
                   help='Archive explicitly SHA-reviewed LightGenV2 .cmd source only; never execute launchers')
    p.add_argument('--include-t12-delivery-tools', action='store_true',
                   help='Permit only the three named reviewed T12 delivery helper Python files')
    args = p.parse_args()
    print(json.dumps(snapshot(args.root, json.loads(args.manifest.read_text()),
                              include_documentation=args.include_documentation,
                              include_untracked_source=args.include_untracked_source,
                              include_reviewed_launchers=args.include_reviewed_launchers,
                              include_t12_delivery_tools=args.include_t12_delivery_tools), indent=2))
