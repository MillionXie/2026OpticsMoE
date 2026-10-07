"""Adopt main in an existing non-Git root without replacing existing files.

The source is an existing local Git object store, not copied source files.
Default is a read-only collision audit. Existing runtime subdirectories are
neither migrated nor removed. A matching checkout is not hardware readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def call(git, directory, *args, data=None):
    return subprocess.check_output([git, '-C', str(directory), *args], input=data)


def audit(root: Path, store: Path, git: str, pin: str) -> dict:
    root, store = root.resolve(), store.resolve()
    if not root.is_dir() or (root / '.git').exists():
        raise ValueError('Target must be an existing root without .git')
    if not re.fullmatch(r'[0-9a-f]{40}', pin):
        raise ValueError('Explicit full commit identity required')
    if call(git, store, 'rev-parse', '--verify', pin + '^{commit}').decode().strip() != pin:
        raise ValueError('Source commit identity differs')
    missing, existing, conflicts = [], [], []
    for entry in call(git, store, 'ls-tree', '-r', '-z', pin).split(b'\0'):
        if not entry:
            continue
        metadata, raw_path = entry.split(b'\t', 1)
        mode, kind, oid = metadata.split()
        relative = raw_path.decode('utf8')
        target = root / relative
        if (not target.resolve().is_relative_to(root) or '\\' in relative
                or mode not in (b'100644', b'100755') or kind != b'blob'):
            conflicts.append({'path': relative, 'reason': 'unsafe_path_or_mode'})
            continue
        if any(p.is_symlink() for p in [target, *target.parents] if p != root):
            conflicts.append({'path': relative, 'reason': 'symlink'})
        elif target.exists():
            if not target.is_file():
                conflicts.append({'path': relative, 'reason': 'not_regular_file'})
                continue
            raw = target.read_bytes()
            source = call(git, store, 'cat-file', 'blob', oid.decode())
            if raw.replace(b'\r\n', b'\n') != source.replace(b'\r\n', b'\n'):
                conflicts.append({'path': relative, 'reason': 'different_existing_bytes'})
            else:
                existing.append({'path': relative, 'sha256': hashlib.sha256(raw).hexdigest()})
        elif any(p.exists() and not p.is_dir() for p in target.parents if p != root):
            conflicts.append({'path': relative, 'reason': 'parent_not_directory'})
        else:
            missing.append(relative)
    return {'pin': pin, 'missing': missing, 'existing': existing, 'conflicts': conflicts}


def adopt(root: Path, store: Path, git: str, pin: str) -> dict:
    before = audit(root, store, git, pin)
    if before['conflicts']:
        raise ValueError('Unresolved collisions; target left unchanged')
    # Older lab/server Git lacks --initial-branch. No commit or extra branch
    # exists yet; select the sole intended branch with an unborn symbolic HEAD.
    call(git, root, 'init')
    call(git, root, 'symbolic-ref', 'HEAD', 'refs/heads/main')
    # Source manifests bind Git bytes, not the platform's preferred newlines.
    call(git, root, 'config', 'core.autocrlf', 'false')
    call(git, root, 'fetch', '--no-tags', str(store), pin)
    call(git, root, 'update-ref', 'refs/heads/main', pin)
    call(git, root, 'read-tree', pin)
    # No --force: an unexpected concurrent file is refused, not overwritten.
    call(git, root, 'checkout-index', '-z', '--stdin',
         data=('\0'.join(before['missing']) + '\0').encode('utf8'))
    for row in before['existing']:
        if hashlib.sha256((root / row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Existing file changed during adoption')
    missing = [p for p in before['missing'] if not (root / p).is_file()]
    diff = call(git, root, 'diff', '--name-only').decode('utf8').splitlines()
    # Audited pre-existing CRLF files are retained byte-for-byte even when
    # canonical Git text is LF. Report that difference, never rewrite it.
    preserved_paths = {row['path'] for row in before['existing']}
    if missing or set(diff) - preserved_paths or call(git, root, 'rev-parse', 'HEAD').decode().strip() != pin:
        raise RuntimeError('Adoption incomplete; preserve files and inspect Git state')
    return {'pin': pin, 'new_tracked_files': len(before['missing']),
            'existing_files_preserved': before['existing'], 'tracked_diff': diff,
            'runtime_subdirectories_changed': False, 'hardware_readiness_proven': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--git', default='git')
    parser.add_argument('--pin', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = (adopt if args.apply else audit)(args.root, args.store, args.git, args.pin)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
