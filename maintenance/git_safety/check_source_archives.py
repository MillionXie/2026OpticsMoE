"""Verify source-only archive receipts without moving, restoring or deleting files."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

MANIFESTS = (
    'REMOTE_STAGING_SOURCE_ARCHIVE_20261006.json',
    'REMOTE_PATCH_SOURCE_ARCHIVE_20261006.json',
    'LEGACY_MAINTENANCE_SOURCE_ARCHIVE_20261006.json',
    'T12_PERCEPTUAL_SOURCE_ARCHIVE_20261006.json',
)


def confined(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError('Manifest path escapes repository')
    return path


def inspect_manifest(root, manifest):
    errors = []
    archive = confined(root, manifest['archive_directory'])
    original = confined(root, manifest['original_directory'])
    checked = 0
    for row in manifest['files']:
        name = row['name']
        if Path(name).name != name or name in ('.', '..'):
            errors.append('Invalid source basename: '+name)
            continue
        path = confined(root, str(archive / name))
        if not path.is_file():
            errors.append('Missing archived source: '+name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append('Archived source SHA mismatch: '+name)
        else:
            checked += 1
        if (original / name).exists():
            errors.append('Original path reoccupied; review before restoring: '+name)
    reports = manifest.get('preserved_original_report_sha256', manifest.get('preserved_original_reports', {}))
    for relative, digest in reports.items():
        if manifest.get('preserved_report_location') == 'archive_directory_with_original_basename':
            path = confined(root, str(archive / Path(relative).name))
        else:
            path = confined(root, relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append('Preserved report missing or changed: '+relative)
    return {'sources_checked': checked, 'reports_checked': len(reports), 'errors': errors}


def inspect(root):
    rows = []
    for name in MANIFESTS:
        path = root / 'maintenance/storage' / name
        result = inspect_manifest(root, json.loads(path.read_text(encoding='utf8')))
        rows.append({'manifest': name, **result})
    return {'archives': rows, 'read_only': True, 'mutations': [],
            'errors': [error for row in rows for error in row['errors']],
            'runtime_compatibility_or_deletion_permission_proven': False}


def inspect_timing_manifest(root, descriptor):
    """Verify original payload bytes; only index text may normalize CRLF."""
    manifest = confined(root, descriptor['manifest'])
    raw = manifest.read_bytes()
    windows_text = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    if hashlib.sha256(windows_text).hexdigest() != descriptor['manifest_original_windows_sha256']:
        return {'payloads_checked': 0, 'errors': ['Timing manifest identity mismatch']}
    seen = set(); errors = []; checked = 0
    relocations = descriptor.get('relocations', {})
    for source, target in relocations.items():
        for name in (source, target):
            pure = PurePosixPath(name)
            if (pure.is_absolute() or '..' in pure.parts or '\\' in name
                    or ':' in name or name in ('', '.')):
                raise ValueError('Invalid timing relocation path')
    for line in raw.decode('utf8').splitlines():
        digest, name = line.split(None, 1); name = name.strip()
        pure = PurePosixPath(name)
        if (not re.fullmatch('[0-9a-fA-F]{64}', digest) or pure.is_absolute()
                or '..' in pure.parts or '\\' in name or ':' in name or name in seen):
            raise ValueError('Invalid or duplicate timing payload path')
        seen.add(name)
        path = confined(manifest.parent, relocations.get(name, name))
        if os.name == 'nt':
            absolute = str(path)
            path = Path('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\')
                        else '\\\\?\\' + absolute)
        if not path.is_file():
            errors.append('Missing timing payload: ' + name); continue
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != digest.lower():
            errors.append('Timing payload SHA mismatch: ' + name)
        else:
            checked += 1
    if len(seen) != descriptor['manifest_entries_verified']:
        errors.append('Timing manifest member count mismatch')
    if set(relocations) - seen:
        errors.append('Relocation references undeclared timing member')
    return {'payloads_checked': checked, 'errors': errors,
            'explicit_relocations_checked': len(set(relocations) & seen),
            'scope': 'Declared original timing export bytes only; not new measurements or unlisted files'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--timing-payloads', action='store_true',
                        help='Also verify locally retained historical timing payloads; requires private files')
    parser.add_argument('--demo-timing-payloads', action='store_true',
                        help='Audit Sep 27 historical payloads including explicit relocations; requires local files and reports unresolved members')
    args = parser.parse_args()
    result = inspect(args.repo)
    if args.demo_timing_payloads:
        descriptor = json.loads((args.repo/'maintenance/storage/DEMO_TIMING_MANIFEST_AUDIT_20261006.json').read_text(encoding='utf8'))['descriptor']
        result['demo_timing_payloads'] = inspect_timing_manifest(args.repo, descriptor)
        result['errors'].extend(result['demo_timing_payloads']['errors'])
    if args.timing_payloads:
        descriptor = json.loads((args.repo/'maintenance/storage/TIMING_SOURCE_RECOVERY_20261004.json').read_text(encoding='utf8'))['historical_redbox_payload_visibility_20261006']
        result['timing_payloads'] = inspect_timing_manifest(args.repo, descriptor)
        result['errors'].extend(result['timing_payloads']['errors'])
    print(json.dumps(result, indent=2))
    return bool(result['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
