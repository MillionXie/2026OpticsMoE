"""Verify source-only archive receipts without moving, restoring or deleting files."""
import argparse
import hashlib
import json
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = inspect(args.repo)
    print(json.dumps(result, indent=2))
    return bool(result['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
