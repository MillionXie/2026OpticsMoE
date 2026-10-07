"""Read-only SHA/CRC audit of the six historical baseline assets; no model execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile


def readable_path(path: Path) -> Path:
    # Retained snapshots preserve very long original experiment names.
    # Native Windows MAX_PATH is not evidence that the asset is missing.
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        return Path('\\\\?\\' + str(path.resolve()))
    return path


def source_identity(payload: bytes | None, expected: str) -> str:
    if payload is None:
        return 'missing'
    if hashlib.sha256(payload).hexdigest() == expected:
        return 'exact'
    if hashlib.sha256(payload.replace(b'\r\n', b'\n')).hexdigest() == expected:
        return 'line_endings_only_not_accepted'
    return 'content_difference_not_accepted'


def inspect(root: Path) -> dict:
    base = root / 'LightGenV2/reports/20260915_baseline_methods/code_packages'
    build = json.loads((base / 'BUILD_MANIFEST.json').read_text(encoding='utf8'))
    packages = []
    for package in build['tasks']:
        name = package['task']
        if Path(name).name != name:
            raise ValueError('Package name escapes asset directory')
        folder = base / name
        manifest = json.loads((folder / 'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
        counts, differences = {}, []
        for row in manifest['files']:
            path = (folder / 'source' / row['path']).resolve()
            if not path.is_relative_to((folder / 'source').resolve()):
                raise ValueError('Manifest source escapes package')
            native = readable_path(path)
            result = source_identity(native.read_bytes() if native.is_file() else None, row['sha256'])
            counts[result] = counts.get(result, 0) + 1
            if result != 'exact':
                differences.append({'path': row['path'], 'classification': result})
        archive_path = base / (name + '.zip')
        archive = {'exists': archive_path.is_file(), 'sha256_matches': False, 'crc_valid': False,
                   'source_members_match': False}
        if archive_path.is_file():
            with archive_path.open('rb') as handle:
                archive['sha256_matches'] = hashlib.file_digest(handle, 'sha256').hexdigest() == package['sha256']
            with zipfile.ZipFile(archive_path) as handle:
                archive['crc_valid'] = handle.testzip() is None
                archive['source_members_match'] = all(
                    hashlib.sha256(handle.read(name + '/source/' + row['path'])).hexdigest() == row['sha256']
                    for row in manifest['files'])
        packages.append({'task': name, 'source_commit': manifest['source_commit'],
                         'files': len(manifest['files']), 'source_counts': counts,
                         'differences': differences, 'original_zip': archive})
    return {'read_only': True, 'models_or_data_evaluated': False,
            'packages': packages, 'all_current_source_bytes_exact': all(not p['differences'] for p in packages),
            'all_original_archives_verified': all(all(p['original_zip'].values()) for p in packages)}


if __name__ == '__main__':
    report = inspect(Path(__file__).resolve().parents[2])
    print(json.dumps(report, indent=2))
    raise SystemExit(not (report['all_current_source_bytes_exact'] and report['all_original_archives_verified']))
