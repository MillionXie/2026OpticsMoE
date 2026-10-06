"""Read-only review-package identities; never inference, extraction or cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(root):
    root = Path(root).resolve()
    manifest_path = root / 'SHA256.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    failures = []
    for relative, expected in manifest.items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe manifest path: ' + relative)
        if not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid SHA: ' + relative)
        if not path.is_file():
            failures.append({'path': relative, 'reason': 'missing'})
        elif sha(path) != expected:
            failures.append({'path': relative, 'reason': 'sha_mismatch'})
    for key in ('release.json', 'weights/best_checkpoint.pt'):
        if key not in manifest:
            failures.append({'path': key, 'reason': 'not_manifest_bound'})
    release = json.loads((root / 'release.json').read_text(encoding='utf8'))
    fields = release['fields']
    ids = []
    for field in fields:
        if field['file'] not in manifest or field['sha256'] != manifest[field['file']]:
            failures.append({'path': field['file'], 'reason': 'field_identity'})
        if not (len(field['valid']) == len(field['sample_ids']) == len(field['targets'])):
            raise ValueError('Field slot identity mismatch')
        ids.extend(sid for sid, valid in zip(field['sample_ids'], field['valid']) if valid)
    if len(ids) != release['test_videos_in_package'] or len(set(ids)) != len(ids):
        failures.append({'path': 'release.json', 'reason': 'valid_video_identity'})
    if manifest.get('weights/best_checkpoint.pt') != release['checkpoint_sha256']:
        failures.append({'path': 'weights/best_checkpoint.pt', 'reason': 'checkpoint_identity'})
    return {'root': str(root), 'manifest_sha256': sha(manifest_path), 'files_checked': len(manifest),
            'checkpoint_sha256': release['checkpoint_sha256'], 'fields': len(fields),
            'valid_videos': len(ids), 'unique_valid_videos': len(set(ids)),
            'declared_runtime_commit': release.get('reproduction_runtime_commit'),
            'recorded_simulation_metrics': release['simulation_metrics'],
            'declared_reference_srcc': release.get('reference_srcc'),
            'input_hashes': {f['file']: f['sha256'] for f in fields},
            'source_hashes': {p: h for p, h in manifest.items() if p.endswith(('.py', '.yaml'))},
            'model_selection': release.get('model_selection'), 'failures': failures,
            'integrity_passed': not failures, 'model_evaluated': False, 'mutations': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result['integrity_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
