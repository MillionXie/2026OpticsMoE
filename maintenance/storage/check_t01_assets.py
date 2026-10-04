"""Read-only T01 reproduction assets checker; no model, download or run writes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def file_identity(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise ValueError(f'Asset changed during hashing: {path}')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def image_content(rows: list[dict], dataset_root: Path,
                  recorded_root: Path) -> dict:
    ids, paths, counts, digest, total = set(), set(), Counter(), hashlib.sha256(), 0
    root = dataset_root.resolve(strict=True)
    for row in sorted(rows, key=lambda r: r['sample_id']):
        sid = row['sample_id']
        if sid in ids:
            raise ValueError(f'Duplicate sample identity: {sid}')
        ids.add(sid)
        relative = Path(row['image_path']).relative_to(recorded_root)
        if '..' in relative.parts:
            raise ValueError(f'Image path traversal: {relative}')
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)  # Reject traversal and symlinks leaving the data root.
        if relative.as_posix() in paths:
            raise ValueError(f'Repeated image path: {relative}')
        paths.add(relative.as_posix())
        info = file_identity(path)
        item = {'sample_id': sid, 'split': row['split'],
                'relative_path': relative.as_posix(), **info}
        digest.update((json.dumps(item, ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':')) + '\n').encode('utf-8'))
        counts[row['split']] += 1
        total += info['bytes']
    return {'count': len(ids), 'split_counts': dict(counts), 'total_bytes': total,
            'content_manifest_sha256': digest.hexdigest()}


def check(receipt: dict, repo_root: Path, frontend_snapshot: Path | None = None) -> dict:
    errors, checked = [], 0
    task = repo_root / 'LightGenV2/tasks/t01_object_retrieval/runs/simulation'

    def compare(path, expected):
        nonlocal checked
        try:
            actual = file_identity(path)
            checked += 1
            if actual != expected:
                errors.append(f'Content identity mismatch: {path}')
        except (OSError, ValueError) as exc:
            errors.append(f'{path}: {exc}')

    for name, expected in receipt['run_manifests'].items():
        compare(task / name / 'manifests/caltech101_10class_subset.csv', expected)
    first = next(iter(receipt['run_manifests']))
    try:
        with (task / first / 'manifests/caltech101_10class_subset.csv').open(
            encoding='utf-8', newline=''
        ) as stream:
            rows = list(csv.DictReader(stream))
        recorded = Path(receipt['dataset']['root'])
        actual = image_content(rows, repo_root / 'data/Caltech101', recorded)
        checked += actual['count']
        for key, value in actual.items():
            if value != receipt['dataset'][key]:
                errors.append(f'Dataset identity mismatch: {key}')
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f'Dataset: {exc}')
    for name, files in receipt['student_checkpoints'].items():
        for label, expected in files.items():
            compare(task / name / label, expected)
    warm = receipt['initialization_checkpoint']
    recorded_repo = Path(receipt['dataset']['root']).parent.parent
    warm_relative = Path(warm['path']).relative_to(recorded_repo)
    compare(repo_root / warm_relative, {k: warm[k] for k in ['bytes', 'sha256']})
    snapshot = frontend_snapshot or Path(receipt['frozen_frontend']['snapshot_path'])
    for relative, expected in receipt['frozen_frontend']['files'].items():
        compare(snapshot / relative, expected)
    return {'read_only': True, 'files_checked': checked, 'errors': errors,
            'historical_frontend_revision_proven': False,
            'scientific_metrics_recomputed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--receipt', type=Path, default=Path(__file__).with_name('T01_CONTENT_ASSETS_20261004.json'))
    parser.add_argument('--frontend-snapshot', type=Path)
    args = parser.parse_args()
    report = check(json.loads(args.receipt.read_text(encoding='utf-8')), args.repo_root,
                   args.frontend_snapshot)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
