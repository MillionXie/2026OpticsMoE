"""Read-only T08 content audit. Never reconstruct a cache or evaluate queries."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

try:
    from .check_t01_assets import file_identity, image_content
except ImportError:
    from check_t01_assets import file_identity, image_content


def asset_path(root, value):
    relative = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if relative.is_absolute() or windows.drive or '\\' in value or '..' in relative.parts:
        raise ValueError('Expected a repository-relative asset path')
    target = root / relative
    target.resolve().relative_to(root.resolve())
    return target


def relative_image_content(rows, dataset_root):
    converted = []
    for row in rows:
        relative = Path(row['image_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Expected a dataset-relative image path')
        converted.append(dict(row, image_path=str(dataset_root / relative)))
    return image_content(converted, dataset_root, dataset_root)


def audit_split(manifest, train, test, titles):
    def ids(rows):
        result = {r['sample_id'] for r in rows}
        if len(result) != len(rows):
            raise ValueError('Repeated sample identity')
        return result
    combined, fit, query = ids(manifest), ids(train), ids(test)
    if fit & query or combined != fit | query:
        raise ValueError('Train/test identity overlap or missing samples')
    by_id = {r['sample_id']: r for r in manifest}
    for split, rows in [('train', train), ('test', test)]:
        for row in rows:
            if row != by_id[row['sample_id']] or row['split'] != split:
                raise ValueError('Split rows differ from the original manifest')
    products = {r['product_id'] for r in titles}
    if len(products) != len(titles):
        raise ValueError('Repeated title product identity')
    if products != {r['product_id'] for r in manifest}:
        raise ValueError('Title candidates differ from image product identities')
    return {'train': len(train), 'test': len(test), 'titles': len(titles)}


def check(receipt, repo_root):
    dataset = repo_root / 'data/abo_easy100_dataset_20260906'
    runs = repo_root / 'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation'
    errors, checked = [], 0
    def compare(path, expected):
        nonlocal checked
        try:
            actual = file_identity(path)
            checked += 1
            if actual != expected:
                errors.append(f'Content identity mismatch: {path}')
        except (OSError, ValueError) as exc:
            errors.append(f'{path}: {exc}')
    for filename, expected in receipt['dataset_csv'].items():
        compare(dataset / filename, expected)
    try:
        rows = {}
        for name in ['manifest.csv', 'train.csv', 'test.csv', 'titles.csv']:
            with (dataset / name).open(encoding='utf-8', newline='') as stream:
                rows[name] = list(csv.DictReader(stream))
        audit_split(*(rows[n] for n in ['manifest.csv', 'train.csv', 'test.csv', 'titles.csv']))
        actual = relative_image_content(rows['manifest.csv'], dataset)
        checked += actual['count']
        if actual != receipt['image_content']:
            errors.append('Dataset image-content identity mismatch')
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f'Dataset: {exc}')
    for run, files in receipt['student_artifacts'].items():
        for filename, expected in files.items():
            compare(runs / run / filename, expected)
    for relative, expected in receipt.get('adopted_readout_artifacts', {}).items():
        try:
            compare(asset_path(repo_root, relative), expected)
        except ValueError as exc:
            errors.append(f'Asset path: {exc}')
    # Cache absence is an explicit reproduction dependency, not permission to
    # regenerate embeddings, change prompts, or substitute the reverse cache.
    return {'read_only': True, 'files_checked': checked, 'errors': errors,
            'scientific_metrics_recomputed': False,
            'teacher_cache_status': receipt['teacher_cache_status'],
            'full_reproduction_ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--receipt', type=Path,
                        default=Path(__file__).with_name('T08_CONTENT_ASSETS_20261004.json'))
    args = parser.parse_args()
    result = check(json.loads(args.receipt.read_text(encoding='utf-8')), args.repo_root)
    print(json.dumps(result, indent=2))
    return int(bool(result['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
