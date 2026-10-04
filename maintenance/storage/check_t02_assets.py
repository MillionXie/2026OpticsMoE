"""Read-only T02 official/provisional asset check, without models or evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from check_t01_assets import file_identity, image_content


def personal_content(directory: Path, annotations: dict) -> dict:
    digest, seen = hashlib.sha256(), set()
    root = directory.resolve(strict=True)
    for row in sorted(annotations['images'], key=lambda item: item['id']):
        if row['id'] in seen:
            raise ValueError(f'Duplicate personal photo ID: {row["id"]}')
        seen.add(row['id'])
        relative = Path(row['image'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Personal image path escapes its dataset')
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
        info = file_identity(path)
        if info['sha256'] != row['image_sha256']:
            raise ValueError(f'Personal image SHA differs from annotation: {relative}')
        item = {'id': row['id'], 'image': row['image'], **info}
        digest.update((json.dumps(item, sort_keys=True, separators=(',', ':')) + '\n').encode())
    return {'image_count': len(seen), 'image_content_manifest_sha256': digest.hexdigest()}


def check(receipt: dict, repo_root: Path) -> dict:
    official, errors, checked = receipt['official'], [], 0
    data = repo_root / 'data/lsp_pose'
    runs = repo_root / 'LightGenV2/tasks/t02_keypoint_detection/runs/simulation'

    def compare(path: Path, expected: dict):
        nonlocal checked
        try:
            if file_identity(path) != {k: expected[k] for k in ['bytes', 'sha256']}:
                errors.append(f'Asset identity mismatch: {path}')
            checked += 1
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{path}: {exc}')

    for name, expected in official['run_manifests'].items():
        compare(runs / name / 'pose_protocol_split.csv', expected)
    try:
        first = next(iter(official['run_manifests']))
        with (runs / first / 'pose_protocol_split.csv').open(encoding='utf-8', newline='') as stream:
            actual = image_content(list(csv.DictReader(stream)), data, Path(official['dataset_root']))
        if actual != official['image_content']:
            errors.append('Official image content manifest mismatch')
        checked += actual['count']
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f'Official images: {exc}')
    for relative, expected in official['annotations'].items():
        compare(data / relative, expected)
    for name, files in official['artifacts'].items():
        for relative, expected in files.items():
            compare(runs / name / relative, expected)
    recorded_repo = Path(official['dataset_root']).parent.parent
    for label in ['shared_initialization', 'distillation_teacher_cache', 'baseline_deconv128']:
        asset = receipt[label]
        compare(repo_root / Path(asset['path']).relative_to(recorded_repo), asset)
    for name, expected in receipt['personal_provisional'].items():
        directory = data / name
        for relative, info in expected['metadata_files'].items():
            compare(directory / relative, info)
        try:
            annotations = json.loads((directory / 'annotations_provisional.json').read_text(encoding='utf-8'))
            actual = personal_content(directory, annotations)
            if any(value != expected[key] for key, value in actual.items()):
                errors.append(f'Personal content manifest mismatch: {name}')
            checked += actual['image_count']
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'Personal images {name}: {exc}')
    for run in receipt['personal_runs']:
        for relative, expected in run['assets'].items():
            compare(runs / run['run'] / relative, expected)
    return {'read_only': True, 'files_checked': checked, 'errors': errors,
            'models_loaded': False, 'scientific_metrics_recomputed': False,
            'personal_annotations_remain_provisional': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--receipt', type=Path, default=Path(__file__).with_name('T02_CONTENT_ASSETS_20261004.json'))
    args = parser.parse_args()
    report = check(json.loads(args.receipt.read_text(encoding='utf-8')), args.repo_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
