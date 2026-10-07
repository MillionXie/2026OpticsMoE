"""Read-only identity gate for the selected layered application assets."""
import argparse
import hashlib
import json
from pathlib import Path

IDENTITY = Path(__file__).with_name('layered_asset_identity_20261007.json')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inspect(dataset_dir, svg_dir, expected=None):
    expected = expected if expected is not None else json.loads(IDENTITY.read_text(encoding='utf8'))
    errors = []
    checked = []
    for name, record in expected['dataset_files'].items():
        path = Path(dataset_dir) / name
        if not path.is_file():
            errors.append({'path': str(path), 'error': 'missing'})
            continue
        if digest(path) != record['sha256']:
            errors.append({'path': str(path), 'error': 'sha256_mismatch'})
            continue
        if 'records' in record:
            with path.open(encoding='utf8') as stream:
                count = sum(1 for line in stream if line.strip())
            if count != record['records']:
                errors.append({'path': str(path), 'error': 'record_count_mismatch'})
        if 'bytes' in record and path.stat().st_size != record['bytes']:
            errors.append({'path': str(path), 'error': 'size_mismatch'})
        checked.append(str(path))
    for record in expected['svg_files']:
        path = Path(svg_dir) / record['path']
        if not path.is_file():
            errors.append({'path': str(path), 'error': 'missing'})
        elif digest(path) != record['sha256']:
            errors.append({'path': str(path), 'error': 'sha256_mismatch'})
        else:
            checked.append(str(path))
    return {'read_only': True, 'checked_files': checked, 'errors': errors,
            'assets_regenerated': False, 'models_loaded': False, 'test_evaluated': False,
            'limits': 'No tensor-to-sample binding, per-image pixel identity or scientific performance validation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--svg-dir', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.dataset_dir, args.svg_dir)
    print(json.dumps(result, indent=2))
    return 1 if result['errors'] else 0


def bind_existing(settings, task_asset_root):
    """Bind only the selected application; never invoke a preparation routine."""
    if settings.layout_version != 'layered_anchor6_svg_v3' or not settings.embedding_only or settings.qwen_shared_baseline:
        raise ValueError('Existing asset binding requires the layered embedding application')
    if settings.prompt_cache_path.name != 'token_embeddings_v1.pt':
        raise ValueError('Unexpected application cache name')
    base = Path(task_asset_root).expanduser().resolve()
    data = base / 'dataset/openmoji_layered_anchor6_svg_v3'
    svg = base / 'assets/openmoji-17.0.0-svg'
    result = inspect(data, svg)
    if result['errors']:
        raise ValueError('Existing layered assets failed identity check: ' + json.dumps(result['errors']))
    expected = json.loads(IDENTITY.read_text(encoding='utf8'))
    summary = json.loads((data / 'dataset_summary.json').read_text(encoding='utf8'))
    if summary.get('type') != 'openmoji_layered_anchor6_proportional_svg_v3' or summary.get('seed') != settings.seed:
        raise ValueError('Existing layered dataset summary has a different protocol')
    for split in ('train', 'test'):
        identity = expected['dataset_files'][split + '.jsonl']
        if summary[split]['sha256'] != identity['sha256'] or summary[split]['samples'] != identity['records']:
            raise ValueError('Existing layered summary differs from pinned split')
    settings.data_dir = data
    settings.svg_asset_dir = svg
    settings.prompt_cache_path = data / 'token_embeddings_v1.pt'
    return summary


if __name__ == '__main__':
    raise SystemExit(main())
