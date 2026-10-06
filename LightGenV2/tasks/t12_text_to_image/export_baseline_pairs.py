"""Export paired inputs/targets without relying on an external handoff package."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image

EXPECTED_COUNTS = {'train': 20736, 'val': 2304, 'test': 2304}
METADATA_KEYS = ('sample_id', 'source_id', 'target_id', 'category', 'mode',
                 'source_scene', 'target_scene', 'prompt')


def save_tensor(value, path):
    array = value.detach().cpu().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)


def export_pairs(datasets, output, *, limit=0, manifest_format='jsonl'):
    """Validate every split before creating a new output; never overwrite a run."""
    output = Path(output)
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    if not datasets or set(datasets) - set(EXPECTED_COUNTS):
        raise ValueError('Expected train/val/test datasets')
    if manifest_format not in ('jsonl', 'csv') or (manifest_format == 'csv' and len(datasets) != 1):
        raise ValueError('CSV legacy layout requires exactly one split')
    for split, dataset in datasets.items():
        if len(dataset) != EXPECTED_COUNTS[split]:
            raise ValueError(f'{split} count {len(dataset)} != {EXPECTED_COUNTS[split]}')
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split, dataset in datasets.items():
        count = min(len(dataset), limit) if limit else len(dataset)
        images = output if manifest_format == 'csv' else output / split
        (images / 'input').mkdir(parents=True)
        (images / 'target').mkdir()
        with (images / ('pairs.' + manifest_format)).open('x', encoding='utf-8', newline='') as stream:
            columns = ('index', *METADATA_KEYS, 'input_png', 'target_png')
            writer = csv.DictWriter(stream, fieldnames=columns) if manifest_format == 'csv' else None
            if writer:
                writer.writeheader()
            for index in range(count):
                row = dataset[index]
                pair_id = f'{split}_{index:05d}'
                for role, key in (('input', 'reference'), ('target', 'target')):
                    save_tensor(row[key], images / role / f'{pair_id}.png')
                metadata = {key: row[key] for key in METADATA_KEYS}
                metadata.update(pair_id=pair_id, input_path=f'input/{pair_id}.png',
                                target_path=f'target/{pair_id}.png')
                if writer:
                    writer.writerow({'index': index, **{key: row[key] for key in METADATA_KEYS},
                                     'input_png': metadata['input_path'], 'target_png': metadata['target_path']})
                else:
                    stream.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        counts[split] = count
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--instruction-cache', type=Path, required=True)
    parser.add_argument('--split', choices=(*EXPECTED_COUNTS, 'all'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--manifest-format', choices=('jsonl', 'csv'), default='jsonl',
                        help='CSV preserves the old single-split flat input/target layout')
    args = parser.parse_args()
    if args.limit < 0 or args.output.exists() or (args.manifest_format == 'csv' and args.split == 'all'):
        parser.error('Use a nonnegative limit and a new output directory')
    from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
    splits = tuple(EXPECTED_COUNTS) if args.split == 'all' else (args.split,)
    datasets = {split: ExpandedUnifiedProductEditDataset(
        args.dataset_root, split, 256, args.instruction_cache) for split in splits}
    print(json.dumps(export_pairs(datasets, args.output, limit=args.limit, manifest_format=args.manifest_format)))


if __name__ == '__main__':
    main()
