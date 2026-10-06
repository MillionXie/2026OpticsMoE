"""Export paired inputs/targets without relying on an external handoff package."""
import argparse
import json
from pathlib import Path

from PIL import Image

EXPECTED_COUNTS = {'train': 20736, 'val': 2304, 'test': 2304}
METADATA_KEYS = ('sample_id', 'source_id', 'target_id', 'category', 'mode',
                 'source_scene', 'target_scene', 'prompt')


def save_tensor(value, path):
    array = value.clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)


def export_pairs(datasets, output, *, limit=0):
    """Validate every split before creating a new output; never overwrite a run."""
    output = Path(output)
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    if not datasets or set(datasets) - set(EXPECTED_COUNTS):
        raise ValueError('Expected train/val/test datasets')
    for split, dataset in datasets.items():
        if len(dataset) != EXPECTED_COUNTS[split]:
            raise ValueError(f'{split} count {len(dataset)} != {EXPECTED_COUNTS[split]}')
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split, dataset in datasets.items():
        count = min(len(dataset), limit) if limit else len(dataset)
        images = output / split
        (images / 'input').mkdir(parents=True)
        (images / 'target').mkdir()
        with (images / 'pairs.jsonl').open('x', encoding='utf-8') as stream:
            for index in range(count):
                row = dataset[index]
                pair_id = f'{split}_{index:05d}'
                for role, key in (('input', 'reference'), ('target', 'target')):
                    save_tensor(row[key], images / role / f'{pair_id}.png')
                metadata = {key: row[key] for key in METADATA_KEYS}
                metadata.update(pair_id=pair_id, input_path=f'input/{pair_id}.png',
                                target_path=f'target/{pair_id}.png')
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
    args = parser.parse_args()
    if args.limit < 0 or args.output.exists():
        parser.error('Use a nonnegative limit and a new output directory')
    from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
    splits = tuple(EXPECTED_COUNTS) if args.split == 'all' else (args.split,)
    datasets = {split: ExpandedUnifiedProductEditDataset(
        args.dataset_root, split, 256, args.instruction_cache) for split in splits}
    print(json.dumps(export_pairs(datasets, args.output, limit=args.limit)))


if __name__ == '__main__':
    main()
