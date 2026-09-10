"""CPU-only COCO pool audit; no download, GPU, pseudo-labels or training.

COCO train2017 includes SALICON public-test images. Exclude BOTH SALICON splits
by numeric COCO ID, then by exact file SHA256, before any teacher inference.
"""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_index(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    for path in sorted(root.glob('*.jpg')):
        try:
            image_id = int(path.stem.rsplit('_', 1)[-1])
        except ValueError as error:
            raise ValueError(f'Non-COCO filename: {path.name}') from error
        if image_id in result:
            raise ValueError(f'Duplicate numeric COCO ID: {image_id}')
        result[image_id] = path
    if not result:
        raise ValueError(f'No JPEG images: {root}')
    return result


def ids_sha(ids):
    return hashlib.sha256(('\n'.join(map(str, sorted(ids)))+'\n').encode()).hexdigest()


def curate(coco_root, salicon_root, count=20000, seed=17042, expected_counts=(10000, 5000)):
    coco = image_index(coco_root)
    train = image_index(Path(salicon_root)/'images/train')
    test = image_index(Path(salicon_root)/'images/val')
    if (len(train), len(test)) != tuple(expected_counts):
        raise ValueError('SALICON exclusion roots do not contain the expected full splits')
    if set(train) & set(test):
        raise ValueError('SALICON train/test COCO ID overlap')
    eligible = sorted(set(coco) - set(train) - set(test))
    if not 0 < count <= len(eligible):
        raise ValueError('Requested pool count exceeds disjoint available images')
    selected = sorted(random.Random(seed).sample(eligible, count))
    # Hash the existing complete exclusion set, not only filenames/selected IDs.
    # At most one image is read at once; no additional copy of the dataset.
    blocked_hashes = {file_sha(p) for p in [*train.values(), *test.values()]}
    seen, records, rejected = set(), [], []
    for index, image_id in enumerate(selected, 1):
        path = coco[image_id]
        digest = file_sha(path)
        if digest in blocked_hashes or digest in seen:
            rejected.append({'image_id': image_id, 'reason': 'excluded_or_duplicate_file_sha256'})
        else:
            seen.add(digest)
            records.append({'image_id': image_id, 'image_file': path.name,
                            'file_sha256': digest, 'bytes': path.stat().st_size})
        if index % 2000 == 0:
            print(f'[unlabeled pool CPU] checked {index}/{count}', flush=True)
    if not records:
        raise ValueError('No images remain after content exclusions')
    return {
        'schema_version': 1, 'status': 'audited_images_only_not_teacher_predictions',
        'coco_root': str(Path(coco_root).resolve()),
        'salicon_root': str(Path(salicon_root).resolve()), 'seed': seed,
        'salicon_train_count': len(train), 'salicon_test_count': len(test),
        'salicon_train_ids_sha256': ids_sha(train), 'salicon_test_ids_sha256': ids_sha(test),
        'coco_train2017_count': len(coco),
        'overlap_with_salicon_train_before_exclusion': len(set(coco) & set(train)),
        'overlap_with_salicon_test_before_exclusion': len(set(coco) & set(test)),
        'eligible_after_id_exclusion': len(eligible),
        'requested_count': count, 'selected_count': len(records),
        'selected_ids_sha256': ids_sha(r['image_id'] for r in records),
        'ids_hash_convention': 'numeric sorted decimal IDs, newline separated, trailing newline',
        'existing_image_bytes': sum(r['bytes'] for r in records),
        'new_image_bytes_written': 0, 'teacher_predictions_generated': False,
        'train_id_overlap_after_exclusion': 0, 'test_id_overlap_after_exclusion': 0,
        'exclusion': 'full SALICON train and val IDs plus exact file SHA256; selected files deduplicated',
        'limitations': 'Does not detect visually similar images or re-encoded duplicates with different IDs',
        'rejected_content_duplicates': rejected, 'images': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path, required=True)
    parser.add_argument('--salicon-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=17042)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = curate(args.coco_root, args.salicon_root, args.count, args.seed)
    report['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never overwrite an existing audited pool.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'images'}, indent=2))
    print('MANIFEST_SHA256', file_sha(args.output))


if __name__ == '__main__':
    main()
