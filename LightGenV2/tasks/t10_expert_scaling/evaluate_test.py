"""One-time evaluation of a locked checkpoint on the held-out test split."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .model import ScalingOptics
from .plan import load_protocol
from .train import evaluate, sha


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--test-source', type=Path,
                    help='Optional immutable source archive containing the official test split. '
                         'Training-cache identity is still verified with --data.')
    ap.add_argument('--selection-lock', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=2)
    args = ap.parse_args()

    lock = json.loads(args.selection_lock.read_text())
    locked = next((x for x in lock['runs'] if Path(x['run_dir']).resolve() == args.run.resolve()), None)
    if locked is None:
        raise ValueError(f'Run is absent from selection lock: {args.run}')
    result = json.loads((args.run / 'result.json').read_text())
    metadata = json.loads((args.run / 'metadata.json').read_text())
    checkpoint = args.run / 'best_checkpoint.pt'
    if sha(checkpoint) != result['checkpoint_sha256'] or result['checkpoint_sha256'] != locked['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from locked selection')
    if sha(args.data) != metadata['data_sha256']:
        raise ValueError('Dataset hash differs from training metadata')

    manifest_path = args.data.parent / 'data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if sha(args.data) != manifest['cache_sha256']:
        raise ValueError('Dataset hash differs from data manifest')
    test_archive = args.test_source or args.data
    if args.test_source and sha(args.test_source) != manifest['source_sha256']:
        raise ValueError('Official test-source hash differs from data manifest')
    archive = np.load(test_archive, allow_pickle=False)
    images = archive['test_images']
    labels = archive['test_labels'].reshape(-1).astype(np.int64)
    if 'test_ids' in archive.files:
        ids = archive['test_ids'].astype(str)
    else:
        ids = np.asarray([f"{manifest['dataset']}:test:{i:06d}" for i in range(len(labels))])
    archive.close()
    if images.ndim == 3:
        images = np.repeat(images[..., None], 3, axis=-1)
    if 'test' in manifest['split_ids'] and ids.tolist() != manifest['split_ids']['test']:
        raise ValueError('Test IDs differ from manifest')
    if set(ids) & set(manifest['split_ids']['train']) or set(ids) & set(manifest['split_ids']['val']):
        raise ValueError('Test IDs overlap train or validation IDs')
    split = (torch.from_numpy(images.transpose(0, 3, 1, 2).copy()), torch.from_numpy(labels), ids)

    actual = metadata['arguments']
    cfg = load_protocol()
    model = ScalingOptics(cfg, int(actual['experts']), int(actual['top_k']), actual['arch'],
                          len(manifest['classes']), int(actual['layers']), int(actual['padding'])).cuda()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    started = time.time()
    metrics, probabilities = evaluate(model, split, args.batch)
    predictions = probabilities.argmax(1).numpy()
    np.savez_compressed(args.run / 'test_predictions.npz', ids=ids, labels=labels,
                        scores=probabilities.numpy(), predictions=predictions)
    payload = dict(
        state='complete', selection_lock_sha256=sha(args.selection_lock),
        checkpoint_sha256=result['checkpoint_sha256'], data_sha256=sha(args.data),
        data_manifest_sha256=sha(manifest_path), test_source_sha256=sha(test_archive),
        test_ids_sha256=hashlib.sha256(
            '\n'.join(ids.tolist()).encode()).hexdigest(), test_samples=len(ids),
        test=metrics, elapsed_seconds=time.time()-started,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        device=torch.cuda.get_device_name(), source_commit=metadata['git_commit'])
    save(args.run / 'test_result.json', payload)
    print(json.dumps({'run': args.run.name, 'test_accuracy': metrics['accuracy'],
                      'test_macro_f1': metrics['macro_f1'], 'elapsed_seconds': payload['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
