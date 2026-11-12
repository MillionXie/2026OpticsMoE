"""Evaluate all 558 original TEST videos through the standalone pixel entry."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

if __package__:
    from . import _paths
else:
    import _paths

if __package__ and __package__ != 'tools':
    from .data import groups, load_inputs, sha256
    from ..infer import iter_video_fields
    from ..model import load_lightgen
    from .run import ROOT, ensure_idle_gpu, regression_metrics, write
else:
    from data import groups, load_inputs, sha256
    from infer import iter_video_fields
    from model import load_lightgen
    from run import ROOT, ensure_idle_gpu, regression_metrics, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT/'configs/selected.json')
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    profile = json.loads(args.profile.read_text('utf-8'))
    weights = ROOT/profile['checkpoint']
    if sha256(weights) != profile['checkpoint_sha256']:
        raise ValueError('Selected PT SHA differs')
    if args.device.startswith('cuda'):
        ensure_idle_gpu()
    torch.set_num_threads(4)
    model, metadata = load_lightgen(weights, args.device)
    with (weights.parent/'test_predictions.csv').open(encoding='utf-8-sig') as stream:
        saved_predictions = {row['sample_id']: float(row['prediction']) for row in csv.DictReader(stream)}
    payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
    if payload['identity']['manifest']['sha256'] != metadata['dataset']['manifest']['sha256']:
        raise ValueError('Manifest identity differs')
    indices, valid = groups(payload, 'test')
    real_indices = [i for i,s in enumerate(payload['splits']) if s == 'test']
    videos = [Path(payload['video_paths'][i]) for i in real_indices]
    if not all(path.is_file() for path in videos):
        raise FileNotFoundError('Some original videos are unavailable')
    args.out.mkdir(parents=True)
    captured = {}

    def capture_front(module, inputs, output):
        captured['features'] = output.detach()

    handle = model.front.register_forward_hook(capture_front)
    records = []
    cached_predictions = []
    native_predictions = []
    targets = []
    maximum_feature_error = 0.
    absolute_feature_error = 0.
    feature_values = 0
    started = time.time()
    try:
        for group, (chunk, pixels) in enumerate(iter_video_fields(videos)):
            index, keep = indices[group], valid[group]
            # Verify the actual decoded/padded pixels use the evaluator's IDs.
            expected_sources = [Path(payload['video_paths'][i]) for i in index.tolist()]
            padded_sources = list(chunk) + videos[:16-len(chunk)]
            if padded_sources != expected_sources:
                raise ValueError('Native and cached field membership differs')
            reference = payload['vision'][index][None].to(args.device, dtype=torch.float32)
            with torch.inference_mode():
                cached = model(reference)
                native = model(pixels=pixels.to(args.device))
                error = (captured['features'].float() - reference).abs()[0, keep]
                maximum_feature_error = max(maximum_feature_error, float(error.max()))
                absolute_feature_error += float(error.double().sum())
                feature_values += error.numel()
            for slot, i in enumerate(index.tolist()):
                if not bool(keep[slot]):
                    continue
                target = float(payload['target'][i])
                cached_prediction = float(cached['prediction'][0,slot])
                native_prediction = float(native['prediction'][0,slot])
                records.append({'sample_id': payload['ids'][i], 'group': group, 'slot': slot,
                                'target': target, 'cached_prediction': cached_prediction,
                                'native_prediction': native_prediction,
                                'difference': native_prediction-cached_prediction})
                targets.append(target)
                cached_predictions.append(cached_prediction)
                native_predictions.append(native_prediction)
            write(args.out/'status.json', {'state': 'running', 'groups_finished': group+1,
                                          'videos_finished': len(records), 'elapsed_seconds': time.time()-started})
            print(json.dumps({'groups_finished': group+1, 'videos_finished': len(records)}), flush=True)
    finally:
        handle.remove()
    if len(records) != 558 or len({row['sample_id'] for row in records}) != 558:
        raise ValueError('Native evaluation identity/count differs')
    with (args.out/'native_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    cached_metrics = regression_metrics(cached_predictions, targets)
    native_metrics = regression_metrics(native_predictions, targets)
    saved_metrics = regression_metrics(
        [saved_predictions[row['sample_id']] for row in records], targets)
    if abs(saved_metrics['srcc'] - profile['evaluation']['srcc']) > 1e-12:
        raise ValueError('Saved predictions differ from the selected result')
    differences = np.abs(np.asarray(native_predictions)-np.asarray(cached_predictions))
    cached_saved_error = max(abs(row['cached_prediction']-saved_predictions[row['sample_id']]) for row in records)
    if cached_saved_error > 2e-4:
        raise ValueError('Cached prediction differs beyond float32 batch-shape tolerance')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    record = {'state': 'completed', 'videos': 558, 'groups': len(indices), 'padding_excluded': True,
              'elapsed_seconds': time.time()-started, 'checkpoint_sha256': sha256(weights),
              'saved_metrics': saved_metrics, 'cached_metrics': cached_metrics, 'native_metrics': native_metrics,
              'maximum_MOS_difference': float(differences.max()), 'mean_MOS_difference': float(differences.mean()),
              'cached_batch1_vs_saved_batch4_max_MOS_difference': cached_saved_error,
              'maximum_feature_difference': maximum_feature_error,
              'mean_absolute_feature_difference': absolute_feature_error/feature_values,
              'native_SRCC_within_0_02_of_original_simulation': abs(native_metrics['srcc']-.8043868643) <= .02,
              'source_commit': os.environ.get('LIGHTGEN_SOURCE_COMMIT', commit),
              'source_SHA256': {p.name: sha256(p) for p in (ROOT/'model.py', ROOT/'infer.py', Path(__file__))},
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'device': args.device}, 'data_identity': payload['identity']}
    write(args.out/'audit.json', record)
    write(args.out/'status.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
