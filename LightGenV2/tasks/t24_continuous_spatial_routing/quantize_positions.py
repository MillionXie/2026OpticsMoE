"""Replay a locked continuous model on VAL after quantizing only its landing coordinates."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from .model import SpatialRoutingLG
from .train import REPO, TASK, encode, evaluate, load_raw, require_authorized_gpu, sha256, state_digest, write


class QuantizedPlacement(SpatialRoutingLG):
    def locate(self, amplitude):
        positions, power, capture = super().locate(amplitude)
        if self.quantization == 'pixel':
            positions = positions.round()
        elif self.quantization == 'grid17':
            nearest = (positions.unsqueeze(-1) - self.axis).abs().argmin(-1)
            positions = self.axis[nearest].to(positions.dtype)
        return positions, power, capture


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_validation(config, device):
    splits, identity = load_raw(Path(config['data_root']), config)
    val = splits['val']
    del splits
    if any(identity[k] != config[k] for k in ('manifest_sha256', 'index_sha256')):
        raise ValueError('Validation data identity mismatch')
    fields = []
    for start in range(0, len(val['ids']), 64):
        images = torch.from_numpy(val['images'][start:start + 64]).to(device)
        fields.append(encode(images).cpu())
    return torch.cat(fields), torch.as_tensor(val['labels'], dtype=torch.long), val['ids'].astype(str)


@torch.inference_mode()
def audit(run, out, device):
    if read_json(run / 'status.json')['state'] != 'complete':
        raise ValueError('A completed run is required')
    manifest = read_json(run / 'manifest.json')
    for name in ('metadata.json', 'selection_lock.json', 'best_checkpoint.pt', 'predictions_val.npz', 'result.json'):
        if sha256(run / name) != manifest[name]:
            raise ValueError(f'Run manifest mismatch: {name}')
    meta = read_json(run / 'metadata.json')
    lock = read_json(run / 'selection_lock.json')
    if lock['mode'] != 'continuous' or manifest['best_checkpoint.pt'] != lock['best_checkpoint_sha256']:
        raise ValueError('A locked continuous checkpoint is required')
    model_source = TASK / 'model.py'
    if sha256(model_source) != meta['source_sha256'][model_source.relative_to(REPO).as_posix()]:
        raise ValueError('Current model code differs from training')
    config = meta['config']
    gpu = require_authorized_gpu(config['authorized_gpu_uuids']) if device.type == 'cuda' else None
    payload = torch.load(run / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    if payload['epoch'] != lock['selected_epoch'] or payload['config'] != config:
        raise ValueError('Checkpoint metadata mismatch')
    model = QuantizedPlacement(mode='continuous', num_classes=config['num_classes']).to(device)
    model.load_state_dict(payload['model'], strict=True)
    parameter_sha = state_digest(model)
    inputs = load_validation(config, device)
    _, labels, ids = inputs
    expected_correct = read_json(run / 'result.json')['val']['correct']
    result = {
        'dataset': config['dataset'], 'seed': lock['seed'], 'run': run.relative_to(TASK).as_posix(),
        'split': 'val', 'selected_epoch': lock['selected_epoch'], 'best_checkpoint_sha256': lock['best_checkpoint_sha256'],
        'model_sha256': sha256(model_source), 'audit_source_sha256': sha256(Path(__file__)),
        'input_data_manifest_sha256': config['manifest_sha256'], 'original_validation_sha256': manifest['predictions_val.npz'],
        'gpu_uuid': gpu, 'torch_version': str(torch.__version__), 'command': sys.argv,
        'source_commit': meta['source_commit'], 'optimization_steps': 0, 'variants': {},
    }
    truth = labels.numpy()
    for mode in ('canonical', 'pixel', 'grid17'):
        model.quantization = mode
        path = out / f'val_{mode}.npz'
        metrics = evaluate(model, inputs, config['batch'], device, path)
        with np.load(path, allow_pickle=False) as current:
            predictions = current['predictions']
            positions = current['positions']
            if mode == 'canonical':
                with np.load(run / 'predictions_val.npz', allow_pickle=False) as original:
                    if not np.array_equal(ids, original['ids']) or not np.array_equal(truth, original['labels']):
                        raise ValueError('Replay has different validation rows or labels')
                    if not np.array_equal(predictions, original['scores'].argmax(1)) or metrics['correct'] != expected_correct:
                        raise ValueError('Canonical replay has different predicted classes')
                    result['canonical_max_abs_score_delta'] = float(np.max(np.abs(current['scores'] - original['scores'])))
                canonical_predictions = predictions
                canonical_positions = positions
                canonical_power = current['power']
            if not np.array_equal(canonical_power, current['power']):
                raise ValueError('Gate powers changed between evaluations')
        originally_correct = canonical_predictions == truth
        now_correct = predictions == truth
        result['variants'][mode] = {
            'n': metrics['n'], 'correct': metrics['correct'], 'accuracy': metrics['accuracy'],
            'mean_displacement_per_copy_px': float(np.linalg.norm(positions - canonical_positions, axis=-1).mean()),
            'prediction_changes': int((predictions != canonical_predictions).sum()),
            'correct_to_wrong': int((originally_correct & ~now_correct).sum()),
            'wrong_to_correct': int((~originally_correct & now_correct).sum()),
        }
    if state_digest(model) != parameter_sha:
        raise RuntimeError('Model parameters changed during evaluation')
    write(out / 'result.json', result)
    write(out / 'manifest.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.name != 'status.json'})
    write(out / 'status.json', {'state': 'complete', 'pid': os.getpid()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    if not run.is_relative_to(TASK / 'runs' / 'simulation') or not out.is_relative_to(TASK / 'runs' / 'smoke'):
        parser.error('Use a T24 simulation run and a fresh output directory under T24/runs/smoke')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    write(out / 'status.json', {'state': 'evaluating', 'pid': os.getpid()})
    try:
        audit(run, out, torch.device(args.device))
    except BaseException as exc:
        write(out / 'status.json', {'state': 'failed', 'error': str(exc), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
