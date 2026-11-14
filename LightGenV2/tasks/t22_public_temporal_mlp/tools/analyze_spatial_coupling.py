"""Quantify residual inter-video coupling with a controlled optical diagnostic."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch

if __package__:
    from . import _paths
    from .analyze_global_ablation import evaluate_condition
    from .checkpoint_utils import tensor_fingerprint
    from .data import load_inputs, sha256
    from .run import write
    from .spatial_locality import STAGES, video_region_masks
    from ..model import load_lightgen
else:
    import _paths
    from analyze_global_ablation import evaluate_condition
    from checkpoint_utils import tensor_fingerprint
    from data import load_inputs, sha256
    from run import write
    from spatial_locality import STAGES, video_region_masks
    from model import load_lightgen


def own_video_propagation(stage, original, field):
    """Keep each video's own propagated field inside its detector region.

    Every source is propagated across the original whole canvas. This diagnostic
    suppresses fields arriving from the other video slots before detection.
    """
    masks = video_region_masks(stage, field)
    result = torch.zeros_like(field)
    for start in range(0, len(masks), 4):
        selected = masks[start:start + 4]
        sources = field[:, None] * selected[None]
        propagated = original(sources.flatten(0, 1)).reshape_as(sources)
        result += (propagated * selected[None]).sum(1)
    return result


def suppressed_forward(stage):
    original = stage.propagation.forward

    def forward(field):
        return own_video_propagation(stage, original, field)

    return forward


def prediction_rows(directory):
    with (directory / 'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if len(args.labels) != len(args.checkpoints) or len(set(args.labels)) != len(args.labels):
        raise ValueError('Use one unique label for each checkpoint')
    if any(Path(label).name != label or label in ('.', '..') for label in args.labels):
        raise ValueError('Labels must be simple directory names')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
    records = {}
    with torch.no_grad():
        for label, checkpoint in zip(args.labels, args.checkpoints):
            model, metadata = load_lightgen(checkpoint, args.device)
            if metadata['dataset']['manifest']['sha256'] != payload['identity']['manifest']['sha256']:
                raise ValueError('Dataset identity differs')
            state = tensor_fingerprint(model.state_dict())
            checkpoint_hash = sha256(checkpoint)
            directory = args.out / label
            joint = evaluate_condition(model, payload, args.device, directory / 'joint')
            with ExitStack() as stack:
                for attribute in STAGES.values():
                    stage = getattr(model, attribute)
                    stack.enter_context(patch.object(stage.propagation, 'forward', suppressed_forward(stage)))
                isolated = evaluate_condition(model, payload, args.device, directory / 'coupling_suppressed')
            normal_rows = prediction_rows(directory / 'joint')
            isolated_rows = prediction_rows(directory / 'coupling_suppressed')
            if [(row['sample_id'], row['target']) for row in normal_rows] != [
                    (row['sample_id'], row['target']) for row in isolated_rows]:
                raise ValueError('Diagnostic evaluation identities differ')
            difference = np.asarray([float(other['prediction']) - float(normal['prediction'])
                                     for normal, other in zip(normal_rows, isolated_rows)])
            record = {'checkpoint_sha256': checkpoint_hash, 'videos': len(difference),
                      'joint_metrics': joint, 'coupling_suppressed_metrics': isolated,
                      'delta_SRCC_suppressed_minus_joint': isolated['srcc'] - joint['srcc'],
                      'mean_absolute_prediction_change_MOS': float(np.abs(difference).mean()),
                      'root_mean_square_prediction_change_MOS': float(np.sqrt(np.mean(difference**2))),
                      'maximum_absolute_prediction_change_MOS': float(np.abs(difference).max())}
            if tensor_fingerprint(model.state_dict()) != state or sha256(checkpoint) != checkpoint_hash:
                raise ValueError('Checkpoint state changed during the diagnostic')
            records[label] = record
            write(args.out / 'comparison.json', records)
            print(json.dumps({label: record}), flush=True)
    write(args.out / 'protocol.json', {
        'source_commit': os.environ.get('LIGHTGEN_SOURCE_COMMIT',
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_paths.ROOT, text=True).strip()),
        'source_sha256': sha256(Path(__file__)), 'data_identity': payload['identity'],
        'role': 'post_training_cross_video_coupling_diagnostic_on_original_TEST_development',
        'formal_forward': 'Sixteen-video whole-canvas coherent propagation',
        'diagnostic_forward': 'Propagate each source over the whole canvas and retain its own detector-region field',
        'parameters_retrained': False, 'padding_excluded_from_metrics': True,
    })


if __name__ == '__main__':
    main()
