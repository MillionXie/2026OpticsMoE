"""Test the contribution of phase texture outside the fixed incident support."""
from contextlib import ExitStack
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

if __package__:
    from . import _paths
    from .analyze_global_ablation import evaluate_condition
    from .checkpoint_utils import tensor_fingerprint
    from .data import load_inputs, sha256
    from .phase_regularization import illumination_support
    from .run import write
    from .spatial_locality import STAGES
    from ..model import load_lightgen
else:
    import _paths
    from analyze_global_ablation import evaluate_condition
    from checkpoint_utils import tensor_fingerprint
    from data import load_inputs, sha256
    from phase_regularization import illumination_support
    from run import write
    from spatial_locality import STAGES
    from model import load_lightgen


def constant_unilluminated_modulation(stage):
    original = stage.modulation
    support = illumination_support(stage.frame_level).to(stage.raw_global_phase.device)

    def modulation():
        values = original()
        return torch.where(support, values, torch.ones_like(values))
    return modulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, metadata = load_lightgen(args.weights, args.device)
    payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
    if payload['identity']['manifest']['sha256'] != metadata['dataset']['manifest']['sha256']:
        raise ValueError('Dataset identity differs')
    fingerprint = tensor_fingerprint(model.state_dict())
    with torch.no_grad():
        original = evaluate_condition(model, payload, args.device, args.out / 'original')
        with ExitStack() as stack:
            for attribute in STAGES.values():
                stage = getattr(model, attribute)
                stack.enter_context(patch.object(stage, 'modulation', constant_unilluminated_modulation(stage)))
            modified = evaluate_condition(model, payload, args.device, args.out / 'unilluminated_constant')
    rows = []
    for name in ('original', 'unilluminated_constant'):
        with (args.out / name / 'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows.append(list(csv.DictReader(stream)))
    if [(row['sample_id'], row['target']) for row in rows[0]] != [
            (row['sample_id'], row['target']) for row in rows[1]]:
        raise ValueError('Evaluation identities differ')
    difference = np.asarray([float(second['prediction']) - float(first['prediction'])
                            for first, second in zip(*rows)])
    if tensor_fingerprint(model.state_dict()) != fingerprint:
        raise ValueError('Diagnostic changed model parameters')
    result = {'checkpoint_sha256': sha256(args.weights), 'videos': len(difference),
              'original_SRCC': original['srcc'], 'modified_SRCC': modified['srcc'],
              'maximum_absolute_MOS_difference': float(np.abs(difference).max()),
              'predictions_exactly_equal': bool(np.array_equal(difference, np.zeros_like(difference))),
              'definition': 'Replace modulation outside fixed incident-light support by unit modulation in both stages',
              'data_identity': payload['identity'], 'script_sha256': sha256(Path(__file__))}
    write(args.out / 'comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
