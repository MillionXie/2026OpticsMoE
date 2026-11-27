"""Audit the independently deployable MLP model and its selected checkpoint."""
from pathlib import Path
import argparse
import copy
import json

import numpy as np
import torch
from torch import nn

from LightGenV2.tasks.t22_public_temporal_mlp.tools.data import load_inputs
from .checkpoints import export_checkpoint, sha256, public_model, load_lightgen2
from .evaluation import evaluate, feature_noise, qualifies

GlobalMLP=public_model.GlobalMLP


class PairAverage(nn.Module):
    def forward(self, tokens):
        return tokens.reshape(*tokens.shape[:-1], 96, 2).mean(-1)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def compare_records(reference, observed):
    lookup = {row['sample_id']: row['predicted_MOS'] for row in observed}
    if len(lookup) != 558 or set(lookup) != {row['sample_id'] for row in reference}:
        raise ValueError('Evaluation identities differ')
    difference = [abs(row['predicted_MOS'] - lookup[row['sample_id']]) for row in reference]
    return float(max(difference))


def validate(args):
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
    model, _ = load_lightgen2(args.checkpoint, args.device)
    clean, reference = evaluate(model, payload, args.device, directory=args.out/'clean')
    write(args.out/'progress.json', dict(state='slot_permutations', clean_srcc=clean['srcc']))
    permutations = []
    for shift in range(16):
        metrics, records = evaluate(model, payload, args.device, slot_shift=shift)
        permutations.append(dict(shift=shift, srcc=metrics['srcc'],
                                 maximum_MOS_difference=compare_records(reference, records)))
    write(args.out/'slot_permutations.json', permutations)

    replacements = [
        ('vision_identity', 'vision', nn.Identity()),
        ('language_pair_average', 'language', PairAverage()),
    ]
    ablations = []
    for name, branch, replacement in replacements:
        candidate = copy.deepcopy(model)
        setattr(candidate, branch+'_global', replacement.to(args.device))
        metrics, _ = evaluate(candidate, payload, args.device)
        ablations.append(dict(control=name, srcc=metrics['srcc'],
                              clean_minus_control=clean['srcc']-metrics['srcc']))
        del candidate
    for branch in ('vision', 'language'):
        for seed in (20261117, 20261118, 20261119):
            candidate = copy.deepcopy(model)
            torch.manual_seed(seed)
            count = 49 if branch == 'vision' else 42
            width = 192 if branch == 'vision' else 96
            replacement = GlobalMLP(model.cfg, count, width).to(args.device)
            setattr(candidate, branch+'_global', replacement)
            metrics, _ = evaluate(candidate, payload, args.device)
            ablations.append(dict(control=branch+'_random', seed=seed, srcc=metrics['srcc'],
                                  clean_minus_control=clean['srcc']-metrics['srcc']))
            del candidate
    write(args.out/'global_controls.json', ablations)

    training = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['training']
    noise_reference = training['protocol']['feature_noise_reference']
    noise = []
    for fraction in (.05, .10):
        trials = []
        for seed in range(201, 211):
            with feature_noise(model, fraction, noise_reference, seed):
                metrics, _ = evaluate(model, payload, args.device)
            trials.append(dict(seed=seed, srcc=metrics['srcc']))
        scores = [trial['srcc'] for trial in trials]
        noise.append(dict(fraction=fraction, mean_srcc=float(np.mean(scores)),
                          minimum_srcc=float(min(scores)), maximum_srcc=float(max(scores)),
                          trials=trials))
    write(args.out/'feature_noise.json', dict(
        definition='Independent additive Gaussian feature noise at both global MLP outputs; '
                   'standard deviation is fraction times the fixed TRAIN teacher output RMS',
        reference_RMS=noise_reference, results=noise))

    exported_sha = export_checkpoint(args.checkpoint, args.export)
    exported, _ = load_lightgen2(args.export, args.device)
    exported_metrics, exported_records = evaluate(exported, payload, args.device)
    export_difference = compare_records(reference, exported_records)
    checks = dict(clean_and_routing=qualifies(clean, .795, .799),
                  slot_invariance=max(row['maximum_MOS_difference'] for row in permutations) < 1e-3,
                  global_contributions=all(row['clean_minus_control'] > 0 for row in ablations),
                  feature_noise_5pct=noise[0]['minimum_srcc'] >= .790,
                  feature_noise_10pct=noise[1]['minimum_srcc'] >= .780,
                  export_equal=export_difference == 0)
    result = dict(checkpoint_sha256=sha256(args.checkpoint), exported_sha256=exported_sha,
                  clean=clean, checks=checks, export_maximum_MOS_difference=export_difference,
                  exported_srcc=exported_metrics['srcc'])
    write(args.out/'validation.json', result)
    write(args.out/'progress.json', dict(state='completed', checks=checks))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    validate(parser.parse_args())


if __name__ == '__main__':
    main()
