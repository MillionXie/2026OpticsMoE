"""Measure actual incident-light coverage of each global phase region."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import _paths
    from .data import load_inputs, groups, batch, sha256
    from .run import write
    from .spatial_locality import STAGES
    from ..model import load_lightgen
else:
    import _paths
    from data import load_inputs, groups, batch, sha256
    from run import write
    from spatial_locality import STAGES
    from model import load_lightgen


def field_statistics(fields):
    """Return effective area, nonzero area and incident energy per video."""
    intensity = fields.flatten(2).double().square()
    energy = intensity.sum(-1)
    effective = energy.square() / intensity.square().sum(-1).clamp_min(1e-20)
    positive = (intensity > 0).sum(-1)
    return torch.stack((100 * effective / (111 * 111),
                        100 * positive / (111 * 111), energy), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if len(args.labels) != len(args.checkpoints) or len(set(args.labels)) != len(args.labels):
        raise ValueError('Provide one unique label per checkpoint')
    if any(Path(label).name != label or label in ('.', '..') for label in args.labels):
        raise ValueError('Labels must be simple filenames')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
    indices, valid = groups(payload, 'test')
    reports = {}
    with torch.no_grad():
        for label, path in zip(args.labels, args.checkpoints):
            model, metadata = load_lightgen(path, args.device)
            if metadata['dataset']['manifest']['sha256'] != payload['identity']['manifest']['sha256']:
                raise ValueError('Dataset identity differs')
            captured = {}
            handles = []

            def capture(name):
                def hook(stage, arguments):
                    captured[name] = field_statistics(stage.fields(arguments[0])).cpu()
                return hook

            for branch, attribute in STAGES.items():
                handles.append(getattr(model, attribute).register_forward_pre_hook(capture(branch)))
            rows = []
            try:
                for start in range(0, len(indices), 4):
                    item = batch(payload, indices[start:start + 4], valid[start:start + 4], args.device)
                    result = model(item['vision'])
                    keep = item['valid'].cpu()
                    ids = item['indices'][keep].tolist()
                    prediction = result['prediction'][item['valid']].cpu().tolist()
                    values = {branch: captured[branch][keep].tolist() for branch in STAGES}
                    for index, sample in enumerate(ids):
                        row = {'sample_id': payload['ids'][sample], 'prediction': prediction[index]}
                        for branch in STAGES:
                            for key, value in zip(('effective_area_percent', 'positive_area_percent', 'incident_energy'), values[branch][index]):
                                row[branch + '_' + key] = value
                        rows.append(row)
            finally:
                for handle in handles:
                    handle.remove()
            if len(rows) != 558 or len({row['sample_id'] for row in rows}) != 558:
                raise ValueError('Expected all 558 unique evaluation videos')
            with (args.out / (label + '.csv')).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            reports[label] = {'checkpoint_sha256': sha256(path), 'videos': len(rows)}
            for branch in STAGES:
                reports[label][branch] = {}
                for key in ('effective_area_percent', 'positive_area_percent', 'incident_energy'):
                    values = np.asarray([row[branch + '_' + key] for row in rows])
                    reports[label][branch][key] = {'mean': float(values.mean()),
                        'median': float(np.median(values)), 'minimum': float(values.min()),
                        'maximum': float(values.max())}
            write(args.out / 'summary.json', reports)
    write(args.out / 'definition.json', {
        'effective_area_pixels': '(sum I)^2 / sum(I^2)',
        'area_denominator_pixels': 111 * 111,
        'positive_area_pixels': 'Count of incident pixels with I > 0',
        'incident_energy': 'Sum of incident squared amplitude before phase modulation',
        'data_identity': payload['identity'], 'source_sha256': sha256(Path(__file__)),
    })
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
