"""CPU-only reference evaluator from the SHA-registered ZIP; no SDK or training."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .audit_package import audit, SOURCE, ROOT
from .reference.models import SharedFrontend, FrontendOptics


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--model', choices=('MoE_AB', 'D2NN_A_only', 'D2NN_AB'), required=True)
    p.add_argument('--per-domain', type=int, default=1000)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.per_domain <= 1000:
        p.error('per-domain must be 1..1000; small runs are smoke only')
    report = audit(args.zip)
    identity = next(r for r in report['models'] if r['model'] == args.model)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with zipfile.ZipFile(args.zip) as z:
        # weights_only avoids executing arbitrary pickle globals.
        front = torch.load(io.BytesIO(z.read(SOURCE + 'checkpoints/frontend/best_checkpoint.pt')),
                           map_location='cpu', weights_only=True)
        state = torch.load(io.BytesIO(z.read(identity['checkpoint_member'])),
                           map_location='cpu', weights_only=True)
        config = json.loads(z.read(SOURCE + 'optical_config.json'))
        with np.load(io.BytesIO(z.read(SOURCE + 'data/data.npz')), allow_pickle=False) as a:
            domains = a['validation_domains']
            indices = np.sort(np.concatenate([np.flatnonzero(domains == d)[:args.per_domain] for d in (0, 1)]))
            images, labels, ids = (a['validation_' + k][indices] for k in ('images', 'labels', 'ids'))
        csv_member = (SOURCE+'reference/dynamic_four/validation_predictions.csv' if args.model=='MoE_AB'
                      else SOURCE+'reference/full_d2nn/validation_predictions.csv' if args.model=='D2NN_AB'
                      else ROOT+'single_domain/runs/d2nn_A_only/all_domain_validation_predictions.csv')
        stored=list(csv.DictReader(io.StringIO(z.read(csv_member).decode())))
        expected=np.array([[float(stored[i][f'p{k}']) for k in range(10)] for i in indices],dtype=np.float32)
    architecture = 'dynamic_four' if args.model == 'MoE_AB' else 'full_d2nn'
    model = FrontendOptics(SharedFrontend(front['model']), architecture, config).eval()
    model.optical.load_state_dict(state['model'], strict=True)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(images), 4):
            outputs.append(model(torch.from_numpy(images[start:start+4]))['probabilities'].numpy())
    probabilities = np.concatenate(outputs)
    result = {'scope': 'fresh CPU simulation, validation not independent TEST',
              'identity': identity, 'per_domain': args.per_domain,
              'accuracy': {str(d): float((probabilities.argmax(1)[domains[indices] == d] == labels[domains[indices] == d]).mean()) for d in (0, 1)},
              'hardware_measurement': False}
    result['stored_prediction_probability_maxerror']=float(np.abs(expected-probabilities).max())
    result['stored_prediction_argmax_agreement']=float((expected.argmax(1)==probabilities.argmax(1)).mean())
    np.savez(args.out/'predictions.npz', probabilities=probabilities, labels=labels, ids=ids, source_indices=indices)
    (args.out/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
