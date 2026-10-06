"""Evaluate validation-selected phase-only checkpoints on a fixed spatial holdout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from models import PhaseOnly, objective


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def evaluate(model, data, batch=32):
    images, labels, domains = data
    probabilities = []
    for start in range(0, len(labels), batch):
        result = model(images[start:start+batch])
        probabilities.append(result['probabilities'].cpu())
    probabilities = torch.cat(probabilities).numpy()
    labels_np = labels.cpu().numpy()
    domains_np = domains.cpu().numpy()
    prediction = probabilities.argmax(1)
    return {
        'accuracy': float((prediction == labels_np).mean()),
        'domain_accuracy': {str(d): float((prediction[domains_np == d] == labels_np[domains_np == d]).mean()) for d in (0, 1)},
        'nll': float(-np.log(np.maximum(probabilities[np.arange(len(labels_np)), labels_np], 1e-30)).mean()),
    }, probabilities


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    assert json.loads((a.run/'status.json').read_text())['state'] == 'complete'
    manifest = json.loads(a.data.with_name('manifest.json').read_text())
    assert sha(a.data) == manifest['data_sha256']
    with np.load(a.data, allow_pickle=False) as z:
        assert set(z.files) == {'test_images', 'test_labels', 'test_domains', 'test_ids'}
        arrays = {key: z[key].copy() for key in z.files}
    assert len(arrays['test_labels']) == 2000
    cfg = json.loads((a.run/'metadata.json').read_text())['config']
    selected = []
    for architecture in cfg['architectures']:
        checkpoint = a.run/architecture/'best_checkpoint.pt'
        summary = json.loads((a.run/architecture/'summary.json').read_text())
        assert sha(checkpoint) == summary['checkpoint_sha256']
        selected.append({'architecture': architecture, 'checkpoint': str(checkpoint), 'sha256': sha(checkpoint),
                         'epoch': summary['selected_epoch'], 'validation': summary['validation']})
    (a.out/'selection_lock.json').write_text(json.dumps({'selection': selected, 'rule': 'validation-only checkpoint selection'}, indent=2))
    torch.set_num_threads(4)
    data = tuple(torch.from_numpy(arrays['test_'+key]).cuda() for key in ['images', 'labels', 'domains'])
    results = []
    for entry in selected:
        model = PhaseOnly(entry['architecture'], cfg).cuda()
        model.load_state_dict(torch.load(entry['checkpoint'], map_location='cpu', weights_only=False)['model'])
        metrics, probabilities = evaluate(model, data)
        with (a.out/(entry['architecture']+'_predictions.csv')).open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'domain', 'label', 'prediction'] + [f'p{i}' for i in range(10)])
            writer.writerows([str(i), int(d), int(y), int(p.argmax()), *p.tolist()]
                             for i, d, y, p in zip(arrays['test_ids'], arrays['test_domains'], arrays['test_labels'], probabilities))
        results.append({**entry, 'test': metrics})
        print(json.dumps(results[-1]), flush=True)
        del model
        torch.cuda.empty_cache()
    (a.out/'results.json').write_text(json.dumps(results, indent=2))
    (a.out/'status.json').write_text(json.dumps({'state': 'complete', 'test_used_for_selection': False}, indent=2))


if __name__ == '__main__':
    main()
