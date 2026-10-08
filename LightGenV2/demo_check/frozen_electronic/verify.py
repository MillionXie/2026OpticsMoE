"""CPU-only independent recomputation of predictions, selection and paired controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions(path, arrays):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r['sample_id'] for r in rows] == arrays['validation_ids'].tolist()
    y = np.array([int(r['label']) for r in rows])
    d = np.array([int(r['domain']) for r in rows])
    p = np.array([[float(r[f'p{i}']) for i in range(10)] for r in rows])
    assert np.array_equal(y, arrays['validation_labels'])
    assert np.array_equal(d, arrays['validation_domains'])
    assert np.array_equal(p.argmax(1), [int(r['prediction']) for r in rows])
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-6)
    return p


def verify_metrics(p, y, d, expected):
    pred = p.argmax(1)
    assert float((pred == y).mean()) == expected['accuracy']
    for domain in (0, 1):
        mask = d == domain
        assert float((pred[mask] == y[mask]).mean()) == expected['domain_accuracy'][str(domain)]
    matrix = np.zeros((10, 10), dtype=int); np.add.at(matrix, (y, pred), 1)
    assert matrix.tolist() == expected['confusion_matrix']
    loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()
    assert abs(loss - expected['loss']) < 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pure-run', type=Path, required=True)
    args = parser.parse_args(); root = args.run
    metadata = read(root/'metadata.json'); cfg = metadata['config']
    old_metadata = read(args.pure_run/'metadata.json')
    assert metadata['data_sha256'] == old_metadata['data_sha256'] == sha(args.pure_run/'pilot_data.npz')
    assert metadata['split_sha256'] == old_metadata['split_sha256']
    assert metadata['data_manifest_sha256'] == old_metadata['data_manifest_sha256'] == sha(args.pure_run/'dataset_manifest.json')
    manifest = read(args.pure_run/'dataset_manifest.json')
    groups = {split: {r['spatial_group'] for r in manifest['records'] if r['split'] == split} for split in ('train', 'validation')}
    assert not groups['train'] & groups['validation']
    arrays = np.load(args.pure_run/'pilot_data.npz', allow_pickle=False)
    y, d = arrays['validation_labels'], arrays['validation_domains']
    frozen = read(root/'frozen_electronic.json')
    assert frozen['checkpoint_sha256'] == sha(root/'electronic/best_checkpoint.pt')
    e = predictions(root/'electronic/validation_electronic.csv', arrays)
    controls = []; contributions = []; orders = []
    for summary in read(root/'results.json'):
        stage = summary['stage']; directory = root/stage
        assert sha(directory/'best_checkpoint.pt') == summary['checkpoint_sha256']
        history = read(directory/'history.json')
        assert [r['epoch'] for r in history] == list(range(1, 21))
        which = 'electronic' if stage == 'electronic' else 'fused'
        selected = max(history, key=lambda r: (r['validation'][which]['accuracy'], -r['validation'][which]['loss']))
        assert selected['epoch'] == summary['selected_epoch']
        assert selected['validation'] == summary['validation']
        for name in (['electronic'] if stage == 'electronic' else ['electronic', 'optical', 'fused']):
            p = predictions(directory/('validation_'+name+'.csv'), arrays)
            verify_metrics(p, y, d, summary['validation'][name])
            if name == 'electronic':
                assert np.array_equal(p, e)
        if stage != 'electronic':
            p = predictions(directory/'validation_fused.csv', arrays)
            o = predictions(directory/'validation_optical.csv', arrays)
            assert np.allclose(p, .5*e+.5*o, rtol=0, atol=6e-8)
            assert summary['frozen_electronic_sha256'] == frozen['tensors_sha256']
            assert all(r['frozen_electronic_verified'] for r in history)
            assert all(all(v > 0 and np.isfinite(v) for v in r['optical_first_batch_gradients'].values()) for r in history)
            old = read(args.pure_run/stage/'summary.json')
            assert summary['initial_parameters_sha256'] == old['initial_parameters_sha256']
            old_history = read(args.pure_run/stage/'history.json')
            assert [r['order_sha256'] for r in history] == [r['order_sha256'] for r in old_history]
            orders.append(tuple((r['order_sha256'], r['augmentation_seed']) for r in history))
            ec, fc = e.argmax(1) == y, p.argmax(1) == y
            for domain, mask in [('all', np.ones(len(y), dtype=bool)), ('RGB', d == 0), ('SAR', d == 1)]:
                contributions.append(dict(stage=stage, domain=domain, samples=int(mask.sum()),
                    electronic_correct=int((ec & mask).sum()), fused_correct=int((fc & mask).sum()),
                    corrected=int((~ec & fc & mask).sum()), spoiled=int((ec & ~fc & mask).sum()),
                    prediction_changed=int(((e.argmax(1) != p.argmax(1)) & mask).sum())))
        controls.append(dict(stage=stage, selected_epoch=summary['selected_epoch'], validation=summary['validation'][which]))
    assert len(set(orders)) == 1
    fairness = read(root/'fairness.json')
    assert fairness['electronic_state_and_predictions_identical'] and fairness['optical_order_and_augmentation_identical']
    result = dict(passed=True, verifier_sha256=sha(Path(__file__)), same_data_split=True,
                  train_validation_spatial_overlap=0, same_optical_initialization_as_pure=True,
                  frozen_electronic_predictions_identical=True, same_optical_training_order_and_augmentation=True,
                  normalized_half_fusion_verified=True, checkpoints_hash_verified=True,
                  results=controls, paired_contributions=contributions)
    (root/'independent_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
