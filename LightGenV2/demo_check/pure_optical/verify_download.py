"""Independently verify downloaded pilot data, predictions and selection records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    root = parser.parse_args().run
    manifest = json.loads((root/'dataset_manifest.json').read_text())
    metadata = json.loads((root/'metadata.json').read_text())
    assert sha(root/'pilot_data.npz') == manifest['data_sha256'] == metadata['data_sha256']
    assert sha(root/'dataset_manifest.json') == metadata['data_manifest_sha256']
    arrays = np.load(root/'pilot_data.npz', allow_pickle=False)
    assert not any(k.startswith('test') for k in arrays.files)
    groups = {}
    for split, count in [('train', 6000), ('validation', 2000)]:
        records = [r for r in manifest['records'] if r['split'] == split]
        images, labels, domains, ids = [arrays[split+'_'+key] for key in ['images','labels','domains','ids']]
        assert images.shape == (count,56,56,3) and len(set(ids)) == count
        assert len(records) == count
        for domain in (0,1):
            assert np.bincount(labels[domains == domain], minlength=10).tolist() == [count//20]*10
        for row, image, label, domain, sid in zip(records,images,labels,domains,ids):
            assert hashlib.sha256(image.tobytes()).hexdigest() == row['pixel_sha256']
            assert label == row['label'] and domain == row['domain']
            assert sid == row['pair_id']+':'+str(domain)
        groups[split] = {r['spatial_group'] for r in records}
    assert not groups['train'] & groups['validation']
    results = json.loads((root/'results.json').read_text())
    checked = []; orders = []
    for summary in results:
        directory = root/summary['architecture']
        assert sha(directory/'best_checkpoint.pt') == summary['checkpoint_sha256']
        with (directory/'validation_predictions.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert [r['sample_id'] for r in rows] == arrays['validation_ids'].tolist()
        label = np.array([int(r['label']) for r in rows])
        domain = np.array([int(r['domain']) for r in rows])
        probabilities = np.array([[float(r[f'p{i}']) for i in range(10)] for r in rows])
        prediction = probabilities.argmax(1)
        assert np.array_equal(label, arrays['validation_labels'])
        assert np.array_equal(domain, arrays['validation_domains'])
        assert np.array_equal(prediction, [int(r['prediction']) for r in rows])
        assert np.all(probabilities >= 0) and np.allclose(probabilities.sum(1),1,atol=1e-6)
        matrix = np.zeros((10,10),dtype=int)
        np.add.at(matrix,(label,prediction),1)
        metrics = summary['validation']
        assert matrix.tolist() == metrics['confusion_matrix']
        assert float((label == prediction).mean()) == metrics['accuracy']
        for d in (0,1):
            assert float((label[domain==d] == prediction[domain==d]).mean()) == metrics['domain_accuracy'][str(d)]
        history = json.loads((directory/'history.json').read_text())
        assert [r['epoch'] for r in history] == list(range(1,21))
        chosen = max(history,key=lambda r:(r['validation']['accuracy'],-r['validation']['loss']))
        assert chosen['epoch'] == summary['selected_epoch']
        assert chosen['validation'] == metrics
        orders.append(tuple(r['order_sha256'] for r in history))
        checked.append(dict(architecture=summary['architecture'],accuracy=metrics['accuracy'],
                            selected_epoch=summary['selected_epoch'],checkpoint_sha256=summary['checkpoint_sha256']))
    assert len(set(orders)) == 1
    initial = [r['initial_parameters_sha256'] for r in results]
    assert {k:v for k,v in initial[0].items() if k!='router_phase'} == initial[1]
    assert len({r['global_phase'] for r in initial}) == 1
    report = dict(passed=True,data_sha256=metadata['data_sha256'],train_validation_spatial_overlap=0,
                  pixel_hashes_verified=8000,validation_predictions_per_model=2000,
                  epochs_per_model=20,identical_training_orders=True,identical_main_initialization=True,
                  results=checked,verifier_sha256=sha(Path(__file__)))
    (root/'independent_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
