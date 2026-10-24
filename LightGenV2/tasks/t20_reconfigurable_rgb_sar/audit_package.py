"""Read-only ZIP/data/prediction audit. Never executes bundled code or unpickles PT."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

ZIP_SHA256 = '7b9cd19058e2808334f57e23895923b99102e163ca633e3d890fcfed2d31f7de'
ROOT = 'eurosat_shared_frontend_all_tasks_20260916/'
SOURCE = ROOT + 'source/eurosat_shared_frontend_moe_d2nn/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(path):
    path = Path(path)
    if digest(path.read_bytes()) != ZIP_SHA256:
        raise ValueError('Unregistered ZIP identity')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate ZIP member')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe ZIP path')
        manifest = json.loads(z.read(ROOT + 'FILES_SHA256.json'))['files']
        for name, row in manifest.items():
            data = z.read(ROOT + name)
            if digest(data) != row['sha256'] or len(data) != row['bytes']:
                raise ValueError('Manifest mismatch: ' + name)
        with np.load(io.BytesIO(z.read(SOURCE + 'data/data.npz')), allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        if any(k.startswith('test') for k in arrays):
            raise ValueError('Unexpected new TEST arrays')
        perf = json.loads(z.read(ROOT + 'performance.json'))
        results = []
        for item in perf['models']:
            key = item['task']
            stem = {'MoE_AB': SOURCE + 'reference/dynamic_four/',
                    'D2NN_AB': SOURCE + 'reference/full_d2nn/',
                    'D2NN_A_only': ROOT + 'single_domain/runs/d2nn_A_only/',
                    'D2NN_B_only': ROOT + 'single_domain/runs/d2nn_B_only/'}[key]
            csv_name = stem + ('all_domain_validation_predictions.csv'
                              if key.endswith('_only') else 'validation_predictions.csv')
            rows = list(csv.DictReader(io.StringIO(z.read(csv_name).decode())))
            if [r['sample_id'] for r in rows] != arrays['validation_ids'].tolist():
                raise ValueError('Prediction ID/order mismatch')
            labels = np.array([int(r['label']) for r in rows])
            domains = np.array([int(r['domain']) for r in rows])
            if not (np.array_equal(labels, arrays['validation_labels']) and
                    np.array_equal(domains, arrays['validation_domains'])):
                raise ValueError('Prediction/data labels or domains mismatch')
            pred = np.array([int(r['prediction']) for r in rows])
            result = {'model': key, 'training_domains': item['optical_training_domains'],
                      'checkpoint_member': ROOT + item['best_checkpoint'],
                      'checkpoint_sha256': digest(z.read(ROOT + item['best_checkpoint']))}
            for domain, label in ((0, 'RGB'), (1, 'SAR')):
                ids = np.flatnonzero(domains == domain)[:1000]
                accuracy = float((pred[ids] == labels[ids]).mean())
                if len(ids) != 1000 or abs(accuracy - item[label.lower() + '_accuracy']) > 1e-12:
                    raise ValueError('Published score mismatch')
                result[label] = {'samples': len(ids), 'accuracy': accuracy,
                                 'ids_sha256': digest(json.dumps(arrays['validation_ids'][ids].tolist(), separators=(',', ':')).encode()),
                                 'class_counts': np.bincount(labels[ids], minlength=10).tolist()}
            results.append(result)
        return {'zip_sha256': ZIP_SHA256, 'manifest_files_verified': len(manifest),
                'scope': 'stored validation predictions, not fresh inference or hardware',
                'independent_test_images_bundled': False,
                'train_samples': len(arrays['train_labels']),
                'validation_samples': len(arrays['validation_labels']),
                'models': results}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    report = audit(args.zip)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as f:
            f.write(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
