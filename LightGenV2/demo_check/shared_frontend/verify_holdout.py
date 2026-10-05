"""Independent test CSV/pixel audit and paired spatial-group bootstrap."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--training-data',type=Path,required=True);parser.add_argument('--training-manifest',type=Path,required=True)
    args=parser.parse_args();root=args.run;manifest=read(root/'test_manifest.json');metadata=read(root/'metadata.json')
    assert sha(root/'test_data.npz')==manifest['data_sha256']==metadata['data_sha256']
    arrays=np.load(root/'test_data.npz',allow_pickle=False);training=np.load(args.training_data,allow_pickle=False)
    assert not set(arrays['test_ids'])&(set(training['train_ids'])|set(training['validation_ids']))
    training_manifest=read(args.training_manifest)
    assert not {r['spatial_group'] for r in manifest['records']}&{r['spatial_group'] for r in training_manifest['records']}
    for row,image,label,domain,sid in zip(manifest['records'],arrays['test_images'],arrays['test_labels'],arrays['test_domains'],arrays['test_ids']):
        assert row['pixel_sha256']==hashlib.sha256(image.tobytes()).hexdigest()
        assert row['label']==label and row['domain']==domain and sid==row['pair_id']+':'+str(domain)
    assert len(arrays['test_ids'])==2000 and len(set(arrays['test_ids']))==2000
    lock=read(root/'selection_lock.json');locked={(r['label'],r['architecture']):r for r in lock['models']}
    correct={};checks=[];y=arrays['test_labels'];d=arrays['test_domains']
    for result in read(root/'results.json'):
        key=(result['label'],result['architecture']);assert all(result[k]==v for k,v in locked[key].items())
        path=root/('_'.join(key)+'_predictions.csv')
        with path.open() as f:rows=list(csv.DictReader(f))
        assert [r['sample_id'] for r in rows]==arrays['test_ids'].tolist()
        assert np.array_equal(y,[int(r['label']) for r in rows]) and np.array_equal(d,[int(r['domain']) for r in rows])
        p=np.array([[float(r[f'p{i}']) for i in range(10)] for r in rows]);pred=p.argmax(1)
        assert np.array_equal(pred,[int(r['prediction']) for r in rows]) and np.allclose(p.sum(1),1,atol=1e-6)
        assert np.isfinite(p).all() and (p>=0).all();c=pred==y;metrics=result['test']
        assert float(c.mean())==metrics['accuracy']
        for domain in [0,1]:assert float(c[d==domain].mean())==metrics['domain_accuracy'][str(domain)]
        assert abs(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()-metrics['loss'])<1e-6
        matrix=np.zeros((10,10),dtype=int);np.add.at(matrix,(y,pred),1);assert matrix.tolist()==metrics['confusion_matrix']
        correct[key]=c;checks.append(dict(label=key[0],architecture=key[1],accuracy=metrics['accuracy'],csv_sha256=sha(path)))
    groups=np.array([r['spatial_group'] for r in manifest['records']]);unique,inverse=np.unique(groups,return_inverse=True)
    counts=np.bincount(inverse);rng=np.random.default_rng(20260916)
    samples=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=10000)
    comparisons=[]
    for architecture in ['dynamic_four','full_d2nn']:
        old=correct[('original',architecture)];new=correct[('continued',architecture)]
        delta=new.astype(float)-old.astype(float);sums=np.bincount(inverse,weights=delta)
        draws=100*(samples@sums)/(samples@counts)
        comparisons.append(dict(architecture=architecture,delta_percentage_points=float(100*delta.mean()),
            corrected=int((~old&new).sum()),spoiled=int((old&~new).sum()),
            spatial_group_bootstrap_95_percentile_interval_pp=np.quantile(draws,[.025,.975]).tolist()))
    report=dict(passed=True,verifier_sha256=sha(Path(__file__)),test_pixels_verified=2000,no_training_validation_id_or_spatial_overlap=True,
                selected_checkpoints_match_pretest_lock=True,results=checks,paired_comparisons=comparisons,
                bootstrap=dict(seed=20260916,repetitions=10000,resampling_unit='original spatial_group',groups=len(unique)))
    (root/'independent_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
