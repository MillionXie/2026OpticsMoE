"""CPU audit of continuation safeguards, selection, metrics and paired sampling."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);a=p.parse_args();meta=read(a.run/'metadata.json');profile=meta['profile']
    assert read(a.run/'status.json')['state']=='complete'
    assert sha(a.data)==meta['data_sha256']==read(a.source/'metadata.json')['data_sha256']
    assert sha(a.run/'frontend/best_checkpoint.pt')==sha(a.source/'frontend/best_checkpoint.pt')==meta['frontend_checkpoint_sha256']
    arrays=np.load(a.data,allow_pickle=False);checks=[];histories=[]
    for summary in read(a.run/'results.json'):
        name=summary['architecture'];directory=a.run/name;history=read(directory/'history.json');old=read(a.source/name/'summary.json')
        old_history=read(a.source/name/'history.json');resume=read(directory/'resume.json')
        assert resume['epoch']==20 and resume['optimizer_steps']==[3760]
        assert resume['validation']==old_history[-1]['validation']
        assert summary['resume_checkpoint_sha256']==resume['source_checkpoint_sha256']
        assert [x['epoch'] for x in history]==list(range(21,summary['stop_epoch']+1))
        assert summary['epochs_completed']==len(history)<=40
        selected=old['selected_epoch'];best=(old['validation']['accuracy'],-old['validation']['loss']);best_metrics=old['validation'];best_train=old['train_unaugmented']
        minimum=min(x['validation']['loss'] for x in old_history);stale=0
        for row in history:
            v=row['validation'];t=row['train_unaugmented']
            assert abs(row['accuracy_gap']-(t['accuracy']-v['accuracy']))<1e-12
            assert abs(row['nll_gap']-(v['loss']-t['loss']))<1e-12
            assert row['frontend_frozen_verified']
            expected_lr=.0001+.0009*.5*(1+math.cos(math.pi*(row['epoch']-21)/39))
            assert abs(row['learning_rate']-expected_lr)<1e-12
            eligible=v['loss']<=old['validation']['loss'];key=(v['accuracy'],-v['loss'])
            improved=eligible and key>best
            assert eligible==row['eligible_by_nll'] and improved==row['selected_improvement']
            if improved:selected=row['epoch'];best=key;best_metrics=v;best_train=t
            if v['loss']<minimum-profile['early_stopping_min_nll_improvement']:minimum=v['loss'];stale=0
            else:stale+=1
            assert stale==row['nll_stale_epochs']
            if stale>=12:assert row==history[-1]
        assert selected==summary['selected_epoch'] and best_metrics==summary['validation'] and best_train==summary['train_unaugmented']
        assert summary['stop_reason']==('validation_nll_early_stopping' if stale>=12 else 'maximum_epoch')
        if stale<12:assert summary['stop_epoch']==60
        assert sha(directory/'best_checkpoint.pt')==summary['checkpoint_sha256']
        with (directory/'validation_predictions.csv').open() as f:rows=list(csv.DictReader(f))
        assert [r['sample_id'] for r in rows]==arrays['validation_ids'].tolist()
        y=np.array([int(r['label']) for r in rows]);d=np.array([int(r['domain']) for r in rows]);prob=np.array([[float(r[f'p{i}']) for i in range(10)] for r in rows]);pred=prob.argmax(1)
        assert np.array_equal(y,arrays['validation_labels']) and np.array_equal(d,arrays['validation_domains'])
        assert np.array_equal(pred,[int(r['prediction']) for r in rows])
        assert np.allclose(prob.sum(1),1,atol=1e-6) and (prob>=0).all()
        assert float((pred==y).mean())==best_metrics['accuracy']
        for domain in [0,1]:assert float((pred[d==domain]==y[d==domain]).mean())==best_metrics['domain_accuracy'][str(domain)]
        assert abs(-np.log(np.maximum(prob[np.arange(len(y)),y],1e-12)).mean()-best_metrics['loss'])<1e-6
        matrix=np.zeros((10,10),dtype=int);np.add.at(matrix,(y,pred),1);assert matrix.tolist()==best_metrics['confusion_matrix']
        histories.append({r['epoch']:(r['order_sha256'],r['augmentation_seed'],r['input_feature_sha256']) for r in history})
        checks.append(dict(architecture=name,selected_epoch=selected,stop_epoch=summary['stop_epoch'],accuracy=best_metrics['accuracy'],nll=best_metrics['loss'],train_validation_accuracy_gap=summary['accuracy_gap']))
    common=set(histories[0])&set(histories[1]);assert all(histories[0][e]==histories[1][e] for e in common)
    result=dict(passed=True,verifier_sha256=sha(Path(__file__)),early_stopping_recomputed=True,checkpoint_selection_recomputed=True,
                same_frozen_frontend=True,matching_features_epochs=sorted(common),results=checks)
    (a.run/'independent_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
