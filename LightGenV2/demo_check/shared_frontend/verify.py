"""Independent CPU audit of shared-frontend records and saved predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--pure-run',type=Path,required=True);p.add_argument('--electronic-run',type=Path,required=True)
    args=p.parse_args();root=args.run
    metadata=read(root/'metadata.json');old=read(args.pure_run/'metadata.json');electronic=read(args.electronic_run/'metadata.json')
    assert read(root/'status.json')['state']=='complete'
    assert metadata['data_sha256']==old['data_sha256']==electronic['data_sha256']==sha(args.pure_run/'pilot_data.npz')
    assert metadata['split_sha256']==old['split_sha256']==electronic['split_sha256']
    assert metadata['data_manifest_sha256']==old['data_manifest_sha256']==sha(args.pure_run/'dataset_manifest.json')
    manifest=read(args.pure_run/'dataset_manifest.json')
    groups={split:{r['spatial_group'] for r in manifest['records'] if r['split']==split} for split in ['train','validation']}
    assert not groups['train']&groups['validation']
    arrays=np.load(args.pure_run/'pilot_data.npz',allow_pickle=False)
    frontend=read(root/'frontend/summary.json')
    assert sha(root/'frontend/best_checkpoint.pt')==frontend['checkpoint_sha256']
    assert frontend['source_checkpoint_sha256']==metadata['config']['electronic_checkpoint_sha256']==sha(args.electronic_run/'electronic/best_checkpoint.pt')
    assert frontend['frozen_parameters']==93472
    old_frozen=read(args.electronic_run/'frozen_electronic.json')['tensors_sha256']
    assert frontend['frozen_tensors_sha256']=={k:v for k,v in old_frozen.items() if not k.startswith('head.')}
    reports=[];histories=[];probabilities={}
    for summary in read(root/'results.json'):
        architecture=summary['architecture'];dest=root/architecture
        assert sha(dest/'best_checkpoint.pt')==summary['checkpoint_sha256']
        assert summary['frontend_tensors_sha256']==frontend['frozen_tensors_sha256']
        with (dest/'validation_predictions.csv').open() as f:rows=list(csv.DictReader(f))
        assert [r['sample_id'] for r in rows]==arrays['validation_ids'].tolist()
        y=np.array([int(r['label']) for r in rows]);d=np.array([int(r['domain']) for r in rows])
        prob=np.array([[float(r[f'p{i}']) for i in range(10)] for r in rows]);pred=prob.argmax(1)
        assert np.array_equal(y,arrays['validation_labels']) and np.array_equal(d,arrays['validation_domains'])
        assert np.array_equal(pred,[int(r['prediction']) for r in rows])
        assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(1),1,atol=1e-6)
        metrics=summary['validation'];assert metrics['accuracy']==float((pred==y).mean())
        for domain in [0,1]:assert metrics['domain_accuracy'][str(domain)]==float((pred[d==domain]==y[d==domain]).mean())
        matrix=np.zeros((10,10),dtype=int);np.add.at(matrix,(y,pred),1)
        assert metrics['confusion_matrix']==matrix.tolist()
        assert abs(metrics['loss']+np.log(np.maximum(prob[np.arange(len(y)),y],1e-12)).mean())<1e-6
        history=read(dest/'history.json');assert [r['epoch'] for r in history]==list(range(1,21))
        selected=max(history,key=lambda r:(r['validation']['accuracy'],-r['validation']['loss']))
        assert selected['epoch']==summary['selected_epoch'] and selected['validation']==metrics
        assert all(r['frontend_frozen_verified'] for r in history)
        assert all(all(np.isfinite(v) and v>0 for v in r['first_batch_gradients'].values()) for r in history)
        pure=read(args.pure_run/architecture/'summary.json');pure_history=read(args.pure_run/architecture/'history.json')
        assert summary['initial_parameters_sha256']==pure['initial_parameters_sha256']
        assert all(v!=summary['final_parameters_sha256'][k] for k,v in summary['initial_parameters_sha256'].items())
        assert [r['order_sha256'] for r in history]==[r['order_sha256'] for r in pure_history]
        histories.append(tuple((r['order_sha256'],r['augmentation_seed'],r['input_feature_sha256']) for r in history))
        probabilities[architecture]=prob
        reports.append(dict(architecture=architecture,selected_epoch=summary['selected_epoch'],accuracy=metrics['accuracy'],
                            domain_accuracy=metrics['domain_accuracy'],last_epoch_accuracy=history[-1]['validation']['accuracy'],
                            checkpoint_sha256=summary['checkpoint_sha256']))
    assert len(set(histories))==1
    assert all(read(root/'fairness.json')[key] for key in ['frontend_states_unchanged','all_epoch_input_features_bitwise_identical','identical_order_and_augmentation'])
    mc=probabilities['dynamic_four'].argmax(1)==y;dc=probabilities['full_d2nn'].argmax(1)==y
    paired={name:dict(n=int(mask.sum()),moe_only_correct=int((mc&~dc&mask).sum()),d2nn_only_correct=int((dc&~mc&mask).sum()),
                     both_correct=int((dc&mc&mask).sum()),both_wrong=int((~dc&~mc&mask).sum()))
            for name,mask in [('all',np.ones(len(y),bool)),('RGB',d==0),('SAR',d==1)]}
    result=dict(passed=True,verifier_sha256=sha(Path(__file__)),same_frontend_as_pretraining=True,
                same_data_split=True,train_validation_spatial_overlap=0,identical_features_all_20_epochs=True,
                phase_initialization_matches_pure=True,results=reports,paired_comparison=paired)
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
