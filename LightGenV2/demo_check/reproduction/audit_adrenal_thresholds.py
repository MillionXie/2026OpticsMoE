"""Retrospective fixed-weight diagnosis; choose thresholds only from validation."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    archive=Path(__file__).resolve().parents[1]/'adrenal_softsign_code_export_20260915_145336/code'
    sys.path.insert(0,str(archive))
    import torch
    import run_experiment as r
    from calibration import choose_thresholds,full_metrics
    a.out.mkdir(parents=True,exist_ok=False)
    r.EXP['data_npz']=str(a.data.resolve());r.setup();r.setseed(17)
    original_lock=r.read(a.source_run/'subset_test_lock.json')
    metadata=dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=archive,text=True).strip(),
                  script_sha256=r.sha(__file__),data_sha256=r.sha(a.data),source_run=str(a.source_run),
                  source_lock_sha256=r.sha(a.source_run/'subset_test_lock.json'),
                  scope='retrospective fixed-weight re-evaluation; original test results already known; no retraining',
                  threshold_policies=['fixed_0.5','val_accuracy','val_balanced'],
                  selection='original validation-selected checkpoint; thresholds chosen from val only',
                  torch=torch.__version__,python=sys.version,gpu=torch.cuda.get_device_name(),
                  environment=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    r.save(a.out/'metadata.json',metadata)
    train,val=r.getdata('train'),r.getdata('val')
    decisions={};diagnostics={}
    for variant in original_lock['variants']:
        name=variant['id'];source=a.source_run/'runs'/name/'seed17';dest=a.out/name;dest.mkdir()
        expected=original_lock['checkpoints'][f'runs/{name}/seed17/best.pt']
        assert r.sha(source/'best.pt')==expected
        checkpoint=torch.load(source/'best.pt',map_location='cpu',weights_only=False)
        model=r.build(variant['architecture'],r.CONFIGS[name]).cuda()
        model.load_state_dict(checkpoint['model'],strict=True)
        parameters={n:list(v.shape) for n,v in model.named_parameters()}
        assert all('phase' in n for n in parameters),parameters
        split_info={}
        for split,data in [('train',train),('val',val)]:
            metrics,rows=r.evaluate(model,data,predictions=True)
            r.csvwrite(dest/(split+'_predictions.csv'),rows)
            scores=np.array([[row['score0'],row['score1']] for row in rows])
            labels=np.array([row['label_true'] for row in rows])
            split_info[split]=dict(metrics=metrics,score1_quantiles_by_label={str(k):np.quantile(scores[labels==k,1],[0,.25,.5,.75,1]).tolist() for k in (0,1)})
            if split=='val':decisions[name]=choose_thresholds(labels,scores)
        diagnostics[name]=dict(selected_epoch=checkpoint['epoch'],checkpoint_sha256=expected,parameters=parameters,splits=split_info)
        r.save(dest/'diagnostics.json',diagnostics[name])
        del model,checkpoint;torch.cuda.empty_cache()
    # Freeze all policies before opening previously saved test predictions.
    r.save(a.out/'threshold_lock.json',dict(decisions=decisions,checkpoints=original_lock['checkpoints'],locked_at=r.now()))
    results=[]
    for variant in original_lock['variants']:
        name=variant['id'];source=a.source_run/'runs'/name/'seed17'
        with (source/'test_predictions.csv').open() as f:rows=list(csv.DictReader(f))
        labels=np.array([int(x['label_true']) for x in rows])
        scores=np.array([[float(x['score0']),float(x['score1'])] for x in rows])
        assert np.bincount(labels).tolist()==[229,69]
        assert len({x['sample_id'] for x in rows})==298
        old=r.read(source/'test_metrics.json')
        row=dict(variant=name,test_predictions_sha256=r.sha(source/'test_predictions.csv'),
                 train=diagnostics[name]['splits']['train']['metrics'],validation=diagnostics[name]['splits']['val']['metrics'],
                 policies={policy:full_metrics(labels,scores,choice['threshold'],old['detector_plane_mse']) for policy,choice in decisions[name]['policies'].items()})
        assert abs(row['policies']['fixed_0.5']['auroc']-old['auroc'])<1e-12
        results.append(row)
    r.save(a.out/'results.json',results)
    r.save(a.out/'status.json',dict(state='complete',weights_changed=False,test_used_for_threshold_selection=False))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
