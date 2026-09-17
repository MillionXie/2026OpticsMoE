"""Independent prediction/selection checks and paired validation uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np

from .prepare import save, digest


def metrics(p,y):
    return dict(accuracy=float(np.mean(p.argmax(1)==y)),
                nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((a.data/'val_questions.json').read_text())
    image_ids=[r['image_id'] for r in rows]
    keys=sorted(set(image_ids))
    groups=[np.array([i for i,x in enumerate(image_ids) if x==k]) for k in keys]
    report={};correct={};frontends={};phase_hashes={}
    for run in a.runs:
        status=json.loads((run/'status.json').read_text());assert status['status']=='complete'
        assert status['test_accessed'] is False
        for path in run.glob('*/*/result.json'):
            mode=path.parent.parent.name;arch=path.parent.name;key=mode+'/'+arch
            result=json.loads(path.read_text());pred=np.load(path.parent/'predictions.npz')
            history=json.loads((path.parent/'history.json').read_text())
            selected=min(history,key=lambda r:r['val']['nll'])
            assert selected['epoch']==result['epoch']
            for split in ['train','val']:
                score=metrics(pred[split],pred[split+'_labels'])
                for metric,value in score.items():assert abs(value-result[split][metric])<2e-6
                assert np.isfinite(pred[split]).all()
                assert np.allclose(pred[split].sum(1),1,atol=1e-6)
            assert digest((path.parent/'best_checkpoint.pt').read_bytes())==result['best_checkpoint_sha256']
            frontends.setdefault(mode,[]).append(result['frontend_sha256'])
            correct[key]=(pred['val'].argmax(1)==pred['val_labels']).astype(float)
            report[key]=dict(result,verified_prediction_metrics=True,verified_validation_selection=True,
                            source_run=str(run))
    for hashes in frontends.values():assert len(set(hashes))==1
    rng=np.random.default_rng(20260917);comparisons={}
    for left,right in [('fixed/moe','fixed/d2nn'),('learned/moe','learned/d2nn'),
                       ('learned/moe','fixed/moe'),('learned/d2nn','fixed/d2nn')]:
        if left not in correct or right not in correct:continue
        delta=np.array([(correct[left][g]-correct[right][g]).mean() for g in groups])
        bootstrap=delta[rng.integers(0,len(groups),(5000,len(groups)))].mean(1)
        comparisons[left+' minus '+right]=dict(mean_percentage_points=float(delta.mean()*100),
                                               image_cluster_bootstrap_95ci_percentage_points=(np.quantile(bootstrap,[.025,.975])*100).tolist())
    # A query-only lookup baseline trained without validation labels.
    train=json.loads((a.data/'train_questions.json').read_text());counts={}
    for r in train:
        key=(r['color'],r['shape']);counts.setdefault(key,[0,0]);counts[key][r['label']]+=1
    prior=np.array([int(counts[(r['color'],r['shape'])][1]>counts[(r['color'],r['shape'])][0]) for r in rows])
    truth=np.array([r['label'] for r in rows])
    save(a.out/'verification.json',dict(passed=True,results=report,comparisons=comparisons,
                                       question_only_train_prior_accuracy=float((prior==truth).mean()),
                                       image_only_balanced_query_baseline=.5,
                                       unique_validation_images=len(groups),test_accessed=False))
    print(json.dumps(dict(comparisons=comparisons,results={k:{'train':v['train']['accuracy'],'val':v['val']['accuracy']} for k,v in report.items()}),indent=2))


if __name__=='__main__':main()
