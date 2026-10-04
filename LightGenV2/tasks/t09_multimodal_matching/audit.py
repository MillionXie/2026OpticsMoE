"""Independent prediction/selection checks and paired validation uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

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
    group_ids=[r.get('speaker',r['image_id']) for r in rows]
    keys=sorted(set(group_ids))
    groups=[np.array([i for i,x in enumerate(group_ids) if x==k]) for k in keys]
    report={};correct={};frontends={};phase_hashes={}
    visual_hashes=[]
    for run in a.runs:
        status=json.loads((run/'status.json').read_text());assert status['status']=='complete'
        assert status['test_accessed'] is False
        if (run/'shared_visual_frontend.json').exists():
            visual_hashes.append(json.loads((run/'shared_visual_frontend.json').read_text()))
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
            checkpoint=torch.load(path.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
            phase_hashes.setdefault(arch,[]).append(checkpoint['initial_phase_sha256'])
            frontends.setdefault(mode,[]).append(result['frontend_sha256'])
            correct[key]=(pred['val'].argmax(1)==pred['val_labels']).astype(float)
            report[key]=dict(result,verified_prediction_metrics=True,verified_validation_selection=True,
                            source_run=str(run))
    for hashes in frontends.values():assert len(set(hashes))==1
    for hashes in phase_hashes.values():assert len(set(hashes))==1
    if visual_hashes:assert all(v==visual_hashes[0] for v in visual_hashes)
    rng=np.random.default_rng(20260917);comparisons={}
    for left,right in [('fixed/moe','fixed/d2nn'),('learned/moe','learned/d2nn'),
                       ('learned/moe','fixed/moe'),('learned/d2nn','fixed/d2nn'),
                       ('fixed_dense/moe','fixed_dense/d2nn'),
                       ('fixed_dense/moe','fixed/moe'),('fixed_dense/d2nn','fixed/d2nn'),
                       ('learned/moe','fixed_dense/moe'),('learned/d2nn','fixed_dense/d2nn')]:
        if left not in correct or right not in correct:continue
        delta=np.array([(correct[left][g]-correct[right][g]).sum() for g in groups])
        sizes=np.array([len(g) for g in groups])
        draws=rng.integers(0,len(groups),(5000,len(groups)))
        bootstrap=delta[draws].sum(1)/sizes[draws].sum(1)
        comparisons[left+' minus '+right]=dict(mean_percentage_points=float(delta.sum()/sizes.sum()*100),
                                               cluster_bootstrap_95ci_percentage_points=(np.quantile(bootstrap,[.025,.975])*100).tolist())
    # A query-only lookup baseline trained without validation labels.
    train=json.loads((a.data/'train_questions.json').read_text());counts={}
    def query_key(row):
        return (row['color'],row['shape']) if 'color' in row else row['query_class']
    for r in train:
        key=query_key(r);counts.setdefault(key,[0,0]);counts[key][r['label']]+=1
    prior=np.array([int(counts[query_key(r)][1]>counts[query_key(r)][0]) for r in rows])
    truth=np.array([r['label'] for r in rows])
    save(a.out/'verification.json',dict(passed=True,results=report,comparisons=comparisons,
                                       question_only_train_prior_accuracy=float((prior==truth).mean()),
                                       image_only_balanced_query_baseline=.5,
                                       shared_visual_features_identical=bool(visual_hashes),
                                       identical_optical_initialization_across_encodings=True,
                                       uncertainty_scope='Conditional on validation-selected models; excludes training-seed and model-selection uncertainty',
                                       bootstrap_unit='speaker' if 'speaker' in rows[0] else 'image',
                                       validation_groups=len(groups),unique_validation_images=len(set(image_ids)),test_accessed=False))
    print(json.dumps(dict(comparisons=comparisons,results={k:{'train':v['train']['accuracy'],'val':v['val']['accuracy']} for k,v in report.items()}),indent=2))


if __name__=='__main__':main()
