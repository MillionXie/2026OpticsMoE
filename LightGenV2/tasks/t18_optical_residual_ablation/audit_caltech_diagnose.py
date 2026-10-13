"""CPU-only saved diagnostic audit, no model propagation or test arrays."""
import argparse,csv,json,hashlib,os
from pathlib import Path
import numpy as np
import torch
import caltech_diagnose as d


def predictions(path,expected,ids,y):
    with path.open() as f:rows=list(csv.DictReader(f))
    assert [r['sample_id'] for r in rows]==ids.tolist() and [int(r['label_true']) for r in rows]==y.tolist()
    p=np.array([[float(r[f'score{k}']) for k in range(20)] for r in rows]);assert np.array_equal(p.argmax(1),[int(r['label_pred']) for r in rows])
    metrics=d.common.metrics(y,p)
    for key in ['accuracy','balanced_accuracy','balanced_nll','recall','support','confusion_matrix']:assert np.allclose(metrics[key],expected[key],atol=1e-10),key
    return metrics


def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    meta=d.read(a.base/'metadata.json');status=d.read(a.base/'status.json');assert status['state']=='diagnostics_complete_awaiting_CPU_review' and not status['failures']
    assert not (Path('/proc')/str(meta['pid'])).exists()
    for job in status['completed']:assert not (Path('/proc')/str(job['pid'])).exists()
    d.verify(meta['sources'],meta['git_commit']);m,(train,val)=d.dataset(meta['config']);summary=[]
    for variant in meta['config']['variants']:
        p=a.base/variant;r=d.read(p/'fit_result.json');cfg=r['config'];hist=d.read(p/'history.json');gr=d.read(p/'gradients.json')
        assert len(hist)==len(gr)==r['epochs_completed']==cfg['epochs']==40
        assert [h['epoch'] for h in hist]==list(range(1,41)) and r['test_evaluations']==r['test_arrays_materialized']==0
        assert d.sha(p/'history.json')==r['history_sha256'] and d.sha(p/'best_checkpoint.pt')==r['checkpoint_sha256']
        best=torch.load(p/'best_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(p/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        chosen=min(hist,key=lambda h:h['val']['balanced_nll']);assert best['epoch']==chosen['epoch']==r['selected_epoch'] and chosen['val']==r['metrics']['val']==best['validation']
        assert best['config']==last['config']==cfg and best['sources']==last['sources']==r['sources']==meta['sources']
        assert best['data_sha256']==last['data_sha256']==m['cache_sha256']
        assert all(torch.isfinite(x).all() for x in best['model'].values())
        for x in gr:assert all(np.isfinite(v) and v>0 for v in x['norms'].values())
        assert set(gr[0]['norms'])==set(last['optimizer']['param_groups'][0]['params']) or len(gr[0]['norms'])==len(last['optimizer']['state'])
        steps={int(x['step']) for x in last['optimizer']['state'].values()};assert steps=={40*86},steps
        orders=[];transforms=[]
        for epoch in range(1,41):
            order,theta=d.common.epoch_plan(epoch,len(train[1]),cfg);orders.append(d.digest(order));mh=hashlib.sha256()
            for batch,start in enumerate(range(0,len(order),16)):
                n=len(order[start:start+16]);rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,20261013,4]));lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rng.permutation(n)
                mh.update(d.digest(np.r_[lam,perm.astype(float)].astype('<f8')).encode())
            assert hist[epoch-1]['mixup_sha256']==mh.hexdigest()
            transforms.append(hashlib.sha256((d.digest(theta)+mh.hexdigest()).encode()).hexdigest())
        assert orders==last['orders'] and transforms==last['transforms']
        tm=predictions(p/'train_predictions.csv',r['metrics']['train'],train[2],train[1].numpy());vm=predictions(p/'val_predictions.csv',r['metrics']['val'],val[2],val[1].numpy())
        summary.append(dict(variant=variant,selected_epoch=r['selected_epoch'],train_accuracy=tm['accuracy'],val_accuracy=vm['accuracy'],val_macro=vm['balanced_accuracy'],val_balanced_nll=vm['balanced_nll'],gap_pp=100*(tm['accuracy']-vm['accuracy']),class_health=r['class_health'],seconds=r['seconds'],deployment_eligible=r['deployment_eligible']))
    a.out.mkdir(parents=True,exist_ok=False);d.save(a.out/'audit.json',dict(passed=True,summary=summary,model_forward_calls=0,test_arrays_materialized=0,test_evaluations=0,all40_mixup_tail12_regenerated=True,source_commit=meta['git_commit']))
    d.common.csvwrite(a.out/'summary.csv',[{k:v for k,v in row.items() if k!='class_health'} for row in summary])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);audit(p.parse_args())
