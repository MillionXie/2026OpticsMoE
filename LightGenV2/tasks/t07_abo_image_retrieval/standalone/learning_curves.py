"""Matched-checkpoint clean train/test retrieval, separate from stochastic batches.

Training queries exclude their entire own product from the 120-product gallery.
Old per-epoch clean values cannot be reconstructed without those checkpoints;
the offline audit uses saved final-best features only and never invents values.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import _load_contract, _gallery_centroids, _ranking_metrics, _evaluate, _category_prototypes
from .io import sha256, write_json


@torch.no_grad()
def clean_train_metrics(features, samples):
    features=F.normalize(features.float().cpu(),dim=-1)
    gallery,items=_gallery_centroids(samples,features)
    score=(features@gallery.T).numpy()
    own=np.asarray([s.product_id for s in samples])[:,None]==np.asarray([g.product_id for g in items])[None]
    if not np.all(own.sum(1)==1):raise ValueError('Each training query must have exactly one own-product center')
    order=np.argsort(-np.where(own,-np.inf,score),axis=1)[:,:len(items)-1]
    relevant=np.asarray([g.category_id for g in items])[order]==np.asarray([s.category_id for s in samples])[:,None]
    result=_ranking_metrics(relevant)
    result.update(protocol='clean eval, no augmentation/noise/dropout, exclude entire own product',own_product_excluded=True)
    return result


def curve_rows(history):
    rows=[]
    for entry in history:
        if entry['epoch']<0:continue
        loss=entry.get('losses',{})
        row=dict(epoch=entry['epoch'],train_batch_hit1=loss.get('train_gallery_hit1'),
                 train_batch_proxy_accuracy=loss.get('correct'),train_loss=loss.get('loss'))
        for variant,key in [('ema','test'),('live','test_live')]:
            metrics=entry.get(key,{})
            if entry['epoch']==0:metrics=entry.get('test',{}) if variant=='live' else {}
            tr=metrics.get('train_clean_leave_product_out',{}).get('hit_at_1')
            te=metrics.get('hit_at_1')
            row['clean_train_hit1_'+variant]=tr
            row['test_hit1_'+variant]=te
            row['gap_'+variant+'_pp']=100*(tr-te) if tr is not None and te is not None else None
        rows.append(row)
    return rows


def write_learning_curves(history, output, fixed_best=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);rows=curve_rows(history)
    if not rows:return
    if fixed_best is not None:
        for row in rows:
            if row['epoch']==fixed_best['epoch']:
                variant=fixed_best['variant']
                row['clean_train_hit1_'+variant]=fixed_best['train']['hit_at_1']
                row['test_hit1_'+variant]=fixed_best['test']['hit_at_1']
                row['gap_'+variant+'_pp']=fixed_best['gap_pp']
    with (output/'learning_curves.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    def line(ax,key,label,scale=1,**style):
        available=[r for r in rows if r.get(key) is not None]
        if available:ax.plot([r['epoch'] for r in available],[scale*r[key] for r in available],marker='o',ms=3,label=label,**style)
    for variant,style in [('live','-'),('ema','--')]:
        line(axes[0],'clean_train_hit1_'+variant,'clean train '+variant,100,linestyle=style,color='#0072B2')
        line(axes[0],'test_hit1_'+variant,'test '+variant,100,linestyle=style,color='#D55E00')
        line(axes[1],'gap_'+variant+'_pp',variant,linestyle=style)
    line(axes[2],'train_batch_hit1','augmented batch Hit@1',100,color='#009E73')
    line(axes[2],'train_batch_proxy_accuracy','auxiliary classification',100,color='#CC79A7')
    for ax,title,ylabel in zip(axes,['Clean retrieval / same checkpoint','Clean train minus test','Training batches (not clean eval)'],['Hit@1 (%)','Gap (percentage points)','Accuracy (%)']):
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
        if ax.lines:ax.legend(fontsize=7)
    if fixed_best is not None:fig.suptitle('Historical audit: clean train available only at selected best; other epochs unavailable',fontsize=9)
    fig.savefig(output/'learning_curves.png',dpi=180)
    fig.savefig(output/'learning_curves.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,nargs='+',required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    samples,_=_load_contract(args.data)
    train=[s for s in samples if s.split=='train'];test=[s for s in samples if s.split=='test']
    args.output.mkdir(parents=True,exist_ok=False)
    summary=[]
    for path in args.artifacts:
        execution=json.loads((path/'execution.json').read_text())
        if execution['target_manifest_sha256']!=sha256(args.data/'data/abo_similarity10_manifest.csv'):
            raise ValueError('Dataset identity changed')
        saved=torch.load(path/'retrieval_features.pt',map_location='cpu',weights_only=True)
        if saved['train_ids']!=[s.sample_id for s in train] or saved['test_ids']!=[s.sample_id for s in test]:
            raise ValueError('Cached feature identities do not match dataset')
        tr=clean_train_metrics(saved['train'],train)
        gallery,items=_gallery_centroids(train,F.normalize(saved['train'].float(),dim=-1))
        te,_,_=_evaluate(saved['test'],test,gallery,items,_category_prototypes(gallery,items))
        final=json.loads((path/'final_report.json').read_text())
        if abs(te['hit_at_1']-final['metrics']['hit_at_1'])>1e-12:raise ValueError('Cached features do not reproduce final test')
        fixed=dict(run=str(path),epoch=final['selected_epoch'],variant=final.get('selected_variant','live'),
                   train=tr,test=te,gap_pp=100*(tr['hit_at_1']-te['hit_at_1']),
                   checkpoint_sha256=sha256(path/'best.pt'),features_sha256=sha256(path/'retrieval_features.pt'),
                   source_commit=execution['source_commit'])
        if fixed['variant'] not in ('live','ema'):fixed['variant']='live'
        target=args.output/path.parent.name;target.mkdir()
        write_json(target/'clean_best_train_test.json',fixed)
        write_learning_curves(json.loads((path/'history.json').read_text()),target,fixed)
        summary.append(fixed)
    write_json(args.output/'summary.json',summary)
    print(json.dumps([dict(run=r['run'],train=r['train']['hit_at_1'],test=r['test']['hit_at_1'],gap_pp=r['gap_pp']) for r in summary]))


if __name__=='__main__':main()
