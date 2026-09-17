"""Independently verify test predictions and plot all three data partitions."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--test',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    old=json.loads(a.audit.read_text())['results']
    results=json.loads((a.test/'results.json').read_text())
    rows=json.loads((a.test/'test_questions.json').read_text())
    assert len(rows)==1500 and len({r['image_id'] for r in rows})==250
    labels=np.array([r['label'] for r in rows]);assert labels.sum()==750
    for key,v in results.items():
        pred=np.load(a.test/(key.replace('/','_')+'_predictions.npz'))
        assert np.array_equal(labels,pred['labels'])
        pr=pred['probabilities'];assert np.isfinite(pr).all() and np.allclose(pr.sum(1),1,atol=1e-6)
        assert abs(float((pr.argmax(1)==labels).mean())-v['test']['accuracy'])<1e-6
        assert abs(float(-np.log(pr[np.arange(len(labels)),labels].clip(1e-12)).mean())-v['test']['nll'])<2e-6
        assert v['sha256']==old[key]['best_checkpoint_sha256'] and v['epoch']==old[key]['epoch']
    keys=list(results);x=np.arange(len(keys))
    fig,ax=plt.subplots(figsize=(11,5))
    for offset,split,color in [(-.25,'train','#ADCDE0'),(0,'val','#738BB5'),(.25,'test','#CE7775')]:
        values=[100*(results[k]['test']['accuracy'] if split=='test' else old[k][split]['accuracy']) for k in keys]
        bars=ax.bar(x+offset,values,.25,label={'train':'Train','val':'Validation','test':'Test'}[split],color=color)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    names={'fixed':'One-hot','fixed_dense':'Dense','learned':'GRU'}
    ax.set_xticks(x,[names[k.split('/')[0]]+'\n'+('MoE' if k.endswith('/moe') else 'D2NN') for k in keys])
    ax.set_ylim(0,100);ax.set_ylabel('Accuracy (%)');ax.axhline(50,color='.5',ls='--',lw=1)
    ax.legend(ncol=3);ax.spines[['top','right']].set_visible(False)
    ax.set_title('Shared frozen visual CNN | 2 layers + per-layer OEO | seed 17\nAll checkpoints selected by validation NLL before test evaluation')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.out/('train_val_test.'+ext),dpi=200)
    plt.close(fig)
    (a.out/'test_verified.json').write_text(json.dumps(dict(verified=True,results=results,test_images=250,test_questions=1500),indent=2))


if __name__=='__main__':main()
