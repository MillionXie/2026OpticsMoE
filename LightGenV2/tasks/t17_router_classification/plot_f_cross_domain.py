"""Separate source-domain learning curves; TEST points appear only after locks."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    task=Path(__file__).resolve().parent;run=task/'runs/simulation/f_cross_domain_s17_20261031';out=task/'reports/f_cross_domain_s17_20261031'
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,name in zip(axes.flat,['LG_RGB','LG_SAR','D2NN_RGB','D2NN_SAR']):
        path=run/name;history=json.loads((path/'history.json').read_text()) if (path/'history.json').exists() else []
        result=json.loads((path/'result.json').read_text()) if (path/'result.json').exists() else None
        clean=[r for r in history if 'train' in r]
        ax.plot([r['epoch'] for r in history],[100*r['validation']['accuracy'] for r in history],color='#333',label='Source VAL')
        ax.plot([r['epoch'] for r in clean],[100*r['train']['accuracy'] for r in clean],'--',color='#4c78a8',label='Source clean TRAIN')
        if result:
            e=result['selected_epoch'];ax.scatter(e,100*result['train']['accuracy'],marker='D',s=30,color='#4c78a8',label='Selected TRAIN')
            for target,color in [('RGB','#e45756'),('SAR','#f2b65d')]:
                value=100*result['original_test_dev']['domains'][target]['accuracy'];ax.scatter(e,value,marker='*',s=100,color=color,label=f'Locked TEST {target}: {value:.1f}%')
        ax.set(title=name+(' complete' if result else ' in progress'),xlabel='Epoch',ylabel='Accuracy (%)',ylim=(0,100));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('F seed17: source-only students; eight source-to-target TEST cells\nOriginal TEST exposed development; no independent-TEST claim',fontsize=12)
    fig.tight_layout();fig.savefig(out/'convergence.svg');fig.savefig(out/'convergence.png',dpi=150);plt.close(fig)
    svg=out/'convergence.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
