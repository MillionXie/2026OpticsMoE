"""Summarize independently audited, validation-only regularization controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    a=p.parse_args()
    tags=['baseline','lr003','drop01','flip','flipdrop01']
    labels=['Original','LR 0.003','Dropout 0.1','Flip','Flip + dropout']
    records={t:json.loads((a.data/(t+'.json')).read_text())['results'] for t in tags}
    fig,axes=plt.subplots(1,2,figsize=(13,4.7))
    for ax,arch in zip(axes,['moe','d2nn']):
        x=np.arange(len(tags))
        for dx,split,color in [(-.18,'train','#adcde0'),(.18,'val','#536eb3')]:
            values=[records[t]['fixed/'+arch][split]['accuracy']*100 for t in tags]
            bars=ax.bar(x+dx,values,.36,color=color,label=split)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set_xticks(x,labels,rotation=15)
        ax.set_ylim(0,100);ax.set_title(arch.upper()+' + per-layer OEO');ax.set_ylabel('Accuracy (%)')
        ax.axhline(50,color='.5',ls='--',lw=1)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend()
    fig.suptitle('One-hot text + shared frozen CNN | seed 17 | validation-NLL-selected checkpoints\nNew controls do not evaluate test data')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.data/('generalization_controls.'+ext),dpi=200)
    plt.close(fig)


if __name__=='__main__':main()
