"""Validation-only audio/text matching results and modality corruption controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    audit=json.loads((a.data/'verification.json').read_text())
    diag=json.loads((a.data/'diagnostics.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    colors={'moe':'#ce7775','d2nn':'#536eb3'}
    for arch in ['moe','d2nn']:
        h=json.loads((a.data/(arch+'_history.json')).read_text())
        for split,style in [('train','--'),('val','-')]:
            axes[0].plot([r['epoch'] for r in h],[r[split]['accuracy']*100 for r in h],style,color=colors[arch],label=arch.upper()+' '+split)
    axes[0].set_xlabel('Epoch');axes[0].set_title('Training and validation');axes[0].legend(fontsize=8)
    x=np.arange(2)
    for dx,split,color in [(-.18,'train','#adcde0'),(.18,'val','#536eb3')]:
        v=[audit['results']['fixed/'+arch][split]['accuracy']*100 for arch in ['moe','d2nn']]
        bars=axes[1].bar(x+dx,v,.36,color=color,label=split);axes[1].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axes[1].set_xticks(x,['MoE','D2NN']);axes[1].set_title('Validation-NLL-selected weights');axes[1].legend(fontsize=8)
    conditions=['baseline','constant_question','constant_image','unpaired_images']
    for dx,arch in [(-.18,'moe'),(.18,'d2nn')]:
        r=diag['results']['fixed/'+arch]
        v=[(r['baseline'] if c=='baseline' else r['controls'][c])['accuracy']*100 for c in conditions]
        bars=axes[2].bar(np.arange(4)+dx,v,.36,color=colors[arch],label=arch.upper());axes[2].bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    axes[2].set_xticks(np.arange(4),['Original','Constant\ntext','Constant\naudio','Shuffled\naudio'])
    axes[2].set_title('Dependence on both modalities');axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_ylim(45,103);ax.set_ylabel('Accuracy (%)');ax.axhline(50,color='.5',ls=':',lw=1)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Speech Commands-derived audio/text matching | shared frozen CNN | 2 layers + OEO | seed 17\nSpeaker-disjoint validation; test not evaluated; corruption controls keep original labels',fontsize=11)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.data/('audio_matching.'+ext),dpi=200)
    plt.close(fig)


if __name__=='__main__':main()
