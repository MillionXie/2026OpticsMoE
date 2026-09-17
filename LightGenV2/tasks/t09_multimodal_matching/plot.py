"""Scientific figures for validation-only text encoding pilot."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--audits',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,len(a.audits),figsize=(6.2*len(a.audits),4.6),squeeze=False)
    for ax,source in zip(axes[0],a.audits):
        data=json.loads(source.read_text());results=data['results']
        keys=['fixed/moe','fixed/d2nn','learned/moe','learned/d2nn']
        x=np.arange(4);train=[results[k]['train']['accuracy']*100 for k in keys]
        val=[results[k]['val']['accuracy']*100 for k in keys]
        ax.bar(x-.18,train,.36,color='#A9CBE5',label='Train (selected checkpoint)')
        bars=ax.bar(x+.18,val,.36,color='#536EB3',label='Validation')
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
        ax.axhline(50,color='0.5',ls='--',lw=1)
        ax.set_xticks(x,['Fixed\nMoE','Fixed\nD2NN','GRU\nMoE','GRU\nD2NN'])
        ax.set_ylim(0,100);ax.set_ylabel('Accuracy (%)')
        ax.set_title('Shared frozen visual CNN' if data['shared_visual_features_identical'] else 'Raw RGB input')
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(loc='upper left',fontsize=9)
    fig.suptitle('CLEVR-derived attribute existence | 2 layers + OEO | seed 17 | NOT test results',fontsize=12)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(a.out/f'text_encoding_validation.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
