"""Training/validation curves for matched 30-epoch phase-dropout controls."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for col,arch in enumerate(['moe','d2nn']):
        for tag,color,label in [('long30','#536eb3','No phase dropout'),('long30_phase05','#ce7775','Phase dropout 5%')]:
            h=json.loads((a.data/(tag+'_'+arch+'_history.json')).read_text())
            best=min(h,key=lambda r:r['val']['nll'])
            for split,style in [('train','--'),('val','-')]:
                axes[0,col].plot([r['epoch'] for r in h],[100*r[split]['accuracy'] for r in h],style,color=color,label=label+' / '+split)
            axes[1,col].plot([r['epoch'] for r in h],[r['val']['nll'] for r in h],color=color)
            axes[1,col].scatter([best['epoch']],[best['val']['nll']],color=color,marker='D')
        axes[0,col].set_title(arch.upper()+' + per-layer OEO');axes[0,col].set_ylim(45,101)
        axes[0,col].set_ylabel('Accuracy (%)');axes[1,col].set_ylabel('Validation NLL')
        axes[1,col].set_xlabel('Epoch');axes[0,col].legend(fontsize=8)
        for ax in axes[:,col]:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same 30-epoch budget and shared frontend | seed 17 | no test evaluation')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.data/('phase_budget.'+ext),dpi=200)
    plt.close(fig)


if __name__=='__main__':main()
