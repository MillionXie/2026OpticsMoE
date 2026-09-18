"""Compare old hard-cutoff OEO and uniformly changed smooth OEO retraining."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    task=Path(__file__).parent;out=task/'reports/figures/oeo_recovery_s17_20260918';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4.7));summary={k:{} for k in ['relu','softplus','positive']}
    for i,arch in enumerate(['moe','d2nn']):
        ax=axes[i]
        for variant,color,label in [('relu','#777777','Centered ReLU'),('softplus','#db9b55','Centered Softplus'),('positive','#2686aa','Non-centered Softsign')]:
            run='audio_raw_twoband_s17_v1' if variant=='relu' else 'audio_raw_twoband_'+variant+'_'+arch+'_s17_v1'
            root=task/'runs/simulation'/run;r=json.loads((root/'fixed'/arch/'result.json').read_text())
            h=json.loads((root/'fixed'/arch/'history.json').read_text())
            troot=root.with_name(run+'_test');t=json.loads((troot/'results.json').read_text())[run+'/'+arch]
            assert r['epoch']==min(h,key=lambda x:x['val']['nll'])['epoch'] and r['best_checkpoint_sha256']==t['sha256']
            pred=np.load(troot/(run+'_'+arch+'_predictions.npz'))
            assert abs((pred['probabilities'].argmax(1)==pred['labels']).mean()-t['test']['accuracy'])<1e-6
            summary[variant][arch]=dict(epoch=r['epoch'],train=r['train'],val=r['val'],test=t['test'],checkpoint_sha256=t['sha256'])
            ax.plot([x['epoch'] for x in h],[100*x['val']['accuracy'] for x in h],color=color,label=label+' val')
            if variant=='positive':ax.plot([x['epoch'] for x in h],[100*x['train']['accuracy'] for x in h],color=color,ls='--',label=label+' train')
            ax.scatter([r['epoch']],[100*r['val']['accuracy']],color=color,edgecolor='black',s=25,zorder=5)
        ax.set_title(arch.upper()+' + per-layer OEO');ax.set_xlabel('Epoch');ax.set_ylim(40,101)
    axes[0].set_ylabel('Accuracy (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.04),ncol=4,fontsize=8,frameon=False)
    for i,(variant,color) in enumerate([('relu','#777777'),('softplus','#db9b55'),('positive','#2686aa')]):
        values=[100*summary[variant][a]['test']['accuracy'] for a in ['moe','d2nn']]
        bars=axes[2].bar(np.arange(2)+(i-1)*.24,values,.24,color=color)
        axes[2].bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
    axes[2].set_xticks([0,1],['MoE','D2NN']);axes[2].set_ylim(0,105);axes[2].set_title('Validation-selected test accuracy')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('No CNN | Same two-band input | Seed 17 | Both architectures retrained for 30 epochs',fontsize=12)
    fig.tight_layout();fig.savefig(out/'oeo_recovery.png',dpi=180,bbox_inches='tight');fig.savefig(out/'oeo_recovery.pdf',bbox_inches='tight');plt.close(fig)
    (out/'verified_results.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({tag:{arch:{k:round(v[k]['accuracy']*100,2) for k in ['train','val','test']} for arch,v in row.items()} for tag,row in summary.items()},indent=2))

if __name__=='__main__':main()
