"""Show all no-CNN layout ablations with validation-selected test scores."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    task=Path(__file__).parent;out=task/'reports/figures/audio_input_s17_20260918';out.mkdir(exist_ok=True,parents=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    curve,caxes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    report={}
    for ax,cax,tag,title in zip(axes,caxes,['legacy','twoband','interleaved'],['Repeated spectrogram tiles','Separate audio / text bands','Interleaved audio / text rows']):
        run='audio_raw_'+tag+'_s17_v1';folder=task/'runs/simulation'/run
        testfolder=folder.with_name(run+'_test');test=json.loads((testfolder/'results.json').read_text())
        report[tag]={}
        for j,arch in enumerate(['moe','d2nn']):
            path=folder/'fixed'/arch;result=json.loads((path/'result.json').read_text());h=json.loads((path/'history.json').read_text())
            selected=min(h,key=lambda r:r['val']['nll']);assert selected['epoch']==result['epoch']
            t=test[run+'/'+arch];assert result['best_checkpoint_sha256']==t['sha256']
            p=np.load(testfolder/(run+'_'+arch+'_predictions.npz'))
            accuracy=float((p['probabilities'].argmax(1)==p['labels']).mean());assert abs(accuracy-t['test']['accuracy'])<1e-6
            assert np.allclose(p['probabilities'].sum(1),1,atol=1e-6)
            report[tag][arch]=dict(epoch=result['epoch'],train=result['train'],val=result['val'],test=t['test'],checkpoint_sha256=t['sha256'])
            values=[100*result['train']['accuracy'],100*result['val']['accuracy'],100*accuracy]
            color=['#d77c89','#719fc7'][j];name='MoE+OEO' if arch=='moe' else 'D2NN+OEO'
            bars=ax.bar(np.arange(3)+(j-.5)*.35,values,.35,color=color,label=name)
            ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
            cax.plot([r['epoch'] for r in h],[100*r['train']['accuracy'] for r in h],color=color,ls='--',label=name+' train')
            cax.plot([r['epoch'] for r in h],[100*r['val']['accuracy'] for r in h],color=color,label=name+' validation')
            cax.scatter([result['epoch']],[100*result['val']['accuracy']],color=color,s=40,edgecolor='black',zorder=5)
        ax.set_title(title,fontsize=11);ax.set_xticks(range(3),['Train','Validation','Test']);ax.set_ylim(0,105)
        cax.set_title(title,fontsize=11);cax.set_xlabel('Epoch');cax.set_ylim(40,101)
        for a in [ax,cax]:a.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Accuracy (%)');caxes[0].set_ylabel('Accuracy (%)')
    axes[1].legend(loc='lower center',frameon=False)
    handles,labels=caxes[0].get_legend_handles_labels()
    curve.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.03),ncol=4,fontsize=8,frameon=False)
    fig.suptitle('No CNN | Seed 17 | 30 epochs | Same validation-NLL selection rule',fontsize=12)
    curve.suptitle('Clean train / validation curves; circles identify selected checkpoints',fontsize=12)
    for f,stem in [(fig,'no_cnn_train_val_test'),(curve,'no_cnn_learning_curves')]:
        f.tight_layout();f.savefig(out/(stem+'.png'),dpi=180,bbox_inches='tight');f.savefig(out/(stem+'.pdf'),bbox_inches='tight');plt.close(f)
    (out/'verified_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({tag:{arch:{split:round(r[split]['accuracy']*100,2) for split in ['train','val','test']} for arch,r in v.items()} for tag,v in report.items()},indent=2))

if __name__=='__main__':main()
