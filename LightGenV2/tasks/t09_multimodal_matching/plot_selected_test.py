"""Plot all declared held-out evaluations and verify saved predictions independently."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).parent/'reports/figures'
    out=root/'selected_test_s17_20260917'
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    combined={}
    for ax,tag,title in zip(axes,['long30','long30_phase05','audio'],['CLEVR: no phase dropout','CLEVR: phase dropout 0.05','Audio / text: no phase dropout']):
        folder=Path(__file__).parent/'runs/simulation'/('audio_test_s17_v1' if tag=='audio' else 'clevr_long30_test_s17_v1')
        audit=root/('audio_matching_s17_20260917/verification.json' if tag=='audio' else 'phase_budget_s17_20260917/'+tag+'.json')
        training=json.loads(audit.read_text())['results'];testing=json.loads((folder/'results.json').read_text())
        rows=json.loads((folder/'test_questions.json').read_text())
        correct=[]
        for j,arch in enumerate(['moe','d2nn']):
            old=training['fixed/'+arch];test=testing[tag+'/'+arch]
            assert old['best_checkpoint_sha256']==test['sha256']
            pred=np.load(folder/(tag+'_'+arch+'_predictions.npz'))
            c=(pred['probabilities'].argmax(1)==pred['labels']);correct.append(c)
            assert abs(c.mean()-test['test']['accuracy'])<1e-6
            values=[old['train']['accuracy']*100,old['val']['accuracy']*100,c.mean()*100]
            combined[tag+'/'+arch]=dict(epoch=test['epoch'],train=values[0],val=values[1],test=values[2])
            bars=ax.bar(np.arange(3)+(j-.5)*.35,values,.35,label='MoE+OEO' if arch=='moe' else 'D2NN+OEO',color=['#d77c89','#719fc7'][j])
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
        group_key='speaker' if tag=='audio' else 'image_id'
        groups=np.array([r[group_key] for r in rows]);unique=np.unique(groups)
        delta=correct[0].astype(float)-correct[1].astype(float)
        sums=np.array([delta[groups==g].sum() for g in unique]);sizes=np.array([(groups==g).sum() for g in unique])
        choices=np.random.default_rng(17).integers(len(unique),size=(5000,len(unique)))
        interval=np.quantile(sums[choices].sum(1)/sizes[choices].sum(1)*100,[.025,.975]).tolist()
        combined[tag+'_paired_test_difference']=dict(percentage_points=float(delta.mean()*100),conditional_cluster_95ci=interval,cluster=group_key,
              scope='Fixed fitted models only; excludes seed and adaptive model-selection uncertainty')
        ax.set_xticks(range(3),['Train','Validation','Test']);ax.set_title(title,fontsize=11)
        ax.set_ylim(0,105);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Accuracy (%)');axes[1].legend(loc='lower center',frameon=False)
    fig.suptitle('Seed 17 | Each checkpoint selected by validation NLL | All declared variants shown',fontsize=11)
    fig.tight_layout();fig.savefig(out/'train_val_test.png',dpi=200);fig.savefig(out/'train_val_test.pdf');plt.close(fig)
    (out/'verified_summary.json').write_text(json.dumps(combined,indent=2))
    print(json.dumps(combined,indent=2))

if __name__=='__main__':main()
