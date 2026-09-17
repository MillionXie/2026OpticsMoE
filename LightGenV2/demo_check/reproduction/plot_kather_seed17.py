"""Independently verify all split predictions and plot three train/test bar charts."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_bloodmnist import predictions,sha

ARMS=['moe_nooeo','d2nn_wide_nooeo','moe','d2nn_wide']
LABELS=['MoE','D2NN','MoE + OEO','D2NN + OEO']

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    results=json.loads((a.run/'results.json').read_text());assert len(results)==12;assert json.loads((a.run/'status.json').read_text())['state']=='complete'
    assert {(x['arch'],x['depth'],x['seed']) for x in results}=={(arch,d,17) for arch in ARMS for d in [2,4,6]}
    rows=[];identities=[]
    for x in results:
        name=f"{x['arch']}_L{x['depth']}_seed17";folder=a.run/name;lock=json.loads((folder/'lock.json').read_text());assert lock['entry']['result']['checkpoint_sha256']==x['checkpoint_sha256'];assert sha(folder/'test_predictions.csv')==x['test_csv_sha256'];ids={}
        for split in ['train','val','test']:ids[split]=set(predictions(folder/(split+'_predictions.csv'),x['metrics'][split])[0])
        assert not(ids['train']&ids['test'] or ids['train']&ids['val'] or ids['val']&ids['test']);assert [len(ids[k]) for k in ['train','val','test']]==[3496,752,752]
        row=dict(arch=x['arch'],depth=x['depth'],seed=17,selected_epoch=x['selected_epoch'],epochs_completed=x['epochs_completed'])
        for split in ['train','val','test']:
            for metric in ['accuracy','balanced_accuracy','macro_f1','macro_ovr_auroc','balanced_nll','detector_capture']:row[split+'_'+metric]=x['metrics'][split][metric]
        row['train_test_gap_pp']=100*(row['train_accuracy']-row['test_accuracy']);row['train_val_gap_pp']=100*(row['train_accuracy']-row['val_accuracy']);rows.append(row);identities.append(dict(model=name,test_csv_sha256=x['test_csv_sha256'],checkpoint_sha256=x['checkpoint_sha256']))
    rows.sort(key=lambda r:(r['depth'],ARMS.index(r['arch'])))
    with (a.out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def panel(ax,depth):
        group=[r for r in rows if r['depth']==depth];xs=np.arange(4);width=.34
        for shift,key,color,label in [(-width/2,'train_accuracy','#aecbe5','Train'),(width/2,'test_accuracy','#557da7','Test')]:
            bars=ax.bar(xs+shift,[100*r[key] for r in group],width,color=color,label=label);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
        ax.set_xticks(xs,LABELS);ax.set_ylim(0,108);ax.set_ylabel('Accuracy (%)');ax.set_title(f'Kather2016: {depth} main layers',pad=14);ax.legend(loc='upper left',ncol=2,frameon=False);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for depth in [2,4,6]:
        fig,ax=plt.subplots(figsize=(9,5.5));panel(ax,depth);fig.text(.5,.02,'Seed 17; corrected input coverage; checkpoint selected on validation only.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,1])
        for ext in ['png','pdf','svg']:fig.savefig(a.out/f'train_test_L{depth}.{ext}',dpi=240)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(19,5.4))
    for ax,depth in zip(axes,[2,4,6]):panel(ax,depth);ax.tick_params(axis='x',labelsize=9)
    fig.text(.5,.025,'Single seed (17), fixed validation-selected configuration; no seed-uncertainty estimate.',ha='center');fig.tight_layout(rect=[0,.06,1,1])
    for ext in ['png','pdf','svg']:fig.savefig(a.out/('train_test_all_depths.'+ext),dpi=240)
    plt.close(fig)
    (a.out/'independent_verification.json').write_text(json.dumps(dict(passed=True,models=12,prediction_files=36,split_ids_disjoint=True,results_sha256=sha(a.run/'results.json'),protocol_lock_sha256=sha(a.run/'protocol_lock.json'),identities=identities,verifier_sha256=sha(Path(__file__))),indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
