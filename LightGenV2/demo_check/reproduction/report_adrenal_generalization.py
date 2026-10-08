"""All-candidate reporting, paired test intervals, and seed-wise comparison plots."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from verify_adrenal_generalization import read,auc

def scores(path):
    with path.open() as f:r=list(csv.DictReader(f))
    return [x['sample_id'] for x in r],np.array([int(x['label_true']) for x in r]),np.array([float(x['score1']) for x in r])

def main():
    p=argparse.ArgumentParser();p.add_argument('--selection-lock',type=Path,required=True);p.add_argument('--task',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);lock=read(a.selection_lock);runs=a.task/'runs/simulation';allrows=[];primary=[]
    for name in lock['runs']:
        root=runs/name;results=read(root/'validation_results.json');tests={(x['variant'],x['seed']):x for x in read(root/'test_results.json')}
        for x in results:
            t=tests[(x['variant'],x['seed'])];v=x['metrics']['val'];tr=x['metrics']['train'];m=t['test']
            allrows.append(dict(run=name,variant=x['variant'],seed=x['seed'],selected_epoch=x['selected_epoch'],epochs_completed=x['epochs_completed'],parameters=x['parameters'],train_auc=tr['auroc'],val_auc=v['auroc'],test_auc=m['auroc'],gap=tr['auroc']-v['auroc'],validation_balanced_nll=v['balanced_nll'],test_accuracy=m['accuracy'],test_balanced_accuracy=m['balanced_accuracy'],test_recall=m['positive_recall'],validation_threshold=t['val_threshold']['threshold'],calibrated_test_accuracy=t['val_threshold']['accuracy'],calibrated_test_balanced_accuracy=t['val_threshold']['balanced_accuracy'],calibrated_test_recall=t['val_threshold']['positive_recall']))
        if name not in [lock['selected_shared_configuration']]+lock['confirmation_runs']:continue
        for seed in sorted({x['seed'] for x in results}):
            for depth in [2,4,6]:
                names=[f'{arch}_L{depth}_oeo_relu_softsign' for arch in ['moe','d2nn']]
                if not all(any(x['variant']==n and x['seed']==seed for x in results) for n in names):continue
                im,y,pm=scores(root/names[0]/('seed'+str(seed))/'test_predictions.csv');ib,yb,pb=scores(root/names[1]/('seed'+str(seed))/'test_predictions.csv');assert im==ib and np.array_equal(y,yb)
                rng=np.random.default_rng(20260916);neg=np.flatnonzero(y==0);pos=np.flatnonzero(y==1);delta=[]
                for _ in range(2000):
                    idx=np.r_[rng.choice(neg,len(neg)),rng.choice(pos,len(pos))];delta.append(auc(y[idx],pm[idx])-auc(y[idx],pb[idx]))
                lo,hi=np.quantile(delta,[.025,.975]);primary.append(dict(run=name,seed=seed,depth=depth,moe_test_auroc=auc(y,pm),d2nn_test_auroc=auc(y,pb),delta=auc(y,pm)-auc(y,pb),paired_bootstrap_ci=[float(lo),float(hi)]))
    with (a.out/'all_candidates.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    repeats=[]
    native=runs/'adrenal_nll_native_s17_20260916';router=runs/'adrenal_router10_s17_20260916'
    if native.name in lock['runs'] and router.name in lock['runs']:
        for depth in [2,4,6]:
            rel=Path(f'd2nn_L{depth}_oeo_relu_softsign/seed17')
            for split in ['train','val','test']:assert (native/rel/(split+'_predictions.csv')).read_bytes()==(router/rel/(split+'_predictions.csv')).read_bytes()
            repeats.append(dict(depth=depth,all_three_splits_bitwise_identical=True))
    obj=dict(selected_configuration=lock['selected_shared_configuration'],primary_comparisons=primary,all_candidates=allrows,d2nn_router_control_replay=repeats,interval_scope='2000 class-stratified paired sample bootstrap replicates, not patient-group or training-seed confidence intervals; no multiplicity correction',test_used_for_selection=False)
    (a.out/'comparison.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5));colors={'moe':'#1571b8','d2nn':'#d17b22'}
    for seed in sorted({x['seed'] for x in primary}):
        rows=sorted([x for x in primary if x['seed']==seed],key=lambda x:x['depth']);style='o-' if seed==17 else 's--'
        for arch in ['moe','d2nn']:
            axes[0].plot([x['depth'] for x in rows],[x[arch+'_test_auroc'] for x in rows],style,color=colors[arch],label=arch+f' seed {seed}')
            chosen=[next(t for t in allrows if t['run']==x['run'] and t['seed']==seed and t['variant']==f'{arch}_L{x["depth"]}_oeo_relu_softsign') for x in rows]
            axes[1].plot([x['depth'] for x in rows],[x['gap'] for x in chosen],style,color=colors[arch],label=arch+f' seed {seed}')
    axes[0].set_ylabel('Test AUROC');axes[0].set_title('Validation-selected shared configuration');axes[1].set_ylabel('Train AUROC - validation AUROC');axes[1].set_title('Same selected checkpoint, clean inputs');axes[1].axhline(0,color='gray',lw=.6)
    for ax in axes:ax.set_xticks([2,4,6]);ax.set_xlabel('Optical depth');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(a.out/'selected_comparison.png',dpi=180);fig.savefig(a.out/'selected_comparison.pdf');print(json.dumps(dict(selected=obj['selected_configuration'],comparisons=primary),indent=2))

if __name__=='__main__':main()
