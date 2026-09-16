"""Independent CPU metrics, selection, pairing audit, and unsmoothed curves."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def auc(y,s):return float(((s[y==1,None]>s[y==0][None,:])+.5*(s[y==1,None]==s[y==0][None,:])).mean())

def predictions(path,expected):
    with path.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    ids=[x['sample_id'] for x in rows];assert len(set(ids))==len(ids)
    y=np.array([int(x['label_true']) for x in rows]);p=np.array([[float(x['score0']),float(x['score1'])] for x in rows])
    assert np.bincount(y).tolist()==expected['support'] and np.allclose(p.sum(1),1,atol=1e-6)
    pred=p[:,1]>.5;cm=np.zeros((2,2),int);np.add.at(cm,(y,pred.astype(int)),1)
    assert cm.tolist()==expected['confusion_matrix']
    assert abs(auc(y,p[:,1])-expected['auroc'])<1e-12
    assert abs((y==pred).mean()-expected['accuracy'])<1e-12
    l=-np.log(p[np.arange(len(y)),y].clip(1e-9));bnll=float(np.mean([l[y==c].mean() for c in [0,1]]))
    assert abs(bnll-expected['balanced_nll'])<2e-6
    return ids,y,p

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
    metadata=read(root/'metadata.json');cfg=metadata['config'];results=read(root/'validation_results.json');lock=read(root/'test_lock.json')
    tests={(x['variant'],x['seed']):x for x in read(root/'test_results.json')} if (root/'test_results.json').exists() else {}
    for rel,h in lock['files'].items():
        if (root/rel).exists():assert sha(root/rel)==h
        else:assert rel.endswith('.pt'),'Missing non-checkpoint artifact'
    output=[];pairs={}
    for x in results:
        dest=root/x['variant']/('seed'+str(x['seed']));hist=read(dest/'history.json');grads=read(dest/'gradients.json')
        best=float('inf');selected=None
        for row in hist:
            if row['validation']['balanced_nll']<best-cfg['min_delta']:best=row['validation']['balanced_nll'];selected=row['epoch']
        assert selected==x['selected_epoch']
        assert all(np.isfinite(g) and g>0 for row in grads for g in row['gradient_norm'].values())
        assert all(v['rms']>0 for v in x['changes'].values())
        ids=[]
        for split in ['train','val']:ids.append(set(predictions(dest/(split+'_predictions.csv'),x['metrics'][split])[0]))
        assert not ids[0]&ids[1]
        pairs.setdefault(x['seed'],[]).append(x)
        row=dict(variant=x['variant'],seed=x['seed'],depth=int(x['variant'].split('_')[1][1:]),selected_epoch=selected,epochs_completed=x['epochs_completed'],parameters=x['parameters'],train_auc=x['metrics']['train']['auroc'],val_auc=x['metrics']['val']['auroc'],val_balanced_nll=x['metrics']['val']['balanced_nll'],val_accuracy=x['metrics']['val']['accuracy'],val_balanced_accuracy=x['metrics']['val']['balanced_accuracy'],val_recall=x['metrics']['val']['positive_recall'],gap=x['metrics']['train']['auroc']-x['metrics']['val']['auroc'])
        if (x['variant'],x['seed']) in tests:
            t=tests[(x['variant'],x['seed'])];ti,ty,tp=predictions(dest/'test_predictions.csv',t['test']);assert not set(ti)&(ids[0]|ids[1]);row.update(test_auc=t['test']['auroc'],test_balanced_accuracy=t['test']['balanced_accuracy'],test_accuracy=t['test']['accuracy'],test_recall=t['test']['positive_recall'])
        output.append(row)
    for seed,items in pairs.items():
        for field in ['orders','transforms']:
            n=min(len(x[field]) for x in items);assert len({tuple(x[field][:n]) for x in items})==1
        for depth in [2,4,6]:
            selected=[x['parameters'] for x in items if f'_L{depth}_' in x['variant']]
            if len(selected)==2:assert max(selected)/min(selected)<1.02
    report=dict(passed=True,models=len(output),test_verified=bool(tests),same_paired_input_order_and_augmentation=True,checkpoint_selection_verified=True,all_phase_tensors_have_nonzero_gradients_and_updates=True,results=output,verifier_sha256=sha(Path(__file__)))
    (root/'independent_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (root/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for seed,items in pairs.items():
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for ax,depth in zip(axes[0],[2,4,6]):
            for x in items:
                if f'_L{depth}_' not in x['variant']:continue
                hist=read(root/x['variant']/('seed'+str(seed))/'history.json');arch=x['variant'].split('_')[0];c={'moe':'#1571b8','d2nn':'#d17b22'}[arch]
                ax.plot([h['epoch'] for h in hist],[h['validation']['auroc'] for h in hist],color=c,label=arch+' val')
                clean=[h for h in hist if 'clean_train' in h];ax.plot([h['epoch'] for h in clean],[h['clean_train']['auroc'] for h in clean],'--',color=c,label=arch+' train')
            ax.set_title(f'{depth} optical layers');ax.set_ylabel('AUROC');ax.set_ylim(.45,1.);ax.grid(alpha=.2);ax.legend(fontsize=8)
        for ax,depth in zip(axes[1],[2,4,6]):
            for x in items:
                if f'_L{depth}_' not in x['variant']:continue
                hist=read(root/x['variant']/('seed'+str(seed))/'history.json');arch=x['variant'].split('_')[0];c={'moe':'#1571b8','d2nn':'#d17b22'}[arch]
                ax.plot([h['epoch'] for h in hist],[h['validation']['balanced_nll'] for h in hist],color=c,label=arch)
                ax.axvline(x['selected_epoch'],color=c,alpha=.3)
            ax.set_ylabel('Validation balanced NLL');ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        fig.suptitle(cfg['profile']+f' / seed {seed}');fig.tight_layout();fig.savefig(root/f'learning_curves_seed{seed}.png',dpi=160);fig.savefig(root/f'learning_curves_seed{seed}.pdf');plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
