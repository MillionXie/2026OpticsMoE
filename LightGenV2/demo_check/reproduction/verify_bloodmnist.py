"""Independent NumPy verification of multiclass scores, pairing and selection."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def auc(y,s):
    order=np.argsort(s,kind='stable');a=s[order];labels=y[order];starts=np.r_[0,np.flatnonzero(np.diff(a)!=0)+1];ends=np.r_[starts[1:],len(a)];ranks=np.empty(len(a),float)
    for b,e in zip(starts,ends):ranks[b:e]=(b+1+e)/2
    n=int(y.sum());return float((ranks[labels==1].sum()-n*(n+1)/2)/(n*(len(y)-n)))

def predictions(p,expected):
    with p.open() as f:rows=list(csv.DictReader(f))
    ids=[x['sample_id'] for x in rows];assert len(ids)==len(set(ids));y=np.array([int(x['label_true']) for x in rows]);prob=np.array([[float(x[f'score{k}']) for k in range(8)] for x in rows]);pred=prob.argmax(1)
    assert np.allclose(prob.sum(1),1,atol=1e-6) and np.isfinite(prob).all() and (prob>=0).all();cm=np.zeros((8,8),int);np.add.at(cm,(y,pred),1);rec=np.diag(cm)/cm.sum(1);f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1);nll=-np.log(prob[np.arange(len(y)),y].clip(1e-12))
    actual=dict(accuracy=float((y==pred).mean()),balanced_accuracy=float(rec.mean()),macro_f1=float(f1.mean()),macro_ovr_auroc=float(np.mean([auc((y==k).astype(int),prob[:,k]) for k in range(8)])),balanced_nll=float(np.mean([nll[y==k].mean() for k in range(8)])))
    assert cm.tolist()==expected['confusion_matrix'] and cm.sum(1).tolist()==expected['support']
    for k,v in actual.items():assert abs(v-expected[k])<2e-6,(k,v,expected[k])
    return ids,y,pred

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run;m=read(root/'metadata.json');cfg=m['config'];rs=read(root/'validation_results.json');lock=read(root/'test_lock.json');assert sha(root/'validation_results.json')==lock['validation_sha256'];tests={x['name']:x['metrics'] for x in read(root/'test_results.json')} if (root/'test_results.json').exists() else {};out=[];preds={}
    for x in rs:
        d=root/x['name'];h=read(d/'history.json');best=float('inf');chosen=None
        for row in h:
            if row['val']['balanced_nll']<best-cfg['min_delta']:best=row['val']['balanced_nll'];chosen=row['epoch']
        assert chosen==x['selected_epoch'];assert all(v>0 for v in x['updates'].values());gs=read(d/'gradients.json');assert all(np.isfinite(v) and v>=0 for row in gs for v in row['norms'].values());assert all(any(row['norms'][name]>0 for row in gs) for name in gs[0]['norms'])
        if (d/'best_checkpoint.pt').exists():assert sha(d/'best_checkpoint.pt')==x['checkpoint_sha256']
        ids=[]
        for split in ['train','val']:
            ii,_,_=predictions(d/(split+'_predictions.csv'),x['metrics'][split]);ids.append(set(ii))
        assert not ids[0]&ids[1]
        row=dict(model=x['name'],parameters=x['parameters'],selected_epoch=chosen,epochs_completed=x['epochs_completed'],train_accuracy=x['metrics']['train']['accuracy'],validation_accuracy=x['metrics']['val']['accuracy'],train_balanced_accuracy=x['metrics']['train']['balanced_accuracy'],validation_balanced_accuracy=x['metrics']['val']['balanced_accuracy'])
        if x['name'] in tests:
            ii,y,pr=predictions(d/'test_predictions.csv',tests[x['name']]);assert not set(ii)&(ids[0]|ids[1]);preds[x['name']]=(ii,y,pr);row.update(test_accuracy=tests[x['name']]['accuracy'],test_balanced_accuracy=tests[x['name']]['balanced_accuracy'],test_macro_f1=tests[x['name']]['macro_f1'],test_macro_ovr_auroc=tests[x['name']]['macro_ovr_auroc'],test_capture=tests[x['name']].get('detector_capture',''))
            clean=next(t['clean_test_metrics'] for t in read(root/'test_results.json') if t['name']==x['name']);ci,_,_=predictions(d/'test_clean_predictions.csv',clean);assert ci==[ii[i] for i in read(root/'image_overlap_audit.json')['clean_test_indices']];row.update(clean_test_accuracy=clean['accuracy'],clean_test_balanced_accuracy=clean['balanced_accuracy'])
        out.append(row)
    for field in ['orders','transforms']:
        n=min(len(x[field]) for x in rs);assert len({tuple(x[field][:n]) for x in rs})==1
    comparisons=[]
    for depth in sorted({x['depth'] for x in rs if x['arch']!='cnn'}):
        mo=next(x for x in rs if x['arch']=='moe' and x['depth']==depth);ba=next(x for x in rs if x['arch']=='d2nn' and x['depth']==depth);assert max(mo['parameters'],ba['parameters'])/min(mo['parameters'],ba['parameters'])<1.02
        if not tests:continue
        ids,y,pm=preds[mo['name']];bi,by,pb=preds[ba['name']];assert ids==bi and np.array_equal(y,by);delta=(pm==y).astype(float)-(pb==y).astype(float);rng=np.random.default_rng(20260916);boot=[]
        for _ in range(2000):
            idx=np.concatenate([rng.choice(np.flatnonzero(y==k),int((y==k).sum()),replace=True) for k in range(8)]);boot.append(delta[idx].mean())
        comparisons.append(dict(depth=depth,accuracy_difference=float(delta.mean()),paired_stratified_sample_bootstrap_ci=np.quantile(boot,[.025,.975]).tolist()))
    dummy={}
    if preds:
        y=next(iter(preds.values()))[1];dummy=dict(training_majority_class=m['majority_class'],test_majority_accuracy=float((y==m['majority_class']).mean()),test_majority_balanced_accuracy=.125,uniform_random_expected_accuracy=.125,constant_score_macro_auroc=.5,test_support=np.bincount(y,minlength=8).tolist())
    report=dict(passed=True,models=len(rs),test_verified=bool(tests),same_input_order_and_augmentation=True,selection_and_gradients_verified=True,results=out,comparisons=comparisons,dummy=dummy,verifier_sha256=sha(Path(__file__)),interval_scope='2000 paired class-stratified image bootstrap draws; not patient/seed uncertainty; no multiple comparison correction')
    (root/'independent_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (root/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    depths=sorted({x['depth'] for x in rs if x['arch']!='cnn'});fig,axes=plt.subplots(2,len(depths),figsize=(4.5*len(depths),7),squeeze=False)
    for col,depth in enumerate(depths):
        for arch,c in [('moe','#1475b9'),('d2nn','#d17b22')]:
            x=next(x for x in rs if x['arch']==arch and x['depth']==depth);h=read(root/x['name']/'history.json');tr=[row for row in h if 'train' in row];axes[0,col].plot([t['epoch'] for t in h],[t['val']['accuracy'] for t in h],color=c,label=arch+' val');axes[0,col].plot([t['epoch'] for t in tr],[t['train']['accuracy'] for t in tr],'--',color=c,label=arch+' train');axes[1,col].plot([t['epoch'] for t in h],[t['val']['detector_capture'] for t in h],color=c,label=arch);axes[0,col].axvline(x['selected_epoch'],color=c,alpha=.2)
        axes[0,col].set_title(f'{depth} layers');axes[0,col].set_ylabel('Accuracy');axes[1,col].set_ylabel('Validation detector capture');axes[1,col].set_xlabel('Epoch')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=180);fig.savefig(root/'learning_curves.pdf');plt.close(fig)
    if tests:
        fig,axes=plt.subplots(1,2,figsize=(11,4.6));positions=np.arange(len(depths));cnn=next(x for x in out if x['model'].startswith('cnn_'))
        for arch,color,shift in [('moe','#1475b9',-.18),('d2nn','#d17b22',.18)]:
            selected=[next(x for x in out if x['model'].startswith(f'{arch}_L{depth}_')) for depth in depths];bars=axes[0].bar(positions+shift,[x['test_accuracy'] for x in selected],width=.34,color=color,label=arch)
            axes[0].bar_label(bars,labels=[f"{x['test_accuracy']:.1%}" for x in selected],fontsize=9,padding=3);axes[1].plot(depths,[x['train_accuracy']-x['validation_accuracy'] for x in selected],'o-',color=color,label=arch)
        axes[0].axhline(dummy['test_majority_accuracy'],color='gray',ls=':',label='Training-majority dummy');axes[0].axhline(cnn['test_accuracy'],color='#257942',ls='--',label='Small CNN reference');axes[0].set_xticks(positions,depths);axes[0].set_ylim(0,1);axes[0].set_ylabel('Official test accuracy');axes[1].set_xticks(depths);axes[1].axhline(0,color='gray',lw=.7);axes[1].set_ylabel('Train accuracy - validation accuracy')
        for ax in axes:ax.set_xlabel('Optical depth');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
        fig.suptitle('BloodMNIST: validation-selected checkpoints, seed17');fig.tight_layout();fig.savefig(root/'selected_comparison.png',dpi=180);fig.savefig(root/'selected_comparison.pdf');plt.close(fig)
        fig,axes=plt.subplots(2,len(depths),figsize=(4.5*len(depths),8),constrained_layout=True,squeeze=False)
        for row,arch in enumerate(['moe','d2nn']):
            for col,depth in enumerate(depths):
                name=next(x['name'] for x in rs if x['arch']==arch and x['depth']==depth);cm=np.array(tests[name]['confusion_matrix']);rec=cm/cm.sum(1,keepdims=True);ax=axes[row,col];im=ax.imshow(rec,vmin=0,vmax=1,cmap='Blues');ax.set_title(f'{arch}, {depth} layers');ax.set_xticks(range(8));ax.set_yticks(range(8));ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
                for i in range(8):
                    for j in range(8):ax.text(j,i,f'{rec[i,j]*100:.0f}',ha='center',va='center',fontsize=8,color='white' if rec[i,j]>.5 else 'black')
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Fraction within each true class');fig.suptitle('Official test: row-normalized confusion matrices (%)');fig.savefig(root/'confusion_matrices.png',dpi=180);fig.savefig(root/'confusion_matrices.pdf');plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
