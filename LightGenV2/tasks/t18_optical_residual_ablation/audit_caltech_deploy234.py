"""Read-only saved-result reporting after complete six-arm single-test suite."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caltech_deploy234 as t
import caltech_deploy234_finalize as f

def savefig(fig,out,name):
    for ext in ('png','svg','pdf'):fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight')
    svg=out/(name+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)

def audit(a):
    fits,m=f.audit_all(a);f.validate_lock(a)
    assert f.exited(t.read(a.out/'suite_started.json')['pid'])
    assert (a.out/'results.json').exists() and set(t.read(a.out/'results.json'))==set(f.KEYS)
    for key in f.KEYS:assert f.exited(t.read(a.out/key/'test_started.json')['pid'])
    with np.load(a.data,allow_pickle=False) as z:y=z['test_labels'].copy();ids=z['test_ids'].copy()
    assert ids.tolist()==m['split_ids']['test']
    a.report.mkdir(parents=True,exist_ok=False);summary=[];curves=[];cms=[];supports=[]
    for key in f.KEYS:
        fit=fits[key];result=t.read(a.out/key/'result.json');claim=t.read(a.out/key/'test_started.json')
        assert result['test_evaluations']==1 and not result['test_used_for_selection']
        assert result['checkpoint_sha256']==claim['checkpoint_sha256']==fit['checkpoint_sha256']
        test=f.predictions(a.out/key/'test_predictions.csv',result['test'],ids,y)
        train,val=fit['metrics']['train'],fit['metrics']['val'];d=fit['depth'];rho=fit['rho']
        summary.append(dict(key=key,depth=d,rho=rho,epochs=fit['epochs_completed'],selected_epoch=fit['selected_epoch'],train_accuracy=train['accuracy'],val_accuracy=val['accuracy'],test_accuracy=test['accuracy'],train_macro=train['balanced_accuracy'],val_macro=val['balanced_accuracy'],test_macro=test['balanced_accuracy'],train_val_gap_pp=100*(train['accuracy']-val['accuracy']),val_balanced_nll=val['balanced_nll'],test_balanced_nll=test['balanced_nll'],target75_passed=test['accuracy']>=.75,checkpoint_sha256=fit['checkpoint_sha256']))
        for split,metric in (('train',train),('val',val),('test',test)):
            for k,name in enumerate(m['classes']):supports.append(dict(key=key,split=split,label=k,class_name=name,support=metric['support'][k],recall=metric['recall'][k]))
            for i,row in enumerate(metric['confusion_matrix']):
                for j,value in enumerate(row):cms.append(dict(key=key,split=split,true=i,predicted=j,count=value))
        history=t.read(a.base/key/'history.json')
        for h in history:
            for split in ('train','val'):
                if split in h:
                    v=h[split];curves.append(dict(key=key,epoch=h['epoch'],split=split,accuracy=v['accuracy'],macro=v['balanced_accuracy'],balanced_nll=v['balanced_nll']))
    t.csvwrite(a.report/'selected_summary.csv',summary);t.csvwrite(a.report/'curve_data.csv',curves);t.csvwrite(a.report/'full_confusion.csv',cms);t.csvwrite(a.report/'per_class_support.csv',supports)
    fig,axs=plt.subplots(3,3,figsize=(15,11),sharex='col')
    metrics=[('accuracy','Accuracy'),('balanced_accuracy','Macro recall'),('balanced_nll','Balanced NLL')]
    for col,d in enumerate((2,3,4)):
        for rho,color in ((0.,'#1565c0'),(.3,'#c62828')):
            key=f'L{d}_'+('rho0' if rho==0 else 'rho03');history=t.read(a.base/key/'history.json');fit=fits[key];test=t.read(a.out/key/'result.json')['test']
            for row,(metric,label) in enumerate(metrics):
                ax=axs[row,col]
                for split,style in (('train','--'),('val','-')):
                    items=[h for h in history if split in h];ax.plot([h['epoch'] for h in items],[h[split][metric] for h in items],style,color=color,label=f'rho{rho} {split}')
                ax.scatter([fit['selected_epoch']],[test[metric]],marker='*',s=130,color=color,edgecolors='black',label=f'rho{rho} once test')
                ax.grid(alpha=.25);ax.set_ylabel(label);ax.set_xlabel('Epoch')
                if row==0:ax.set_title(f'Main {d} + one router')
        axs[0,col].legend(fontsize=7)
    fig.suptitle('Frozen screened Caltech20: exploratory development, validation NLL selection, no error bars')
    fig.tight_layout();savefig(fig,a.report,'learning_curves')
    for d in (2,3,4):
        fig,axs=plt.subplots(1,2,figsize=(16,7))
        for ax,r in zip(axs,('rho0','rho03')):
            key=f'L{d}_{r}';metric=t.read(a.out/key/'result.json')['test'];cm=np.asarray(metric['confusion_matrix']);norm=cm/np.asarray(metric['support'])[:,None]
            image=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(20),m['classes'],rotation=90,fontsize=7);ax.set_yticks(range(20),m['classes'],fontsize=7);ax.set_title(f'{key} single test, row normalized');ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.colorbar(image,ax=ax,fraction=.046)
        fig.tight_layout();savefig(fig,a.report,f'L{d}_test_confusion')
    goal=next(row for row in summary if row['key']=='L4_rho03')
    t.save(a.report/'audit.json',dict(passed=True,all_six_cpu_verified=True,test_evaluations=6,model_forward_calls=0,L4_rho03_test_accuracy=goal['test_accuracy'],target75_passed=goal['target75_passed'],test_is_exploratory_development=True,selected_summary=summary))
    (a.report/'README.md').write_text(f"# Deployable main2/3/4 saved-result audit\n\nAll six saved fits/predictions and single tests independently checked; no model inference. Main4/rho.3 test accuracy {goal['test_accuracy']:.6%}; target75 passed={goal['target75_passed']}. These are exploratory development results because earlier tests and validation screening were observed. See CSV, curves and row-normalized CM figures; actual visual inspection and hardware calibration are still required before claiming complete delivery.\n")

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('base','out','data','report'):p.add_argument('--'+key,type=Path,required=True)
    audit(p.parse_args())