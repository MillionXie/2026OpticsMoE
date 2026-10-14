"""Saved-only hybrid figures; never reuse a pure-optical eligibility/native contract."""
import argparse,csv,json,os,subprocess
from pathlib import Path
import numpy as np
import resisc10_small_electrical as head
s=head.s;ROOT=head.ROOT;HERE=head.HERE

def guard(base):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU-only saved plotting'
    cfg=s.read(head.PROFILE);assert base.name==cfg['run'], 'Wrong hybrid run'
    proof=s.read(base/'cpu_report/audit.json');done=s.read(base/'saved_cpu_head_audit_dispatch_completed_20261015.json');claim=s.read(base/'claim.json');fit=s.read(base/'fit_result.json')
    assert proof['passed'] and done['passed'] and done['audit_sha256']==s.sha(base/'cpu_report/audit.json')
    assert not any((base/n).exists() for n in ('failure.json','cpu_report_failure.json'))
    assert head.exited(claim['pid']) and s.read(base/'status.json')['state']=='head_fit_complete'
    dispatch=s.read(base.with_suffix('.cpu_head_started.json'));assert head.exited(dispatch['pid']) and dispatch['pid']==claim['pid'] and dispatch['CPU_only']
    assert fit['config']==cfg and fit['kind']==proof['kind']=='hybrid_optical_fixed_plus_tiny_electrical'
    assert fit['sources']==proof['sources']==claim['sources'] and fit['git_commit']==claim['git_commit'] and len(fit['sources'])==167
    s.engine.verify_sources(fit['sources'],fit['git_commit'])
    assert proof['fit_result_sha256']==s.sha(base/'fit_result.json') and fit['checkpoint_sha256']==s.sha(base/'best_checkpoint.pt') and fit['history_sha256']==s.sha(base/'history.json')
    assert fit['optical_alpha']==proof['fixed_optical_alpha']==cfg['optical_alpha']==.8 and fit['electrical_weight']==1-.8 and proof['alpha_strictly_greater_than_point6']
    assert fit['electronic_parameters']==proof['electronic_parameters']==490 and fit['optical_weights_frozen'] and proof['optical_weights_frozen'] and not proof['pure_optical']
    assert all(fit[k]==proof[k]==0 for k in ('optical_model_constructed','optical_forward_calls','test_arrays_read','test_evaluations'))
    parent=Path(fit['parent']);pp,pm,key,pf=head.parent(parent)
    assert pf['checkpoint_sha256']==fit['parent_checkpoint_sha256'] and key==fit['parent_key'] and proof['parent_audit_sha256']==fit['parent_audit_sha256']==s.sha(parent/'cpu_report/audit.json')
    history=s.read(base/'history.json');assert len(history)==fit['epochs_completed']==proof['epochs'] and proof['actual_Adam_steps']==fit['optimizer_steps']==350*len(history)
    assert fit['selected_epoch']==min(history,key=lambda x:x['val']['balanced_nll'])['epoch']
    assert fit['target80_achieved']==proof['target80_achieved']==(fit['metrics']['val']['accuracy']>=.8)
    metrics={}
    for role,n in (('train',5600),('val',700)):
        rows=list(csv.DictReader((base/(role+'_predictions.csv')).open()));assert len(rows)==n
        labels=np.array([int(v['label_true']) for v in rows]);p=np.array([[float(v['score'+str(k)]) for k in range(10)] for v in rows]);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        assert np.array_equal(p.argmax(1),[int(v['label_pred']) for v in rows]);metrics[role]=head.study.metrics(labels,p);assert metrics[role]==fit['metrics'][role]
    return fit,pf,history,metrics

def figures(base,out):
    fit,parent,history,metrics=guard(base);assert not out.exists();import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=False);files=[]
    def save(fig,name):
        for ext in ('pdf','png','svg'):
            f=out/(name+'.'+ext);fig.savefig(f,dpi=160,bbox_inches='tight');files.append(f)
        plt.close(fig)
    epochs=[h['epoch'] for h in history];selected=fit['selected_epoch'];picked=next(h for h in history if h['epoch']==selected);classes=head.study.plan()['classes']
    title='RESISC validation-selected top10: hybrid development task; test not evaluated\nFrozen optical alpha0.8 + electrical0.2, only490 electrical parameters'
    fig,axs=plt.subplots(2,1,figsize=(10,8),constrained_layout=True);fig.suptitle(title)
    for ax,k,label,scale in ((axs[0],'accuracy','Validation accuracy (%)',100),(axs[1],'balanced_nll','Validation balancedNLL (lower better)',1)):
        ax.plot(epochs,[h['val'][k]*scale for h in history],label='Hybrid EMA: measured every completed epoch',color='#1976b2');ax.axhline(parent['metrics']['val'][k]*scale,label='Frozen optical parent selected EMA',color='#555',linestyle='--');ax.scatter([selected],[picked['val'][k]*scale],marker='*',s=120,color='#cc4433',label='First minimum validationNLL selected');ax.set_xlabel('Completed electrical-head epoch');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[0].axhline(80,linestyle=':',color='#228844',label='80% validation target');axs[0].set_ylim(0,100);axs[0].legend(fontsize=8);save(fig,'hybrid_validation_curves')
    fig,ax=plt.subplots(figsize=(10,7),constrained_layout=True);x=np.arange(10);ax.barh(x-.18,np.array(parent['metrics']['val']['recall'])*100,.36,label='Frozen optical parent');ax.barh(x+.18,np.array(metrics['val']['recall'])*100,.36,label='Selected hybrid');ax.set_yticks(x,classes);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Validation recall (%) - support70 per class');ax.set_title('Same700 validation IDs: frozen optical parent and selected hybrid\nDevelopment subset; alpha0.8 optical,490 electrical parameters; no test');ax.legend();ax.grid(axis='x',alpha=.2);save(fig,'hybrid_selected_class_recall')
    cm=np.array(metrics['val']['confusion_matrix']);support=np.array(metrics['val']['support']);assert cm.shape==(10,10) and np.all(support==70)
    fig,ax=plt.subplots(figsize=(10,9),constrained_layout=True);im=ax.imshow(cm/support[:,None],vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(10),classes,rotation=50,ha='right');ax.set_yticks(range(10),classes);ax.set_ylabel('True class');ax.set_xlabel('Predicted class');ax.set_title(f'Hybrid selected electrical E{selected}: validation CM\nRow-normalized support70; frozen optical alpha0.8; no test');fig.colorbar(im,ax=ax,label='Fraction of true-class support');save(fig,'hybrid_validation_cm')
    def table(name,rows):
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        files.append(out/name)
    table('hybrid_curve_points.csv',[dict(epoch=h['epoch'],val_accuracy=h['val']['accuracy'],val_macro=h['val']['balanced_accuracy'],val_balanced_nll=h['val']['balanced_nll'],seconds=h['seconds']) for h in history])
    table('hybrid_class_support_recall.csv',[dict(class_name=c,train_support=metrics['train']['support'][i],val_support=70,parent_val_recall=parent['metrics']['val']['recall'][i],hybrid_train_recall=metrics['train']['recall'][i],hybrid_val_recall=metrics['val']['recall'][i]) for i,c in enumerate(classes)])
    table('hybrid_selected_summary.csv',[dict(kind=fit['kind'],optical_alpha=.8,electrical_weight=.2,electronic_parameters=490,epochs=fit['epochs_completed'],selected_epoch=selected,parent_optical_epoch=fit['parent_selected_epoch'],train_accuracy=metrics['train']['accuracy'],val_accuracy=metrics['val']['accuracy'],val_macro=metrics['val']['balanced_accuracy'],val_balanced_nll=metrics['val']['balanced_nll'],target80_achieved=fit['target80_achieved'],test_evaluations=0)])
    table('hybrid_validation_cm.csv',[dict(class_name=c,**{d:int(cm[i,j]) for j,d in enumerate(classes)}) for i,c in enumerate(classes)])
    manifest=dict(pdf_count=3,pure_optical=False,alpha=.8,head_parameters=490,cpu_audit_sha256=s.sha(base/'cpu_report/audit.json'),plot_source_sha256=s.sha(__file__),files={f.name:s.sha(f) for f in files},test_evaluations=0);s.save(out/'artifact_manifest.json',manifest,exclusive=True);return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(figures(a.base,a.out)))
