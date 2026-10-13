"""Read saved audited L4 study only; measured curves/class plots, no inference/test."""
import argparse,csv,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
KEYS=('L4_rho03_mixup02','L4_rho03_mixup04','L4_rho03_mixup06')
DATA_SHA='21436ad0b269d5319176e3e4d40fd382b21e1d34d41345ec979176231fb0e3c0'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exited(pid):
    p=Path('/proc')/str(pid);return not p.exists() or not (p/'cmdline').read_bytes()
def csvwrite(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def guarded(base):
    proof=read(base/'cpu_report/audit.json');assert proof['passed'] and proof['model_forward_calls']==proof['test_evaluations']==proof['test_arrays_read']==0 and set(proof['fits'])==set(KEYS)
    meta=read(base/'metadata.json');status=read(base/'status.json');assert exited(meta['pid']) and len(status['completed'])==3 and status['active']==status['pending']==0 and not status['failures']
    assert all(exited(j['pid']) and j['exit_code']==0 for j in status['completed']);data={}
    for key in KEYS:
        fit=read(base/key/'fit_result.json');hist=read(base/key/'history.json');assert fit==proof['fits'][key] and sha(base/key/'history.json')==fit['history_sha256'] and sha(base/key/'best_checkpoint.pt')==fit['checkpoint_sha256']
        assert fit['depth']==4 and fit['rho']==.3 and fit['config']['mixup_alpha']==dict(zip(KEYS,(.2,.4,.6)))[key]
        assert fit['config']['cache_sha256']==DATA_SHA and fit['metrics']['train']['n']==1246 and fit['metrics']['val']['n']==261 and fit['test_evaluations']==0 and not fit['test_used_for_selection']
        assert min(hist,key=lambda h:h['val']['balanced_nll'])['epoch']==fit['selected_epoch'];assert fit['sources']==meta['sources'];data[key]=(fit,hist)
    for name,identity in meta['sources'].items():assert sha(ROOT/name)==identity and hashlib.sha256(subprocess.check_output(['git','show',meta['git_commit']+':'+name],cwd=ROOT)).hexdigest()==identity
    assert len({tuple(f['config']['classes']) for f,_ in data.values()})==1
    return proof,data

def plot(base,out):
    assert not out.exists();proof,data=guarded(base);out.mkdir(parents=True,exist_ok=False);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42});curves=[];summary=[];classes=next(iter(data.values()))[0]['config']['classes'];class_rows=[]
    def export(fig,name):
        for ext in ('png','svg','pdf'):fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight')
        p=out/(name+'.svg');p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n',encoding='utf-8');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,12),sharex=True)
    for col,key in enumerate(KEYS):
        fit,hist=data[key];selected=fit['selected_epoch'];budget=fit['epochs_completed'];alpha=fit['config']['mixup_alpha'];g=proof['gates'][key];summary.append(dict(case=key,alpha=alpha,epochs=budget,selected_epoch=selected,train_accuracy=fit['metrics']['train']['accuracy'],val_accuracy=fit['metrics']['val']['accuracy'],train_macro=fit['metrics']['train']['balanced_accuracy'],val_macro=fit['metrics']['val']['balanced_accuracy'],train_balanced_nll=fit['metrics']['train']['balanced_nll'],val_balanced_nll=fit['metrics']['val']['balanced_nll'],selected_gap_pp=100*(fit['metrics']['train']['accuracy']-fit['metrics']['val']['accuracy']),eligible=g['passed'],chosen_profile=(proof['chosen_profile_case']==key),seconds=fit['seconds'],test_evaluations=0))
        for h in hist:
            for split in ('train','val'):
                if split in h:
                    m=h[split];curves.append(dict(case=key,epoch=h['epoch'],split=split,state_kind='EMA',point_kind='measured_history',accuracy=m['accuracy'],macro_recall=m['balanced_accuracy'],balanced_nll=m['balanced_nll'],seconds=h['seconds']))
        for split in ('train','val'):
            m=fit['metrics'][split];curves.append(dict(case=key,epoch=selected,split=split,state_kind='selected_EMA',point_kind='selected_saved_predictions',accuracy=m['accuracy'],macro_recall=m['balanced_accuracy'],balanced_nll=m['balanced_nll'],seconds=hist[selected-1]['seconds']))
        for row,(metric,label,scale,gate) in enumerate([('accuracy','Accuracy (%)',100,65),('balanced_accuracy','Macro recall (%)',100,60),('balanced_nll','Balanced NLL (lower is better)',1,None)]):
            ax=axes[row,col];train=[h for h in hist if 'train' in h];ax.plot([h['epoch'] for h in train],[scale*h['train'][metric] for h in train],color='#242424',marker='.',markersize=3,label='Train: measured EMA');ax.plot([h['epoch'] for h in hist],[scale*h['val'][metric] for h in hist],color='#2475ba',label='Val: every epoch EMA');ax.scatter([selected],[scale*fit['metrics']['val'][metric]],s=100,marker='*',color='#cc4739',zorder=5,label='Val at first minimum NLL');ax.scatter([selected],[scale*fit['metrics']['train'][metric]],s=45,marker='D',facecolors='none',edgecolors='#242424',zorder=5,label='Train at same selected EMA')
            if gate is not None:ax.axhline(gate,color='#8a8a8a',ls=':',lw=.8,label=f'Val admission {gate}%');ax.set_ylim(0,100)
            else:ax.set_ylim(bottom=0)
            ax.set_xlim(1,150);ax.grid(alpha=.18);ax.set_ylabel(label)
            if row==0:ax.set_title(f'MixUp {alpha}: {budget} epochs, selected E{selected}\nEligibility: {g["passed"]}')
            if row==2:ax.set_xlabel('Completed epoch (unmeasured future epochs blank)')
            if row==0 and col==0:ax.legend(fontsize=7,loc='lower right')
        for k,name in enumerate(classes):class_rows.append(dict(case=key,class_name=name,train_support=fit['metrics']['train']['support'][k],val_support=fit['metrics']['val']['support'][k],train_recall=fit['metrics']['train']['recall'][k],val_recall=fit['metrics']['val']['recall'][k],zero_val=(fit['metrics']['val']['recall'][k]==0),low_val=(fit['metrics']['val']['recall'][k]<.2)))
    fig.suptitle('L4 rho0.3 / revised20 / seed17 - validation study only; test not evaluated',fontsize=13);fig.tight_layout(rect=(0,0,1,.96));export(fig,'accuracy_macro_nll_curves')
    fig,axes=plt.subplots(1,2,figsize=(11,9));image=None
    for ax,split in zip(axes,('train','val')):
        matrix=np.array([data[k][0]['metrics'][split]['recall'] for k in KEYS]).T;image=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_yticks(range(20),classes);ax.set_xticks(range(3),['MixUp .2','MixUp .4','MixUp .6']);ax.set_title(f'{split.title()} recall at each selected EMA')
        for y in range(20):
            for x in range(3):ax.text(x,y,f'{100*matrix[y,x]:.0f}',ha='center',va='center',color='white' if matrix[y,x]>.6 else 'black',fontsize=8)
    fig.colorbar(image,ax=axes.ravel().tolist(),fraction=.035,pad=.035,label='Recall (0-1)');fig.suptitle('Selected class recall (%) - support remains unequal; no test',fontsize=12);fig.subplots_adjust(left=.13,right=.86,wspace=.65,top=.92,bottom=.08);export(fig,'selected_class_recall')
    for key in KEYS:
        fit=data[key][0];cm=np.asarray(fit['metrics']['val']['confusion_matrix']);support=np.asarray(fit['metrics']['val']['support']);assert np.array_equal(cm.sum(1),support);normalized=cm/support[:,None];fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(20),classes,rotation=55,ha='right',fontsize=7);ax.set_yticks(range(20),classes,fontsize=8);ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title(f'{key}: selected E{fit["selected_epoch"]} validation CM\nRow-normalized by true-class support; no test');fig.colorbar(im,ax=ax,label='Fraction of true-class support',fraction=.04,pad=.03);fig.tight_layout();export(fig,key+'_validation_cm')
        csvwrite(out/(key+'_validation_cm.csv'),[dict(class_true=classes[y],support=int(support[y]),**{name:int(cm[y,x]) for x,name in enumerate(classes)}) for y in range(20)])
    csvwrite(out/'curve_points.csv',curves);csvwrite(out/'selected_summary.csv',summary);csvwrite(out/'class_support_recall.csv',class_rows)
    receipt=dict(source_sha256=sha(__file__),cpu_audit_sha256=sha(base/'cpu_report/audit.json'),training_source_commit=read(base/'metadata.json')['git_commit'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()},scope='Saved audited study only; measured sparse train, every-epoch val, selected same-EMA train diamonds/val stars. No test points or single-seed error bars.',pdf_count=5,model_forward_calls=0,test_arrays_read=0,test_evaluations=0,pdf_visual_inspection_pending=True,chosen_profile_case=proof['chosen_profile_case'])
    (out/'artifact_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists()
    try:plot(a.base,a.out)
    except BaseException as e:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),automatic_retry=False)),encoding='utf-8')
        raise
