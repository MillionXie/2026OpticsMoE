"""Capacity/RESISC saved-audit figures, no models, image arrays, inference or test."""
import argparse,csv,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
STUDIES={
 'capacity':dict(run='caltech_revised20_s17_mainL4rho03_phase_capacity150_gpu256_20261014',commit='cb7bfb3e2e34cb1fb03860b3db13649176b73c0f',source_count=135,cache='21436ad0b269d5319176e3e4d40fd382b21e1d34d41345ec979176231fb0e3c0',keys=('L4_rho03_phase_full','L4_rho03_phase_128','L4_rho03_phase_64'),labels=('Full grid','Main grid 128','Main grid 64'),parameters=(908552,214016,91136),train_n=1246,val_n=261,budget=150,title='Caltech revised20: phase-grid mechanism diagnostic; no architecture adoption'),
 'resisc':dict(run='resisc20_s17_mainL4_rho0rho03_fullres_validate30_sharedgpu256_20261014',commit='513f0e884a16be5f37846974deaeea55fac95f94',source_count=136,cache='652912ee341b51dfce1a5dfc9a2b7618d6d1869bb16a8a38d1f21b7e82447431',keys=('L4_rho0','L4_rho03'),labels=('Full grid rho0','Full grid rho0.3'),parameters=(908552,908552),train_n=11198,val_n=1399,budget=30,title='RESISC fixed20: original full-resolution L4 validation pair')}
STUDIES['resisc_fast']=dict(STUDIES['resisc'],run='resisc20_s17_mainL4_rho0rho03_fullres_microbatch_continue30_gpu256_20261014',commit='75be74b96e49e221c6b35a8edd94f98c0bfedac5',source_count=141,title='RESISC fixed20: original L4 pair, disclosed micro1 to8 execution continuation')

STUDIES['resisc_120']=dict(STUDIES['resisc_fast'],run='resisc20_s17_mainL4_rho0rho03_fullres_continue120_micro8_gpu256_20261014',commit='bbdaf0a852aaf9001b0c82d74dcdd592ea9865ad',source_count=144,budget=120,title='RESISC fixed20: original L4 pair, E31 onward with approved extra fitting budget')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def exited(pid):
 p=Path('/proc')/str(pid);return not p.exists() or not (p/'cmdline').read_bytes()
def csvwrite(path,rows):
 with path.open('x',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def guarded(kind,base):
 spec=STUDIES[kind];assert base.name==spec['run']
 proof=read(base/'cpu_report/audit.json');assert proof['passed'] and proof['model_forward_calls']==proof['test_arrays_read']==proof['test_evaluations']==0
 assert not (base/'cpu_report_failure.json').exists() and set(proof['fits'])==set(spec['keys'])
 dispatch=read(base/'cpu_saved_audit_dispatch_completed_20261014.json');assert dispatch['passed'] and dispatch['audit_sha256']==sha(base/'cpu_report/audit.json')
 if kind=='resisc_fast':
  migration=read(base/'cpu_report/execution_migration_cpu_checks.json');assert migration['passed'] and migration['audit_sha256']==sha(base/'cpu_report/audit.json')
  assert all(c['old_microbatch']==1 and c['new_microbatch']==8 and c['all_prefix_orders_transforms_gradients_history_preserved'] for c in migration['checks'])
 if kind=='resisc_120':
  continuation=read(base/'cpu_report/budget_continuation_cpu_checks.json')
  assert continuation['passed'] and continuation['audit_sha256']==sha(base/'cpu_report/audit.json')
  assert continuation['maximum_total_epochs']==120 and continuation['additional_fit_seconds_per_arm']==7200
  assert all(c['prefix_epochs']==30 and c['first_new_epoch']==31 and c['preserved_prefix_history_gradients_order_affine_mixup'] and c['raw_ema_restored_exact'] and not c['optimizer_ema_scheduler_reset'] for c in continuation['checks'])
 meta=read(base/'metadata.json');status=read(base/'status.json')
 assert meta['git_commit']==spec['commit'] and len(meta['sources'])==spec['source_count']
 assert exited(meta['pid']) and status['active']==status['pending']==0 and not status['failures'] and len(status['completed'])==len(spec['keys'])
 assert not read(base/'current_processes.json')['active']
 assert all(exited(j['pid']) and j['exit_code']==0 for j in status['completed'])
 data={}
 for index,key in enumerate(spec['keys']):
  fit=read(base/key/'fit_result.json');hist=read(base/key/'history.json')
  assert fit==proof['fits'][key] and fit['sources']==meta['sources'] and fit['depth']==4
  assert fit['config']['cache_sha256']==spec['cache'] and fit['metrics']['train']['n']==spec['train_n'] and fit['metrics']['val']['n']==spec['val_n']
  assert fit['test_evaluations']==0 and not fit['test_used_for_selection'] and fit['epochs_completed']==len(hist)<=spec['budget']
  assert min(hist,key=lambda h:h['val']['balanced_nll'])['epoch']==fit['selected_epoch']
  assert sha(base/key/'history.json')==fit['history_sha256'] and sha(base/key/'best_checkpoint.pt')==fit['checkpoint_sha256']
  assert not (base/key/'failure.json').exists()
  if kind=='resisc_120':
   assert fit['config']['maximum_epochs']==120 and fit['config']['maximum_seconds']==7200 and fit['config']['microbatch']==8
   assert fit['budget_continuation']['prefix_epochs']==30 and fit['budget_continuation']['additional_fit_seconds_limit']==7200
   assert fit['budget_continuation']['maximum_total_epochs']==120 and not fit['budget_continuation']['optimizer_ema_scheduler_reset']
  if kind=='capacity':assert fit['rho']==.3 and fit['config']['phase_grid']==('full','128','64')[index]
  else:assert fit['rho']==(0.,.3)[index]
  assert proof['gates'][key]==read(base/key/'validation_gate.json');data[key]=(fit,hist)
 for name,identity in meta['sources'].items():
  assert sha(ROOT/name)==identity and hashlib.sha256(subprocess.check_output(['git','show',spec['commit']+':'+name],cwd=ROOT)).hexdigest()==identity
 assert len({tuple(f['config']['classes']) for f,_ in data.values()})==1
 return spec,proof,data

def plot(kind,base,out):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists()
 spec,proof,data=guarded(kind,base);out.mkdir(parents=True,exist_ok=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
 classes=next(iter(data.values()))[0]['config']['classes'];assert len(classes)==20
 summaries=[];curves=[];classrows=[];routing=[];gate_rows=[]
 def export(fig,name):
  for ext in ('png','svg','pdf'):fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight')
  p=out/(name+'.svg');p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n',encoding='utf-8');plt.close(fig)
 fig,axes=plt.subplots(3,len(data),figsize=(5*len(data),12),sharex=True,squeeze=False)
 for col,key in enumerate(spec['keys']):
  fit,hist=data[key];selected=fit['selected_epoch'];g=proof['gates'][key];tm,vm=(fit['metrics'][s] for s in ('train','val'))
  summaries.append(dict(case=key,label=spec['labels'][col],rho=fit['rho'],execution_microbatch=fit['config']['microbatch'],prefix_epochs=fit.get('budget_continuation',fit.get('execution_migration',{})).get('prefix_epochs',0),additional_fit_seconds=fit.get('budget_continuation',{}).get('additional_fit_seconds',0),free_phase_parameters=spec['parameters'][col],epochs=fit['epochs_completed'],selected_epoch=selected,train_accuracy=tm['accuracy'],val_accuracy=vm['accuracy'],train_macro=tm['balanced_accuracy'],val_macro=vm['balanced_accuracy'],train_balanced_nll=tm['balanced_nll'],val_balanced_nll=vm['balanced_nll'],selected_gap_pp=100*(tm['accuracy']-vm['accuracy']),eligible=g['passed'],chosen_profile=(proof.get('chosen_profile_case')==key),seconds=fit['seconds'],test_evaluations=0))
  for name,passed in g['checks'].items():gate_rows.append(dict(case=key,check=name,passed=passed))
  for h in hist:
   for role in ('train','val'):
    if role in h:
     m=h[role];curves.append(dict(case=key,epoch=h['epoch'],split=role,state_kind='EMA',point_kind='measured_history',accuracy=m['accuracy'],macro_recall=m['balanced_accuracy'],balanced_nll=m['balanced_nll'],seconds=h['seconds']))
  for role in ('train','val'):
   m=fit['metrics'][role];curves.append(dict(case=key,epoch=selected,split=role,state_kind='selected_EMA',point_kind='selected_saved_predictions',accuracy=m['accuracy'],macro_recall=m['balanced_accuracy'],balanced_nll=m['balanced_nll'],seconds=hist[selected-1]['seconds']))
   for slot in range(4):routing.append(dict(case=key,split=role,slot=slot,nonzero_fraction=m['routing']['nonzero_fraction'][slot],weight_std=m['routing']['weight_std'][slot]))
  for row,(metric,label,scale,threshold) in enumerate((('accuracy','Accuracy (%)',100,65),('balanced_accuracy','Macro recall (%)',100,60),('balanced_nll','Balanced NLL (lower is better)',1,None))):
   ax=axes[row,col];train=[h for h in hist if 'train' in h]
   ax.plot([h['epoch'] for h in train],[scale*h['train'][metric] for h in train],color='#242424',marker='.',ms=3,label='Train: measured EMA')
   ax.plot([h['epoch'] for h in hist],[scale*h['val'][metric] for h in hist],color='#2475ba',label='Val: every epoch EMA')
   ax.scatter([selected],[scale*vm[metric]],s=100,marker='*',color='#cc4739',zorder=5,label='Val at first minimum NLL')
   ax.scatter([selected],[scale*tm[metric]],s=45,marker='D',facecolors='none',edgecolors='#242424',zorder=5,label='Train at same selected EMA')
   if threshold is not None:ax.axhline(threshold,color='#888888',ls=':',lw=.8,label=f'Val admission {threshold}%');ax.set_ylim(-2,102)
   else:ax.set_ylim(bottom=0);ax.margins(y=.08)
   ax.set_xlim(.5,spec['budget']+2);ax.grid(alpha=.18);ax.set_ylabel(label)
   if row==0:ax.set_title(f'{spec["labels"][col]}: {len(hist)} epochs, selected E{selected}\nEligibility: {g["passed"]}')
   if row==2:ax.set_xlabel('Completed epoch (unmeasured future epochs blank)')
   if row==0 and col==0:ax.legend(fontsize=7,loc='lower right')
  tail=hist[-min(20,len(hist)):]
  for i,name in enumerate(classes):classrows.append(dict(case=key,class_name=name,train_support=tm['support'][i],val_support=vm['support'][i],train_recall=tm['recall'][i],val_recall=vm['recall'][i],zero_val=vm['recall'][i]==0,low_val=vm['recall'][i]<.2,last_window_epochs=len(tail),last_window_zero_fraction=sum(h['val']['recall'][i]==0 for h in tail)/len(tail)))
 fig.suptitle(spec['title']+'\nSeed17, router extra; test not evaluated',fontsize=12);fig.tight_layout(rect=(0,0,1,.95));export(fig,'accuracy_macro_nll_curves')
 fig,axes=plt.subplots(1,2,figsize=(11,9));im=None
 for ax,role in zip(axes,('train','val')):
  matrix=np.asarray([data[k][0]['metrics'][role]['recall'] for k in spec['keys']]).T
  im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_yticks(range(20),classes);ax.set_xticks(range(len(data)),spec['labels'],rotation=15,ha='right');ax.set_title(f'{role.title()} recall at each selected EMA')
  for y in range(20):
   for x in range(len(data)):ax.text(x,y,f'{100*matrix[y,x]:.0f}',ha='center',va='center',color='white' if matrix[y,x]>.6 else 'black',fontsize=8)
 fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.035,pad=.035,label='Recall (0-1)');fig.suptitle('Selected class recall (%) - actual support in CSV; no test',fontsize=12);fig.subplots_adjust(left=.14,right=.86,wspace=.65,top=.92,bottom=.11);export(fig,'selected_class_recall')
 for key in spec['keys']:
  fit=data[key][0];cm=np.asarray(fit['metrics']['val']['confusion_matrix']);support=np.asarray(fit['metrics']['val']['support']);assert np.array_equal(cm.sum(1),support) and (support>0).all()
  fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(cm/support[:,None],vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(20),classes,rotation=55,ha='right',fontsize=7);ax.set_yticks(range(20),classes,fontsize=8);ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title(f'{key}: selected E{fit["selected_epoch"]} validation CM\nRow-normalized by true-class support; no test');fig.colorbar(im,ax=ax,label='Fraction of true-class support',fraction=.04,pad=.03);fig.tight_layout();export(fig,key+'_validation_cm')
  csvwrite(out/(key+'_validation_cm.csv'),[dict(class_true=classes[y],support=int(support[y]),**{name:int(cm[y,x]) for x,name in enumerate(classes)}) for y in range(20)])
 for name,rows in (('curve_points',curves),('selected_summary',summaries),('class_support_recall',classrows),('routing_slots',routing),('eligibility_checks',gate_rows)):csvwrite(out/(name+'.csv'),rows)
 receipt=dict(study_kind=kind,run_id=base.name,source_sha256=sha(__file__),cpu_audit_sha256=sha(base/'cpu_report/audit.json'),training_source_commit=spec['commit'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()},scope='Audited saved EMA metrics only; sparse train/every-epoch val and same selected EMA. Single seed, no error bars, no test points or architecture adoption.',pdf_count=2+len(data),model_constructed=0,model_forward_calls=0,image_arrays_read=0,test_arrays_read=0,test_evaluations=0,pdf_visual_inspection_pending=True,chosen_profile_case=proof.get('chosen_profile_case'),all_pair_eligible=proof.get('all_pair_eligible'))
 (out/'artifact_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=STUDIES,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert not a.out.exists(), 'Preserve an existing or failed figure directory; do not re-enter'
 try:print(json.dumps(plot(a.kind,a.base,a.out)))
 except BaseException as e:
  if a.out.exists():
   with (a.out/'failure.json').open('x',encoding='utf-8') as f:json.dump(dict(error=repr(e),automatic_retry=False),f)
  raise
