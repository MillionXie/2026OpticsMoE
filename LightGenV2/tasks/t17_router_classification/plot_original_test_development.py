"""Summarize explicit original-TEST development runs; no training or model selection."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
repo=Path(__file__).resolve().parents[3];task=repo/'LightGenV2/tasks/t17_router_classification';runs=task/'runs/simulation';out=task/'reports/panel_a_original_test_development_20261031'
fig,axes=plt.subplots(3,2,figsize=(11,10),sharex=True,sharey='row',constrained_layout=True)
for row,(d,name) in enumerate([('kather2016','Kather2016'),('bloodmnist128','BloodMNIST-128'),('crc9','CRC9')]):
 parent=json.loads((runs/'panel_a_original_test_dev_s17_20261031'/d/'result.json').read_text())['selected_LG']['original_test_dev']['accuracy']*100
 for col,(run,suffix,title) in enumerate([('panel_a_original_test_teacher_kd_s17_20261031','student','D4 distillation'),('panel_a_original_test_kd_identity_s17_20261031','','Identity-view control')]):
  ax=axes[row,col];p=runs/run/d/suffix/'history.json';ax.set_title(f'{name} | {title}',fontsize=11)
  if p.exists():
   h=[z for z in json.loads(p.read_text()) if 'online' in z];x=[z['epoch'] for z in h]
   ax.plot(x,[100*z['online']['validation_diagnostic']['accuracy'] for z in h],'-o',ms=3,label='Validation diagnostic',color='#306c9c')
   ax.plot(x,[100*z['online']['original_test_dev']['accuracy'] for z in h],'-o',ms=3,label='Original TEST development',color='#d67227')
   ax.plot(x,[100*z['ema']['original_test_dev']['accuracy'] for z in h],'--',label='Original TEST development (EMA)',color='#d67227',alpha=.6)
  else:ax.text(.5,.5,'Not complete in this snapshot',ha='center',transform=ax.transAxes)
  ax.axhline(parent,color='#777777',ls=':',label='Parent development score')
  ax.set_ylabel('Accuracy (%)');ax.set_xlim(0,20);ax.set_xticks([0,5,10,15,20]);ax.grid(alpha=.15)
  if row==2:ax.set_xlabel('Additional epochs in each candidate')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=10)
fig.suptitle('Original TEST exposed to model development and teacher training\nSeed 17; not an independent test evaluation',fontsize=13)
fig.savefig(out/'distillation_curves.png',dpi=180);fig.savefig(out/'distillation_curves.svg')
print(out/'distillation_curves.png')

# Additional epochs are shown per branch, never concatenated as a single training chain.
fig,axes=plt.subplots(2,5,figsize=(18,7),sharey='row',constrained_layout=True)
stages=[('panel_a_boundary_development_s17_20261031','frozen_router_boundary','Boundary loss'),
        ('panel_a_boundary_continuation_s17_20261031','frozen_router_boundary','Bounded continuation'),
        ('panel_a_boundary_joint_s17_20261031','gentle_joint_router','Gentle joint routing'),
        ('panel_a_phase_gauge_s17_20261031','phase_gauge_recenter','Phase gauge initialization'),
        ('panel_a_wrapped_phase_s17_20261031','wrapped_phase_adam','Periodic phase Adam')]
summary=json.loads((out/'results.json').read_text(encoding='utf-8'))
for row,(d,name) in enumerate([('bloodmnist128','BloodMNIST-128'),('crc9','CRC9')]):
 ref=next(c for c in summary['cases'] if c['dataset']==d)['D2NN_development_reference']['original_test_dev']['accuracy']*100
 for col,(run,suffix,title) in enumerate(stages):
  ax=axes[row,col];path=runs/run/d/suffix;history=path/'history.json'
  ax.set_title(name+' | '+title,fontsize=10)
  if history.exists():
   data=[z for z in json.loads(history.read_text()) if 'validation_diagnostic' in z]
   epochs=[z['epoch'] for z in data]
   ax.plot(epochs,[100*z['validation_diagnostic']['accuracy'] for z in data],color='#306c9c',label='VAL diagnostic')
   ax.plot(epochs,[100*z['original_test_dev']['accuracy'] for z in data],color='#d67227',label='Original TEST development')
   result=path/'result.json'
   if result.exists():
    result=json.loads(result.read_text());e=result['selected_epoch']
    ax.scatter([e],[100*result['train']['accuracy']],marker='D',color='#26846c',s=25,label='TRAIN, selected checkpoint')
    ax.scatter([e],[100*result['original_test_dev']['accuracy']],marker='*',color='#d67227',s=85,label='Selected development score',zorder=5)
   else:ax.text(.98,.03,'IN PROGRESS',ha='right',transform=ax.transAxes,fontsize=8)
  else:ax.text(.5,.5,'No run in this snapshot',ha='center',transform=ax.transAxes,fontsize=9)
  ax.axhline(ref,ls=':',color='#777',label='Frozen D2NN reference')
  ax.axhline(ref+20,ls='--',color='#a44969',label='Requested +20 pp target')
  ax.set_xlim(0,31 if col==0 else 41);ax.set_xlabel('Additional epochs in this branch');ax.grid(alpha=.15)
  if col==0:ax.set_ylabel('Accuracy (%)')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=10)
fig.suptitle('Seed 17 | Original TEST exposed to teacher training and model selection\nPer-branch convergence; TRAIN markers are selected-checkpoint evaluations, not epoch-wise training curves',fontsize=12)
fig.savefig(out/'boundary_curves.png',dpi=160);fig.savefig(out/'boundary_curves.svg');plt.close(fig)
svg=out/'boundary_curves.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')

fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for col,(variant,label) in enumerate([('dense_pair_aux_002','Dense route KL 0.02'),('dense_pair_aux_010','Dense route KL 0.10')]):
 path=runs/'panel_a_dense_router_supervision_s17_20261031/crc9'/variant;history=path/'history.json'
 if not history.exists():continue
 data=[z for z in json.loads(history.read_text()) if 'original_test_dev' in z];epochs=[z['epoch'] for z in data]
 axes[0,col].plot(epochs,[100*z['validation_diagnostic']['accuracy'] for z in data],label='VAL diagnostic',color='#306c9c')
 axes[0,col].plot(epochs,[100*z['original_test_dev']['accuracy'] for z in data],label='Original TEST development',color='#d67227')
 result=path/'result.json'
 if result.exists():
  result=json.loads(result.read_text());axes[0,col].scatter([result['selected_epoch']],[100*result['train']['accuracy']],marker='D',color='#26846c',label='Selected-checkpoint TRAIN')
 for expert in range(4):axes[1,col].plot(epochs,[100*z['original_test_dev']['router']['mean_power'][expert] for z in data],label=f'Expert {expert+1}')
 axes[0,col].set_title(label);axes[0,col].set_ylabel('Accuracy (%)');axes[1,col].set_ylabel('Mean actual Top-2 power (%)')
 for ax in axes[:,col]:ax.set_xlabel('Additional epochs in this candidate');ax.set_xlim(0,41);ax.grid(alpha=.15)
 axes[0,col].legend(fontsize=8);axes[1,col].legend(fontsize=8);axes[1,col].set_ylim(0,100)
fig.suptitle('CRC9, seed 17 | Training-only router supervision\nOriginal TEST used for development; actual sparse power, not dense auxiliary targets',fontsize=12)
fig.savefig(out/'router_supervision_curves.png',dpi=160);fig.savefig(out/'router_supervision_curves.svg');plt.close(fig)
svg=out/'router_supervision_curves.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')

path=runs/'panel_a_focal_boundary_s17_20261031/crc9/focal_boundary_gamma2'
if (path/'history.json').exists():
 data=json.loads((path/'history.json').read_text());evaluated=[z for z in data if 'original_test_dev' in z]
 fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 epochs=[z['epoch'] for z in evaluated]
 axes[0].plot(epochs,[100*z['validation_diagnostic']['accuracy'] for z in evaluated],label='VAL diagnostic',color='#306c9c')
 axes[0].plot(epochs,[100*z['original_test_dev']['accuracy'] for z in evaluated],label='Original TEST development',color='#d67227')
 if (path/'result.json').exists():
  result=json.loads((path/'result.json').read_text());axes[0].scatter([result['selected_epoch']],[100*result['train']['accuracy']],marker='D',color='#26846c',label='Selected-checkpoint TRAIN')
 axes[0].axhline(100*(858/1436+.2),color='#a44969',ls='--',label='Requested +20 pp target')
 axes[0].set_ylabel('Accuracy (%)');axes[0].legend(fontsize=8)
 axes[1].plot([z['epoch'] for z in data],[z['training_objective'] for z in data],color='#26846c');axes[1].set_ylabel('Focal boundary + 0.1 energy NLL')
 for ax in axes:ax.set_xlabel('Additional epochs in this candidate');ax.set_xlim(0,41);ax.grid(alpha=.15)
 fig.suptitle('CRC9, seed 17 | Original TEST exposed development, not independent test',fontsize=12)
 fig.savefig(out/'focal_boundary_curves.png',dpi=160);fig.savefig(out/'focal_boundary_curves.svg');plt.close(fig)
 svg=out/'focal_boundary_curves.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
