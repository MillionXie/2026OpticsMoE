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
  ax.set_xlim(0,30 if col==0 else 40);ax.set_xlabel('Additional epochs in this branch');ax.grid(alpha=.15)
  if col==0:ax.set_ylabel('Accuracy (%)')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=10)
fig.suptitle('Seed 17 | Original TEST exposed to teacher training and model selection\nPer-branch convergence; TRAIN markers are selected-checkpoint evaluations, not epoch-wise training curves',fontsize=12)
fig.savefig(out/'boundary_curves.png',dpi=160);fig.savefig(out/'boundary_curves.svg');plt.close(fig)
