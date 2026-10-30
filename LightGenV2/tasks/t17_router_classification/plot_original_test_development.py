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
