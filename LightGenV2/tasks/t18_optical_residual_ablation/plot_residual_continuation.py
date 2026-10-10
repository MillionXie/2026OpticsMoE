"""Draw saved residual-only exploratory results; no training or inference."""
import csv
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--record',default='residual_continuation_20261010.json')
p.add_argument('--output',default='residual_continuation_figures_20261010')
a=p.parse_args()
d=json.loads((root/'reports'/a.record).read_text())
out=root/'reports'/a.output
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
depths=[2,4,6]
baseline=[74.40191387559808,82.29665071770334,87.55980861244019]
residual=[d['evaluations'][str(x)]['test']['accuracy']*100 for x in depths]
fig,ax=plt.subplots(figsize=(4.2,3.1))
ax.plot(depths,baseline,'o-',color='#0072B2',label='Fixed baseline · 100 epochs')
ax.plot(depths,residual,'s-',color='#D55E00',label='Residual 0.3 · extra optimization')
for x,a,b in zip(depths,baseline,residual):
    ax.annotate(f'{a:.2f}',(x,a),xytext=(0,7 if a>b else -13),textcoords='offset points',ha='center',color='#0072B2')
    ax.annotate(f'{b:.2f}',(x,b),xytext=(0,-13 if a>b else 7),textcoords='offset points',ha='center',color='#D55E00')
ax.set(xlabel='Optical backbone layers',ylabel='Test accuracy (%)',xticks=depths,
       xlim=(1.6,6.4),ylim=(70,92),title='MangoLeafVarietyBD v2 · seed 17')
ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,fontsize=7,loc='lower right')
fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(out/f'depth_accuracy.{ext}',dpi=350)
plt.close(fig)
rows=[]
for dep,base,res in zip(depths,baseline,residual):
    arm=d['arms'][str(dep)];result=arm['result'];metrics=d['evaluations'][str(dep)]
    rows.append(dict(depth=dep,rho=.3,baseline_test_accuracy=base,test_accuracy=res,
        validation_accuracy=metrics['val']['accuracy']*100,
        train_accuracy=result['metrics']['train']['accuracy']*100,
        val_balanced_nll=metrics['val']['balanced_nll'],test_balanced_nll=metrics['test']['balanced_nll'],
        selected_local_epoch=metrics['epoch'],local_epochs_completed=result['epochs_completed'],
        equal_budget=False,seed=17))
with (out/'source_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(2,3,figsize=(9,5))
for col,dep in enumerate(depths):
    history=d['arms'][str(dep)]['history']
    axes[0,col].plot([x['epoch'] for x in history],[x['val']['balanced_nll'] for x in history],color='#D55E00')
    axes[0,col].axhline(d['arms'][str(dep)]['result']['parent_validation']['balanced_nll'],ls='--',color='gray',lw=1,label='Parent checkpoint')
    axes[0,col].set(title=f'{dep} layers',ylabel='Validation balanced NLL')
    tr=[x for x in history if 'train' in x]
    axes[1,col].plot([x['epoch'] for x in tr],[x['train']['accuracy']*100 for x in tr],ls='--',color='#D55E00',label='Train')
    axes[1,col].plot([x['epoch'] for x in history],[x['val']['accuracy']*100 for x in history],color='#D55E00',label='Validation')
    axes[1,col].set(xlabel='Local continuation epoch',ylabel='Accuracy (%)')
    for ax in axes[:,col]:ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(frameon=False,fontsize=7);axes[1,0].legend(frameon=False,fontsize=7)
fig.suptitle('Residual-only continuation · local epochs of each selected run',fontsize=10)
fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(out/f'learning_curves.{ext}',dpi=300)
plt.close(fig)
for file in out.glob('*.svg'):
    file.write_text('\n'.join(x.rstrip() for x in file.read_text().splitlines())+'\n')
