"""Plot committed evidence, never inference; labels include budget and single seed."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/mango_figures_20261010'
OUT.mkdir(exist_ok=True)
records={budget:json.loads((HERE/'reports'/name).read_text(encoding='utf8')) for budget,name in
    [(30,'mango_variety_s17_20261010.json'),(100,'mango_variety_s17_e100_20261010.json'),
     ('100 lr3','mango_variety_s17_e100_lr3_20261010.json')]}
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.linewidth':.8,
    'svg.fonttype':'none','pdf.fonttype':42})
colors={'0.0':'#0072B2','0.3':'#D55E00'}
rows=[]
fig,axes=plt.subplots(1,3,figsize=(10.5,3.0),sharey=True)
for ax,(budget,data) in zip(axes,records.items()):
    for rho in ['0.0','0.3']:
        values=[data['results'][str(depth)][rho]['test']['accuracy']*100 for depth in [2,4,6]]
        ax.plot([2,4,6],values,marker='o' if rho=='0.0' else 's',color=colors[rho],lw=1.5,
            markersize=4,label='No residual' if rho=='0.0' else 'Residual ($\\rho=0.3$)')
        for depth,y in zip([2,4,6],values):
            other=data['results'][str(depth)]['0.3' if rho=='0.0' else '0.0']['test']['accuracy']*100
            ax.annotate(f'{y:.2f}',(depth,y),xytext=(0,7 if y>other else -13),
                        textcoords='offset points',ha='center',fontsize=7,color=colors[rho])
            e=data['results'][str(depth)][rho]
            rows.append(dict(budget_epochs=budget,depth=depth,rho=rho,seed=17,
                selected_epoch=e['epoch'],test_accuracy=y,val_accuracy=e['val']['accuracy']*100,
                test_nll=e['test']['balanced_nll'],val_nll=e['val']['balanced_nll']))
    ax.set(title='100 epochs · LR 0.003' if budget=='100 lr3' else f'{budget}-epoch budget',xlabel='Optical backbone layers',xticks=[2,4,6],
           xlim=(1.6,6.4),ylim=(55,95))
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Test accuracy (%)')
axes[2].legend(frameon=False,loc='lower right',fontsize=7)
fig.suptitle('MangoLeafVarietyBD v2 · seed 17',fontsize=10)
fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'depth_accuracy.{ext}',dpi=400)
plt.close(fig)
with (OUT/'source_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)

fig,axes=plt.subplots(2,3,figsize=(9,5.2))
data=records['100 lr3']
for j,depth in enumerate([2,4,6]):
    for rho in ['0.0','0.3']:
        history=data['curves'][str(depth)][rho]
        name='No residual' if rho=='0.0' else 'Residual'
        axes[0,j].plot([h['epoch'] for h in history],
            [h['val']['balanced_nll'] for h in history],color=colors[rho],lw=1.2,label=name)
        subset=[h for h in history if 'train' in h]
        axes[1,j].plot([h['epoch'] for h in subset],
            [h['train']['accuracy']*100 for h in subset],color=colors[rho],lw=1.2,ls='--')
        axes[1,j].plot([h['epoch'] for h in history],
            [h['val']['accuracy']*100 for h in history],color=colors[rho],lw=1.2)
    axes[0,j].set_title(f'{depth}-layer backbone')
    axes[0,j].set_ylabel('Validation balanced NLL')
    axes[1,j].set(xlabel='Epoch',ylabel='Accuracy (%)')
    for ax in axes[:,j]:ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(frameon=False,fontsize=7)
fig.suptitle('100 epochs · LR 0.003 · solid: validation, dashed: training accuracy',fontsize=10)
fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'learning_curves.{ext}',dpi=300)
plt.close(fig)
