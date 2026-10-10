"""Plot rejected validation-only candidates without accessing any model/data."""
import csv
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--record',default='residual_L6_capture_smooth_20261010.json')
p.add_argument('--output',default='residual_L6_capture_smooth_figures_20261010')
p.add_argument('--development',action='store_true',help='Label explicitly authorized test-development continuations')
a=p.parse_args()
d=json.loads((root/'reports'/a.record).read_text())
out=root/'reports'/a.output
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(7,5))
rows=[]
for col,(name,arm) in enumerate(d['candidates'].items()):
    history=arm['history'];parent=arm['result']['parent_validation'];selected=arm['result']['metrics']
    axes[0,col].plot([h['epoch'] for h in history],[h['val']['balanced_nll'] for h in history],color='#D55E00',label='Candidate')
    axes[0,col].axhline(parent['balanced_nll'],color='#0072B2',ls='--',label='Parent EMA')
    axes[0,col].set(title=name if a.development else f'{name} · rejected',ylabel='Validation balanced NLL')
    train=[h for h in history if 'train' in h]
    axes[1,col].plot([h['epoch'] for h in history],[h['val']['accuracy']*100 for h in history],color='#D55E00',label='Validation')
    axes[1,col].plot([h['epoch'] for h in train],[h['train']['accuracy']*100 for h in train],color='#D55E00',ls='--',label='Training')
    axes[1,col].set(xlabel='Continuation epoch',ylabel='Accuracy (%)')
    for ax in axes[:,col]:ax.spines[['top','right']].set_visible(False)
    rows.append(dict(candidate=name,selected_epoch=arm['result']['selected_epoch'],
        epochs_completed=arm['result']['epochs_completed'],
        train_accuracy=selected['train']['accuracy']*100,
        val_accuracy=selected['val']['accuracy']*100,
        val_macro_recall=selected['val']['balanced_accuracy']*100,
        val_balanced_nll=selected['val']['balanced_nll'],
        parent_val_macro_recall=parent['balanced_accuracy']*100,
        parent_val_balanced_nll=parent['balanced_nll'],
        detector_capture=selected['val']['detector_capture'],
        selection_metric='test development' if a.development else 'validation',
        tested=a.development))
axes[0,0].legend(frameon=False,fontsize=7);axes[1,0].legend(frameon=False,fontsize=7)
fig.suptitle('Six-layer residual · test-development continuation' if a.development else
             'Six-layer residual · validation-only · neither candidate accepted',fontsize=10)
fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(out/f'validation_curves.{ext}',dpi=300)
plt.close(fig)
with (out/'source_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
for file in out.glob('*.svg'):
    file.write_text('\n'.join(s.rstrip() for s in file.read_text().splitlines())+'\n')
