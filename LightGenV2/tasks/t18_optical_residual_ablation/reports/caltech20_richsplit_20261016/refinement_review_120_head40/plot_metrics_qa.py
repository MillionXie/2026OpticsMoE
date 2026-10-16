import csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,MultipleLocator
HERE=Path(__file__).resolve().parent
def read(n):
 with (HERE/n).open(encoding='utf-8') as f:return list(csv.DictReader(f))
rows=read('depth_comparison.csv');curves=read('refinement_curves.csv');case={q['case']:q for q in rows};out=HERE/'figures_qa';assert not out.exists();out.mkdir()
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,'xtick.direction':'in','ytick.direction':'in','svg.fonttype':'none','savefig.dpi':220})
created=[]
def save(fig,name):
 for ext in ('png','svg'):
  p=out/(name+'.'+ext);fig.savefig(p,bbox_inches='tight',facecolor='white');created.append(p)
 plt.close(fig)
blue='#245b83';orange='#bb714d';pos=np.arange(3);res=np.array([float(case[f'L{d}_rho03']['test_accuracy_percent']) for d in (2,4,6)]);ctrl=np.array([float(case[f'L{d}_rho0']['test_accuracy_percent']) for d in (2,4,6)]);assert min(np.r_[res,ctrl])>=60
fig,ax=plt.subplots(figsize=(7.6,4.7));ax.bar(pos-.17,res,.32,color=blue,edgecolor='black',lw=.5,label='Optical residual, retained candidate');ax.bar(pos+.17,ctrl,.32,color=orange,edgecolor='black',lw=.5,label='No optical residual, frozen100')
for i,(a,b) in enumerate(zip(res,ctrl)):
 ax.text(i-.17,a+.55,f'{a:.2f}',ha='center',fontsize=9);ax.text(i+.17,b+.55,f'{b:.2f}',ha='center',fontsize=9);ax.text(i,max(a,b)+3.7,f'Gap {a-b:+.2f} pts',ha='center',fontsize=9)
ax.axhline(80,color='#767676',ls=(0,(4,3)),lw=.8,zorder=0);ax.set_ylim(60,100);ax.set_xlim(-.55,2.55);ax.set_xticks(pos,['2','4','6']);ax.set_yticks(np.arange(60,101,10));ax.yaxis.set_major_formatter(PercentFormatter(100,0));ax.yaxis.set_minor_locator(MultipleLocator(5));ax.set_ylabel('Test accuracy');ax.set_xlabel('Main optical layers (one router + E-G cycles)');ax.set_title('Caltech rich20 development TEST | retained candidates',pad=12);ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper right',fontsize=8.8)
fig.text(.5,.015,'Residual budgets: D2 100 epochs +40 head updates | D4 100 epochs | D6 100 epochs',ha='center',fontsize=8);fig.tight_layout(rect=(0,.045,1,1));save(fig,'test_accuracy_depth246')
fig,axes=plt.subplots(1,2,figsize=(10.6,4.1),sharey=True)
for ax,kind,match,title,label in [(axes[0],'head_update','L2_same_head_natural_to40','Two layers: same linear head','Head optimizer update'),(axes[1],'optical_epoch','L6_natural_view','Six layers: rejected natural-view trial','Optical training epoch')]:
 p=sorted((q for q in curves if q['step_kind']==kind and match in q['run'] and (kind!='optical_epoch' or int(q['step'])>=100)),key=lambda q:int(q['step']));assert len(p)==(40 if kind=='head_update' else 21)
 ax.plot([int(q['step']) for q in p],[100*float(q['train']) for q in p],color=blue,lw=1.4,label='Train');ax.plot([int(q['step']) for q in p],[100*float(q['val']) for q in p],color=orange,ls=(0,(4,2)),lw=1.3,label='Validation');ax.set_ylim(60,100);ax.set_title(title);ax.set_xlabel(label);ax.yaxis.set_major_formatter(PercentFormatter(100,0));ax.grid(axis='y',color='#dddddd',lw=.5)
axes[0].set_ylabel('Accuracy');axes[1].legend(frameon=False,loc='center right',bbox_to_anchor=(1,.65));axes[0].axvline(27,color='#888888',lw=.8,ls=':');axes[1].axvline(101,color='#888888',lw=.8,ls=':');axes[1].set_xticks([100,105,110,115,120]);fig.suptitle('Unsmoothed actual refinement curves',y=1.01,fontsize=12);fig.tight_layout();save(fig,'refinement_train_validation')
with (out/'artifact_sha256.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['artifact','sha256','bytes']);w.writeheader()
 for p in [q for q in created if q.suffix=='.svg']+[HERE/n for n in ['plot_metrics_qa.py','depth_comparison.csv','class_recall_test.csv','attempts.csv','refinement_curves.csv']]:raw=p.read_bytes();w.writerow(dict(artifact=str(p.relative_to(HERE)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
print('\n'.join(str(p) for p in created))
