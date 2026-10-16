"""Reproduce aggregate equal100 figures; no models, raw IDs or image inference."""
import argparse,csv,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,MultipleLocator
import numpy as np

HERE=Path(__file__).resolve().parent
def read(name):
    with (HERE/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))

def draw(out):
    assert out.resolve().is_relative_to(HERE.resolve()) and not out.exists()
    rows=read('depth_comparison.csv');epochs=read('epoch_metrics.csv');assert len(rows)==6 and len(epochs)==600
    case={r['case']:r for r in rows};assert set(case)=={f'L{d}_rho{r}' for d in (2,4,6) for r in ('03','0')}
    assert all(int(r['actual_epochs'])==100 for r in rows)
    for key in case:assert sorted(int(r['epoch']) for r in epochs if r['case']==key)==list(range(1,101))
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,'xtick.direction':'in','ytick.direction':'in','svg.fonttype':'none','savefig.dpi':220})
    out.mkdir();blue='#245b83';orange='#bb714d';created=[]
    def save(fig,name):
        for ext in ('png','svg'):
            p=out/(name+'.'+ext);fig.savefig(p,bbox_inches='tight',facecolor='white');created.append(p)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.6));positions=np.arange(3);res=np.array([float(case[f'L{d}_rho03']['test_accuracy_percent']) for d in (2,4,6)]);ctrl=np.array([float(case[f'L{d}_rho0']['test_accuracy_percent']) for d in (2,4,6)])
    assert min(np.r_[res,ctrl])>=60
    ax.bar(positions-.17,res,.32,color=blue,edgecolor='black',linewidth=.5,label='Optical residual')
    ax.bar(positions+.17,ctrl,.32,color=orange,edgecolor='black',linewidth=.5,label='No optical residual')
    for i,(a,b) in enumerate(zip(res,ctrl)):
        ax.text(i-.17,a+.55,f'{a:.2f}',ha='center',va='bottom',fontsize=9)
        ax.text(i+.17,b+.55,f'{b:.2f}',ha='center',va='bottom',fontsize=9)
        ax.text(i,max(a,b)+3.8,f'Gap {a-b:+.2f} pts',ha='center',va='bottom',fontsize=9)
    ax.axhline(80,color='#767676',linestyle=(0,(4,3)),linewidth=.8,zorder=0,label='80% reference')
    ax.set_ylim(60,100);ax.set_xlim(-.55,2.55);ax.set_xticks(positions,['2','4','6']);ax.set_yticks(np.arange(60,101,10));ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.set_ylabel('Test accuracy');ax.set_xlabel('Main optical layers (one router + E-G cycles)');ax.set_title('Caltech rich20 development TEST | equal100',pad=13)
    ax.grid(axis='y',color='#dddddd',linewidth=.5);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper right',ncol=3,fontsize=9)
    fig.tight_layout();save(fig,'test_accuracy_depth246')
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.0),sharey=True)
    for ax,depth in zip(axes,(2,4,6)):
        for rho,color,label in (('03',blue,'Optical residual'),('0',orange,'No optical residual')):
            points=sorted((r for r in epochs if r['case']==f'L{depth}_rho{rho}'),key=lambda r:int(r['epoch']));x=[int(r['epoch']) for r in points]
            ax.plot(x,[100*float(r['train_accuracy']) for r in points],color=color,lw=1.3,label=label+' train')
            ax.plot(x,[100*float(r['val_accuracy']) for r in points],color=color,lw=1.2,ls=(0,(4,2)),label=label+' validation')
        ax.set_title(f'{depth} main layers');ax.set_xlabel('Epoch');ax.set_xlim(1,100);ax.set_ylim(0,102);ax.set_xticks([1,25,50,75,100]);ax.set_yticks([0,20,40,60,80,100]);ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.grid(axis='y',color='#dddddd',lw=.5)
    axes[0].set_ylabel('Accuracy');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.02),fontsize=9)
    fig.suptitle('Caltech rich20 | complete100-epoch train / validation curves',y=1.01,fontsize=13);fig.tight_layout(rect=(0,.09,1,.97));save(fig,'training_accuracy_depth246')
    with (out/'artifact_sha256.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['artifact','sha256','bytes']);writer.writeheader()
        # SVG/source/data are the public reproducible artifacts; PNGs are local previews.
        for p in [p for p in created if p.suffix=='.svg']+[HERE/'plot_metrics_utf8.py',HERE/'depth_comparison.csv',HERE/'epoch_metrics.csv']:
            raw=p.read_bytes();writer.writerow(dict(artifact=str(p.relative_to(HERE)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    print('\n'.join(str(p) for p in created))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'figures_corrected');a=p.parse_args();draw(a.out)
