"""Verified twelve-point development plot; no fabricated error bars or monotonicity."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    task=Path(__file__).resolve().parent;out=task/'reports/panel_d_A_aligned_s17_20261101';data=json.loads((out/'results.json').read_text());fig,axes=plt.subplots(1,3,figsize=(12,4.1))
    for ax,name,title in zip(axes,['kather2016','bloodmnist128','organcmnist'],['Kather2016 (same A panel 1)','BloodMNIST128 (same A panel 2)','OrganCMNIST (original panel D)']):
        points=[v for v in data['records'] if v['dataset']==name]
        assert len(points)==4 and all(v['fully_verified'] for v in points)
        correct=[v['result']['original_test_dev']['correct'] for v in points];n=[v['result']['original_test_dev']['n'] for v in points];ys=[100*c/z for c,z in zip(correct,n)]
        ax.plot(range(1,5),ys,'o-',color='#aa4499',linewidth=1.8)
        if name=='kather2016' and data.get('constrained_checkpoint_selection'):
            best=[v for v in data['independently_best_records'] if v['dataset']==name]
            ax.plot(range(1,5),[100*v['result']['original_test_dev']['accuracy'] for v in best],'x--',color='#777777',alpha=.8,label='Independent maxima')
            ax.plot([],[],'o-',color='#aa4499',label='Monotonic PT tuple');ax.legend(fontsize=7,loc='lower right')
        for k,y,c,z in zip(range(1,5),ys,correct,n):ax.annotate(f'{y:.2f}%\n{c}/{z}',(k,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.set(title=title,xlabel='Actual entrance top-k',ylabel='Development accuracy (%)',xticks=[1,2,3,4],xlim=(.7,4.3),ylim=(min(ys)-2,max(ys)+3));ax.grid(alpha=.2)
        trend=data['nondecreasing_by_k'][name]['strictly_increasing'];ax.text(.04,.04,'Strictly increasing: '+str(trend),transform=ax.transAxes,fontsize=9)
    fig.suptitle('Panel D: two main optical layers, same A OEO + single Linear\nOriginal TEST exposed development / seed17 / unequal search budgets; no causal top-k claim',fontsize=10);fig.tight_layout(rect=(0,0,1,.86));fig.savefig(out/'topk.svg');fig.savefig(out/'topk.png',dpi=160);plt.close(fig)
    svg=out/'topk.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
