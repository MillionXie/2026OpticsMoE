"""Plot completed F TRAIN/VAL curves, locked domain predictions and real Top2 power."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    task=Path(__file__).resolve().parent;out=task/'reports/f_teacher_s17_20261031'
    jobs=[('Alternating control',task/'runs/simulation/f_alternating_s17_20261031/control'),
          ('Domain auxiliary',task/'runs/simulation/f_alternating_s17_20261031/domain'),
          ('Teacher KD 0.25',task/'runs/simulation/f_teacher_s17_20261031/025'),
          ('Historical masks',task/'runs/simulation/f_historical_warm_s17_20261031/warm')]
    fig,axs=plt.subplots(1,3,figsize=(15,4.8));colors=['#4c78a8','#f58518','#54a24b','#b279a2']
    results=[]
    for (name,path),color in zip(jobs,colors):
        history=json.loads((path/'history.json').read_text());result=json.loads((path/'result.json').read_text());results.append(result)
        axs[0].plot([r['epoch'] for r in history],[100*r['validation']['accuracy'] for r in history],color=color,label=name+' VAL')
        train=[r for r in history if 'train' in r]
        axs[0].plot([r['epoch'] for r in train],[100*r['train']['accuracy'] for r in train],color=color,linestyle='--',alpha=.7)
        axs[0].scatter(result['selected_epoch'],100*result['val']['accuracy'],color=color,marker='D',s=20)
    axs[0].set(xlabel='Additional epochs within each branch',ylabel='Accuracy (%)',title='Solid: VAL; dashed: clean TRAIN',ylim=(35,75))
    axs[0].legend(fontsize=8,loc='upper left');axs[0].grid(alpha=.2)
    baseline=json.loads((task/'runs/simulation/figure_revision_s17_20261031/panel_f_d2nn_s17_20261031/result.json').read_text())
    labels=['D2NN 3L','Control','Domain aux','Teacher KD','Old masks'];all_results=[baseline]+results;x=np.arange(len(all_results))
    for domain,offset,color in [('RGB',-.18,'#4c78a8'),('SAR',.18,'#f2b65d')]:
        values=[100*r['original_test_dev']['domains'][domain]['accuracy'] for r in all_results]
        axs[1].bar(x+offset,values,.35,label=domain,color=color)
        for px,value in zip(x+offset,values):axs[1].text(px,value+.7,f'{value:.1f}',ha='center',fontsize=8)
    axs[1].set(xticks=x,xticklabels=labels,ylabel='Accuracy (%)',title='Locked original-TEST development',ylim=(0,85))
    axs[1].legend(fontsize=8);axs[1].tick_params(axis='x',labelrotation=18)
    powers=np.asarray([r['original_test_dev']['router']['mean_power'] for r in results]);bottom=np.zeros(len(results))
    for expert,color in enumerate(['#4c78a8','#f2b65d','#54a24b','#b279a2']):
        axs[2].bar(np.arange(len(results)),powers[:,expert]*100,bottom=bottom,color=color,label=f'E{expert+1}')
        bottom+=powers[:,expert]*100
    axs[2].set(xticks=np.arange(len(results)),xticklabels=labels[1:],ylabel='Mean entrance power (%)',title='Actual inference Top2, not dense targets',ylim=(0,105))
    axs[2].legend(ncol=4,fontsize=8,loc='upper center');axs[2].tick_params(axis='x',labelrotation=18)
    fig.suptitle('F: one joint RGB/SAR LG; seed17; original TEST exposed development',fontsize=12)
    fig.tight_layout();fig.savefig(out/'convergence.svg');fig.savefig(out/'convergence.png',dpi=160);plt.close(fig)
    svg=out/'convergence.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')


if __name__=='__main__':main()
