"""Create comparison figures from finished runs; no model/data fitting."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(len(a.runs),3,figsize=(15,4*len(a.runs)),squeeze=False)
    for row,run in enumerate(a.runs):
        cfg=json.loads((run/'config.json').read_text());m=json.loads((run/'metrics.json').read_text());h=json.loads((run/'history.json').read_text());name=cfg.get('router_layout','slot_centers')+' / '+cfg.get('task_b_view','color_shift')
        ax=axes[row,0]; steps=np.arange(1,len(h)+1)
        for task,color in [('A','#245cb1'),('B','#d56a20')]: ax.plot(steps,[r['val_'+task]['accuracy']*100 for r in h],label=task,color=color)
        ax.axvspan(cfg['epochs_A']+.5,cfg['epochs_A']+cfg['epochs_warmup']+.5,color='gray',alpha=.16,label='warmup')
        ax.set(title=name+': validation curves',xlabel='Epoch across stages',ylabel='Accuracy (%)',ylim=(0,100));ax.legend();ax.grid(alpha=.2)
        ax=axes[row,1]; labels=['A before','A after','B after']; vals=[m['A_before']['accuracy'],m['A_all']['accuracy'],m['B_all']['accuracy']]
        ax.bar(labels,np.array(vals)*100,color=['#8aa9d5','#245cb1','#d56a20'])
        for i,v in enumerate(vals):ax.text(i,v*100+2,f'{v*100:.2f}%',ha='center')
        ax.set(title='Selected checkpoints',ylim=(0,100),ylabel='Accuracy (%)')
        ax=axes[row,2];old=[sum(m[t+'_all']['mean_route'][:4])*100 for t in ('A','B')];new=[sum(m[t+'_all']['mean_route'][4:8])*100 for t in ('A','B')]
        ax.bar(['A','B'],old,label='Old E1-E4',color='#245cb1');ax.bar(['A','B'],new,bottom=old,label='New E5-E8',color='#d56a20')
        ax.set(title='Routing power after B',ylabel='Mean power (%)',ylim=(0,110));ax.legend(loc='upper center',fontsize=8)
    fig.tight_layout();fig.savefig(a.out/'initial_results.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
