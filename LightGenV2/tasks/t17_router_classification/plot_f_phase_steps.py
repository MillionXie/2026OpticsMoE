"""Compare recorded TRAIN step acceptance with actual validation trajectories."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    task=Path(__file__).resolve().parent;out=task/'reports/f_backtracking_s17_20261031'
    paired=json.loads((task/'runs/simulation/f_backtracking_s17_20261031/paired/history.json').read_text())
    adam=json.loads((task/'runs/simulation/f_full_TRAIN_phase_s17_20261031/full/history.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    epoch=[r['epoch'] for r in paired];before=[r['TRAIN_phase_backtracking']['baseline_objective'] for r in paired];after=[r['TRAIN_phase_backtracking']['objective_after'] for r in paired]
    axs[0].vlines(epoch,after,before,color='#888',linewidth=2)
    axs[0].scatter(epoch,before,color='#e45756',marker='x',label='Before phase step')
    axs[0].scatter(epoch,after,color='#4c78a8',marker='o',label='Accepted canonical step')
    axs[0].set(xlabel='Additional round',ylabel='TRAIN CE + KD + anchor',title='Within each pair: identical TRAIN views and head')
    axs[0].text(.02,.03,'Different rounds use different random D4 view sets.',transform=axs[0].transAxes,fontsize=8)
    axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    paired_state='complete' if (task/'runs/simulation/f_backtracking_s17_20261031/paired/result.json').exists() else 'in progress'
    for rows,label,color in [(adam,'Full-TRAIN Adam (complete)','#e45756'),(paired,f'Paired backtracking ({paired_state})','#4c78a8')]:
        axs[1].plot([0]+[r['epoch'] for r in rows],[52.4]+[100*r['validation']['accuracy'] for r in rows],'.-',color=color,label=label)
    axs[1].axhline(52.4,color='black',linestyle='--',linewidth=1,label='Protected D4 parent VAL')
    for r in paired:
        if 'TRAIN_head_block' in r:axs[1].axvline(r['epoch'],color='#888',alpha=.3,linewidth=1)
    axs[1].set(xlabel='Additional rounds within each separate branch',ylabel='VAL accuracy (%)',title='VAL never enters phase step acceptance',ylim=(10,60))
    axs[1].legend(fontsize=8,loc='lower right');axs[1].grid(alpha=.2)
    fig.suptitle('F seed17: training-coordinate comparison; no inference-model change',fontsize=12)
    fig.tight_layout();fig.savefig(out/'phase_steps.svg');fig.savefig(out/'phase_steps.png',dpi=150);plt.close(fig)
    path=out/'phase_steps.svg';path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')


if __name__=='__main__':main()
