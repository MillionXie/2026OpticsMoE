"""Joint F domain curves and paired head-block objectives, without interim TEST scores."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    task=Path(__file__).resolve().parent;run=task/'runs/simulation/f_joint_dual_teacher_s17_20261101/joint';out=task/'reports/f_joint_dual_teacher_s17_20261101'
    history=json.loads((run/'history.json').read_text());parent=json.loads((task/'runs/simulation/f_backtracking_s17_20261031/paired/result.json').read_text())
    result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else None
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,domain in zip(axes[:2],('RGB','SAR')):
        ax.plot([0]+[r['epoch'] for r in history],[100*parent['val']['domains'][domain]['accuracy']]+[100*r['validation']['domains'][domain]['accuracy'] for r in history],label='VAL',color='#222222')
        clean=[r for r in history if 'train' in r]
        ax.plot([0]+[r['epoch'] for r in clean],[100*parent['train']['domains'][domain]['accuracy']]+[100*r['train']['domains'][domain]['accuracy'] for r in clean],label='Clean TRAIN',linestyle='--',color='#4477aa')
        ax.axhline(70,linestyle=':',color='#aa7733',label='70% goal')
        if result:
            ax.scatter(result['selected_epoch'],100*result['original_test_dev']['domains'][domain]['accuracy'],marker='*',s=110,color='#bb4444',label='Locked development TEST')
        ax.set(title=domain+' / one joint LG',xlabel='Additional epochs',ylabel='Accuracy (%)',ylim=(0,100));ax.grid(alpha=.2);ax.legend(fontsize=8)
    blocks=[r for r in history if 'TRAIN_head_block' in r]
    for i,r in enumerate(blocks):
        b=r['TRAIN_head_block'];assert b['objective_after']<=b['objective_before']+1e-8
        axes[2].plot([r['epoch'],r['epoch']],[b['objective_before'],b['objective_after']],color='#888888')
        axes[2].scatter(r['epoch'],b['objective_before'],marker='o',color='#cc6677',label='Before head block' if i==0 else None)
        axes[2].scatter(r['epoch'],b['objective_after'],marker='v',color='#228833',label='After head block' if i==0 else None)
    axes[2].set(title='Same-feature head refits',xlabel='Head-block epoch',ylabel='Mixed CE / KD + anchor objective');axes[2].grid(alpha=.2);axes[2].legend(fontsize=8)
    fig.suptitle('F seed17: original TEST exposed development; one joint student\nInterim TEST is not evaluated. Paired head-block losses are not cross-epoch comparisons.',fontsize=11)
    fig.tight_layout();out.mkdir(exist_ok=True);fig.savefig(out/'convergence.svg');fig.savefig(out/'convergence.png',dpi=150);plt.close(fig)
    p=out/'convergence.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')


if __name__=='__main__':main()
