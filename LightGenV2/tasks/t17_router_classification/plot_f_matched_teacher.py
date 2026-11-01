"""Separate axes for optical epochs and frozen-optics head iterations."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    task=Path(__file__).resolve().parent;out=task/'reports/f_matched_optical_teacher_s17_20261101'
    roots=[task/'runs/simulation'/name for name in ['f_matched_optical_teacher_s17_20261101','f_matched_teacher_canonical_attempt2_s17_20261101','f_matched_canonical_phase_s17_20261101']]
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharey=True)
    for column,(root,title)in enumerate(zip(roots,['D4 / canonical optical training','Canonical head refit (optics frozen)','Canonical optical continuation'])):
        for row,stage in enumerate(('teacher','student')):
            ax=axes[row,column];folder=root/stage
            history=json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
            result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
            xs=[r['inner_iterations'] if column==1 else r['epoch'] for r in history]
            for name,color in [('RGB','#4477aa'),('SAR','#cc6677')]:
                ys=[100*r['original_test_development_selection']['domains'][name]['accuracy'] for r in history]
                if ys:ax.plot(xs,ys,label=name,color=color,linewidth=1.4)
                if result:
                    selected=result.get('selected_epoch',result.get('selected_iteration',0));matched=next((r for r in history if r.get('epoch',r.get('iteration'))==selected),None)
                    x=matched['inner_iterations'] if column==1 and matched else selected
                    ax.scatter([x],[100*result['original_test_dev']['domains'][name]['accuracy']],marker='*',s=95,color=color,zorder=4)
            ax.axhline(65,color='#888888',linestyle=':',label='65% target');ax.set(title=stage+': '+title,xlabel='LBFGS internal steps'if column==1 else 'Optical epochs',ylabel='Original TEST development (%)',ylim=(40,76));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Same 2-layer Top2 optical teacher and student, seed17\nTeacher: TRAIN+original TEST labels; student inherits teacher and uses TEST soft targets only. Not independent TEST.',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.92));out.mkdir(exist_ok=True);fig.savefig(out/'convergence.svg');fig.savefig(out/'convergence.png',dpi=150);plt.close(fig)
    p=out/'convergence.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')


if __name__=='__main__':main()
