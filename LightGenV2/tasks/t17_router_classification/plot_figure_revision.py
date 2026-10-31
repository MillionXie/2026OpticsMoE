"""Scientific convergence plots; TEST points only after checkpoint lock."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(path.read_text()) if path.exists() else None


def main():
    task=Path(__file__).resolve().parent;root=task/'runs/simulation/figure_revision_s17_20261031';out=task/'reports/figure_revision_schedule_20261031'
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    jobs=[('panel_b_linear_joint4_s17_20261031','B: LG main4 + Linear'),('panel_b_linear_joint6_s17_20261031','B: LG main6 + Linear'),('panel_f_lg_s17_20261031','F: LG main2 + Linear'),('panel_f_d2nn_s17_20261031','F: D2NN3 + OEO, CCD readout')]
    for ax,(name,title) in zip(axes.flat,jobs):
        history=read(root/name/'history.json') or [];result=read(root/name/'result.json');cfg=read(task/'configs'/f'{name}.json')
        if cfg.get('parent_checkpoint'):
            parent=read((task/cfg['parent_checkpoint']).with_name('result.json'))
            ax.scatter([0],[100*parent['val']['accuracy']],color='black',s=20,label='Parent VAL')
        if history:
            ax.plot([r['epoch'] for r in history],[100*r['validation']['accuracy'] for r in history],color='black',label='VAL')
            train=[r for r in history if 'train' in r]
            ax.plot([r['epoch'] for r in train],[100*r['train']['accuracy'] for r in train],color='#377eb8',marker='.',label='Clean TRAIN')
            if name.startswith('panel_f'):
                for domain,color in [('RGB','#e5a14b'),('SAR','#55a868')]:
                    ax.plot([r['epoch'] for r in history],[100*r['validation']['domains'][domain]['accuracy'] for r in history],color=color,alpha=.65,label=domain+' VAL')
        if result:
            ax.scatter([result['selected_epoch']],[100*result['original_test_dev']['accuracy']],marker='*',s=140,color='#c44e52',label='Locked original TEST development',zorder=5)
            ax.scatter([result['selected_epoch']],[100*result['train']['accuracy']],marker='D',s=36,color='#377eb8',zorder=5,label='Selected model TRAIN')
            ax.axvline(result['selected_epoch'],color='grey',linestyle=':',linewidth=.8)
        ax.set(title=title,xlabel='Additional epochs from parent' if name.startswith('panel_b') else 'Epochs from scratch',ylabel='Accuracy (%)',ylim=(0,100));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Seed17 | Single Linear LG revision | ORIGINAL_TEST_EXPOSED_DEVELOPMENT',fontsize=12)
    fig.savefig(out/'convergence.svg');fig.savefig(out/'convergence.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
