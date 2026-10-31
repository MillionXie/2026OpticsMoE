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
    branches=[('Dual teacher',history,parent,result,'#4477aa')]
    ste=task/'runs/simulation/f_joint_route_STE_s17_20261101/joint'
    if (ste/'history.json').exists():
        sh=json.loads((ste/'history.json').read_text());sr=json.loads((ste/'result.json').read_text()) if (ste/'result.json').exists() else None
        branches.append(('Surrogate backward',sh,result,sr,'#cc6677'))
    projected=task/'runs/simulation/f_joint_router_projection_s17_20261101/joint'
    if (projected/'history.json').exists():
        ph=json.loads((projected/'history.json').read_text());pr=json.loads((projected/'result.json').read_text()) if (projected/'result.json').exists() else None
        branches.append(('Router projection',ph,sr,pr,'#228833'))
    sgd=task/'runs/simulation/f_joint_router_sgd_s17_20261101/joint'
    if (sgd/'result.json').exists() and (task/'reports/f_joint_router_sgd_s17_20261101/model_audit.json').exists():
        gh=json.loads((sgd/'history.json').read_text());gr=json.loads((sgd/'result.json').read_text())
        branches.append(('Router angle SGD',gh,sr,gr,'#aa7733'))
    shared=task/'runs/simulation/f_joint_teacher_student_TEST_s17_20261101/joint'
    if (shared/'result.json').exists() and (task/'reports/f_joint_teacher_student_TEST_s17_20261101/model_audit.json').exists():
        hh=json.loads((shared/'history.json').read_text());hr=json.loads((shared/'result.json').read_text())
        branches.append(('Shared teacher TEST-selected',hh,sr,hr,'#aa4499'))
    fig,axes=plt.subplots(1,3,figsize=(15,5.5))
    for ax,domain in zip(axes[:2],('RGB','SAR')):
        for name,h,prior,final,color in branches:
            ax.plot([0]+[r['epoch'] for r in h],[100*prior['val']['domains'][domain]['accuracy']]+[100*r['validation']['domains'][domain]['accuracy'] for r in h],label=name+' VAL',color=color)
            clean=[r for r in h if 'train' in r]
            ax.plot([0]+[r['epoch'] for r in clean],[100*prior['train']['domains'][domain]['accuracy']]+[100*r['train']['domains'][domain]['accuracy'] for r in clean],label=name+' TRAIN',linestyle='--',color=color,alpha=.7)
            if final:ax.scatter(final['selected_epoch'],100*final['original_test_dev']['domains'][domain]['accuracy'],marker='*',s=110,color=color,label=name+' locked DEV')
        ax.axhline(70,linestyle=':',color='#aa7733',label='70% goal')
        ax.set(title=domain+' / one joint LG',xlabel='Additional epochs within each branch',ylabel='Accuracy (%)',ylim=(0,100));ax.grid(alpha=.2)
    for name,h,prior,final,color in branches:
        blocks=[r for r in h if 'TRAIN_head_block' in r]
        for i,r in enumerate(blocks):
            b=r['TRAIN_head_block'];assert b['objective_after']<=b['objective_before']+1e-8
            axes[2].plot([r['epoch'],r['epoch']],[b['objective_before'],b['objective_after']],color=color,alpha=.5)
            axes[2].scatter(r['epoch'],b['objective_before'],marker='o',color=color,label=name+' before' if i==0 else None)
            axes[2].scatter(r['epoch'],b['objective_after'],marker='v',color=color,label=name+' after' if i==0 else None)
    axes[2].set(title='Same-feature head refits',xlabel='Head-block epoch within each branch',ylabel='Mixed CE / KD + anchor objective');axes[2].grid(alpha=.2);axes[2].legend(fontsize=7)
    fig.suptitle('F seed17: original TEST exposed development; separate training branches\nSolid VAL, dashed clean TRAIN; final DEV stars. New shared-teacher branch selects on TEST every epoch.',fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7)
    fig.tight_layout(rect=(0,.20,1,.94));out.mkdir(exist_ok=True);fig.savefig(out/'convergence.svg');fig.savefig(out/'convergence.png',dpi=150);plt.close(fig)
    p=out/'convergence.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')


if __name__=='__main__':main()
