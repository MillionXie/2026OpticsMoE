from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    b=Path(__file__).resolve().parent;out=b/'reports/f_error_router_recovery_s17_20261101';history=json.loads((b/'runs/simulation/f_router_load_pretrain_s17_20261101/warmup/history.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,8))
    colors=['#4477aa','#ee6677','#228833','#aa3377']
    for ax,d in zip(axes[0],('RGB','SAR')):
        for i,color in enumerate(colors):ax.plot([r['epoch']for r in history],[100*r['actual_expert_usage']['TEST_'+d]['selection_slot_share'][i]for r in history],label='Expert '+str(i+1),color=color)
        ax.axhline(25,color='#888',linestyle=':');ax.set(title=d+' actual TEST selection slots',xlabel='Router-only epochs',ylabel='Share (%)',ylim=(0,55));ax.legend(fontsize=8);ax.grid(alpha=.2)
    history=json.loads((b/'runs/simulation/f_error_router_recovery_s17_20261101/teacher/history.json').read_text())
    for d,color in [('RGB','#4477aa'),('SAR','#ee6677')]:axes[1,0].plot([r['epoch']for r in history],[100*r['original_test_development_selection']['domains'][d]['accuracy']for r in history],label=d,color=color)
    axes[1,0].set(title='Balanced R frozen; E/G and head recover on errors',xlabel='Optical teacher epochs',ylabel='Original TEST development (%)');axes[1,0].axhline(65,color='#888',linestyle=':');axes[1,0].legend();axes[1,0].grid(alpha=.2)
    history=json.loads((b/'runs/simulation/f_error_teacher_focus_s17_20261101/teacher/history.json').read_text())
    for d,color in [('RGB','#4477aa'),('SAR','#ee6677')]:
        for split,style in [('train','--'),('original_test_development_selection','-')]:axes[1,1].plot([r['inner_iterations']for r in history],[100*r[split]['domains'][d]['accuracy']for r in history],label=d+(' TRAIN'if split=='train'else' TEST-dev'),color=color,linestyle=style)
    axes[1,1].set(title='Teacher TEST-error focus x8 (optics frozen)',xlabel='Head LBFGS internal steps',ylabel='Accuracy (%)');axes[1,1].axhline(65,color='#888',linestyle=':');axes[1,1].legend(fontsize=8);axes[1,1].grid(alpha=.2)
    fig.suptitle('TEST-exposed development / seed17 / identical two-main-layer Top2 optical teacher and student\nTeacher uses original TEST labels; student inherits teacher. High TEST-dev is not independent generalization.',fontsize=10);fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'recovery.svg');fig.savefig(out/'recovery.png',dpi=150);plt.close(fig);p=out/'recovery.svg';p.write_text('\n'.join(v.rstrip()for v in p.read_text().splitlines())+'\n')


if __name__=='__main__':main()
