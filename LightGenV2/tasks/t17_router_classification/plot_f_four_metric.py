"""Four original clean metrics on separate teacher/student axes; no seed SD."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(p):return json.loads(p.read_text())if p.exists()else None

def main():
    task=Path(__file__).resolve().parent;root=task/'runs/simulation/f_four_metric_s17_20261101';out=task/'reports/f_four_metric_s17_20261101';fig,axes=plt.subplots(1,3,figsize=(16,4.8));colors={'RGB':'#2563eb','SAR':'#e28a20'}
    for ax,stage in zip(axes[:2],('teacher','student')):
        h=read(root/stage/'history.json')or [];r=read(root/stage/'result.json');meta=read(root/stage/'metadata.json')
        for split,key,style in [('TRAIN','train','-'),('TEST dev','original_test_development_selection','--')]:
            for domain in ('RGB','SAR'):
                ax.plot([v['epoch']for v in h],[100*v[key]['domains'][domain]['accuracy']for v in h],style,color=colors[domain],label=split+' '+domain,lw=1.7)
                if r:
                    finalkey='train'if split=='TRAIN'else'original_test_dev';ax.scatter([r['selected_epoch']],[100*r[finalkey]['domains'][domain]['accuracy']],marker='*',s=90,color=colors[domain],edgecolors='black',lw=.5,zorder=5)
        ax.axhline(65,color='#444',ls=':',label='65% target');ax.set(title=stage.title()+' (separate budget)',xlabel='Optical epoch',ylabel='Original clean accuracy (%)',ylim=(45,85));ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2,loc='lower right')
    audit=read(out/'student_model_audit.json');ax=axes[2]
    if audit:
        x=np.arange(4)
        for offset,domain in [(-.18,'RGB'),(.18,'SAR')]:ax.bar(x+offset,100*np.array(audit['parts']['test']['domains'][domain]['usage']['selection_slot_share']),width=.36,color=colors[domain],label=domain)
        ax.axhline(25,color='#444',ls=':');ax.set_xticks(x,['E1','E2','E3','E4']);ax.set_ylim(0,40);ax.legend();ax.set_ylabel('Share of actual Top2 selections (%)')
    else:ax.text(.5,.5,'Final same-PT load audit pending',ha='center',va='center',transform=ax.transAxes);ax.set_xticks([]);ax.set_yticks([])
    ax.set_title('Frozen balanced optical router');fig.suptitle('Same optical model / four original metrics; TEST-exposed development, seed 17',fontsize=12);fig.text(.01,.015,'Stars: complete selected PT. TRAIN: 3000/domain; TEST: 1000/domain. TEST used in teacher training and checkpoint selection. No independent test claim.',fontsize=9);fig.tight_layout(rect=(0,.055,1,.94));fig.savefig(out/'four_metrics.svg');fig.savefig(out/'four_metrics.png',dpi=130);plt.close(fig)

if __name__=='__main__':main()
