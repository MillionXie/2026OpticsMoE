"""All completed single-domain LG branches and paired target-domain results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    task=Path(__file__).resolve().parent;out=task/'reports/f_cross_domain_s17_20261031'
    fig,axes=plt.subplots(2,2,figsize=(13,8));colors=['#4c78a8','#f58518','#54a24b']
    for row,domain in enumerate(['RGB','SAR']):
        jobs=[('Initial KD',task/'runs/simulation/f_cross_domain_s17_20261031'/f'LG_{domain}'),('Source-TEST KD',task/'runs/simulation/f_source_TEST_KD_s17_20261031'/domain),('Hidden KD',task/'runs/simulation/f_semantic_KD_s17_20261031'/domain)];results=[]
        for (name,path),color in zip(jobs,colors):
            h=json.loads((path/'history.json').read_text());r=json.loads((path/'result.json').read_text());results.append(r);clean=[x for x in h if 'train' in x]
            axes[row,0].plot([x['epoch'] for x in h],[100*x['validation']['accuracy'] for x in h],color=color,label=name+' VAL')
            axes[row,0].plot([x['epoch'] for x in clean],[100*x['train']['accuracy'] for x in clean],color=color,linestyle='--',alpha=.7)
            axes[row,0].scatter(r['selected_epoch'],100*r['train']['accuracy'],color=color,marker='D',s=25)
        axes[row,0].set(title=f'{domain}-trained LG: separate branches',xlabel='Additional epochs within each branch',ylabel='Accuracy (%)',ylim=(0,100));axes[row,0].grid(alpha=.2);axes[row,0].legend(fontsize=8)
        baseline=json.loads((task/'runs/simulation/f_cross_domain_s17_20261031'/f'D2NN_{domain}'/'result.json').read_text());all_results=[baseline]+results;x=np.arange(4)
        for target,offset,color in [('RGB',-.18,'#4c78a8'),('SAR',.18,'#f2b65d')]:
            values=[100*r['original_test_dev']['domains'][target]['accuracy'] for r in all_results];axes[row,1].bar(x+offset,values,.35,color=color,label='TEST '+target)
            for center,value in zip(x+offset,values):axes[row,1].text(center,value+.7,f'{value:.1f}',ha='center',fontsize=8)
        axes[row,1].set(title=f'{domain}-trained model: same PT for both targets',xticks=x,xticklabels=['D2NN 60','Initial KD','TEST KD','Hidden KD'],ylabel='Accuracy (%)',ylim=(0,100));axes[row,1].legend(fontsize=8)
    fig.suptitle('F seed17: original TEST exposed development\nSolid: source VAL; dashed: clean source TRAIN; diamonds: selected TRAIN',fontsize=12)
    fig.tight_layout();fig.savefig(out/'refinement_stages.svg');fig.savefig(out/'refinement_stages.png',dpi=150);plt.close(fig)
    p=out/'refinement_stages.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')


if __name__=='__main__':main()
