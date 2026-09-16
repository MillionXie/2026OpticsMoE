"""Export routing/ablation figures and independently recompute distribution metrics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def heat(ax,matrix,rows,title,vmin=0,vmax=100,cmap='Blues',signed=False):
    im=ax.imshow(matrix,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set_xticks(range(4),['E1','E2','E3','E4']);ax.set_yticks(range(len(rows)),rows)
    ax.set_title(title,pad=12)
    for (y,x),value in np.ndenumerate(matrix):
        text=f'{value:+.1f}' if signed else f'{value:.1f}'
        color='white' if (value>(vmin+vmax)/2 if not signed else abs(value)>.6*max(abs(vmin),abs(vmax))) else '#17202A'
        ax.text(x,y,text,ha='center',va='center',color=color,fontsize=10)
    return im


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    root=a.run;dist=json.loads((root/'routing_distribution.json').read_text());metrics=json.loads((root/'results.json').read_text())
    with (root/'sample_routes.csv').open() as f:rows=list(csv.DictReader(f))
    q=np.array([[float(r[f'q_E{i+1}']) for i in range(4)] for r in rows]);domain=np.array([int(r['domain']) for r in rows]);label=np.array([int(r['label']) for r in rows])
    assert len({r['sample_id'] for r in rows})==2000 and np.allclose(q.sum(1),1,atol=1e-6)
    means=[];largest=[]
    for d in (0,1):
        z=q[domain==d];mean=z.mean(0);counts=np.bincount(z.argmax(1),minlength=4)
        assert np.allclose(mean,dist['domains'][str(d)]['mean_power'],atol=1e-6)
        assert counts.tolist()==dist['domains'][str(d)]['largest_share']
        means.append(mean*100);largest.append(counts/len(z)*100)
    for name,result in metrics.items():
        with (root/(name+'_predictions.csv')).open() as f:predictions=list(csv.DictReader(f))
        assert [x['sample_id'] for x in predictions]==[x['sample_id'] for x in rows]
        pred=np.array([int(x['prediction']) for x in predictions])
        assert float((pred==label).mean())==result['overall']['accuracy']
        for d in (0,1):assert float((pred[domain==d]==label[domain==d]).mean())==result['domains'][str(d)]['accuracy']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    heat(axes[0,0],np.array(means),['RGB','SAR'],'Mean input power share (%)',vmax=45)
    heat(axes[0,1],np.array(largest),['RGB','SAR'],'Samples with largest share (%)',vmax=70)
    colors=['#247BA0','#F3A712','#8E6CBB','#D64F61']
    for d,ax in enumerate(axes[1]):
        boxes=ax.boxplot([q[domain==d,k]*100 for k in range(4)],labels=['E1','E2','E3','E4'],patch_artist=True,showfliers=False,whis=(5,95))
        for box,color in zip(boxes['boxes'],colors):box.set_facecolor(color);box.set_alpha(.7)
        ax.set_ylim(0,100);ax.set_ylabel('Input power share (%)');ax.axhline(25,color='gray',ls='--',lw=1)
        ax.set_title(['RGB: sample distribution','SAR: sample distribution'][d]);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen dynamic MoE | best epoch 15 | validation: 1,000 RGB + 1,000 SAR',fontsize=14)
    fig.supxlabel('All four experts are active. Boxes: 25-75%; whiskers: 5-95%; dashed line: equal power.',fontsize=10)
    for extension in ['png','svg']:fig.savefig(root/('routing_distribution.'+extension))
    plt.close(fig)
    classes=['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']
    fig,axes=plt.subplots(1,2,figsize=(10,7),layout='constrained')
    for d,ax in enumerate(axes):
        matrix=np.array([dist['domain_class'][f'{d}:{c}']['mean_power'] for c in range(10)])*100
        heat(ax,matrix,classes,['RGB: mean power (%)','SAR: mean power (%)'][d],vmax=60)
    fig.suptitle('Routing by class | 100 validation samples per class and domain')
    for extension in ['png','svg']:fig.savefig(root/('routing_by_class.'+extension))
    plt.close(fig)
    single=np.array([[metrics[f'only_E{i+1}']['domains'][str(d)]['accuracy']*100 for i in range(4)] for d in (0,1)])
    drop=np.array([[(metrics['native']['domains'][str(d)]['accuracy']-metrics[f'drop_E{i+1}']['domains'][str(d)]['accuracy'])*100 for i in range(4)] for d in (0,1)])
    fig,axes=plt.subplots(2,1,figsize=(9,6),layout='constrained')
    heat(axes[0],single,['RGB','SAR'],'One expert receives all input power: accuracy (%)',vmax=50)
    bound=max(1,float(abs(drop).max()))
    heat(axes[1],drop,['RGB','SAR'],'Accuracy decrease after removing expert (percentage points)',vmin=-bound,vmax=bound,cmap='RdBu_r',signed=True)
    fig.suptitle('Frozen-weight interventions | input power renormalized to 1')
    fig.supxlabel('Coherent mixing changes under intervention; differences are not additive expert contributions.',fontsize=9)
    for extension in ['png','svg']:fig.savefig(root/('expert_interventions.'+extension))
    plt.close(fig)
    (root/'independent_verification.json').write_text(json.dumps(dict(passed=True,route_rows=2000,policies=len(metrics),
         domain_means_and_largest_counts_verified=True,all_prediction_accuracies_verified=True),indent=2))
    print(json.dumps(dict(means=means,largest=largest,single_expert_accuracy=single.tolist(),drop_accuracy_decrease_pp=drop.tolist()),default=lambda x:x.tolist()))


if __name__=='__main__':main()
