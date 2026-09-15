"""Sealed evaluation after every training/selection decision has finished."""
import contextlib,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset,ConcatDataset
from .settings import load_settings
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.data import prepare_cifar10
from .backbone import load_backbone
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import seed_everything
from . import model as M
from .data import atomic_json,official_b,validation_b,DATA_ROOT,CORRUPTIONS
from .engine import evaluate,evaluate_corrupted,base,autocast,export_visuals

def ci(values,seed=4421):
    # One sampling unit = original image ID. First average correlated corruptions.
    values=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(seed)
    boot=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(2000)])
    return [float(v) for v in np.quantile(boot,[.025,.975])]

def correct_vectors(folder):
    a=np.load(folder/'clean.npz');a=(a['labels']==a['predictions']).astype(float)
    b=[]
    for kind in CORRUPTIONS:
        for sev in range(1,6):
            z=np.load(folder/f'{kind}_{sev}.npz');b.append(z['labels']==z['predictions'])
    return a,np.stack(b).astype(float).mean(0)

def mechanism(loaded,r,h,s,bundle,source,final,out):
    count=[0]*10;idx=[]
    for i in range(len(bundle.validation)):
        _,y=bundle.validation[i]
        if count[y]<20:idx.append(i);count[y]+=1
        if min(count)>=20:break
    clean=Subset(bundle.validation,idx)
    b=ConcatDataset([Subset(ds,list(range(min(20,len(ds))))) for _,_,ds in validation_b(DATA_ROOT)])
    original=M.clone(r,h);result={}
    modes=['learned','source_router','initial_b_experts','zero_b_input'] if r.transfer_architecture=='moe' else ['learned','initial_all_phases']
    for mode in modes:
        M.restore(r,h,original)
        with torch.no_grad():
            for name,p in M.named(r,h).items():
                group=M.group_of(name)
                if (mode=='initial_b_experts' and group=='expert_b') or (mode=='initial_all_phases' and name.endswith('raw_phase')):
                    p.copy_(final['origin_phases'][name].to(p.device))
                if mode=='source_router' and group=='router':
                    pref,key=name.split('.',1);p.copy_(source[pref][key].to(p.device))
        if mode=='zero_b_input':
            for _,sur in M.surrogates(r):sur.core.optical_branch.core.router.suppress_b=True
        try:
            aa=evaluate(loaded,r,h,clean,s,oracle_domain=0 if mode=='oracle_groups' else None)
            bb=evaluate(loaded,r,h,b,s,oracle_domain=1 if mode=='oracle_groups' else None)
            result[mode]=dict(validation_clean=aa,validation_corrupted=bb)
        finally:
            if r.transfer_architecture=='moe':
                for _,sur in M.surrogates(r):sur.core.optical_branch.core.router.suppress_b=False
    M.restore(r,h,original)
    result['scope']='Fixed validation intervention, 200 clean and 20 per 45 corruption conditions; not official test or retraining. zero_b_input reduces power without renormalization. No oracle task-mask diagnostic is used in Vision-only.'
    atomic_json(out/'mechanism.json',result)
    return result

def main():
    from .evaluate import main as implementation
    return implementation()


def plot_results(out,matrix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for name in ['moe_A','d2nn_A','moe_reserved_B','moe_all_B','d2nn_d2nn_B']:
        if not (out/name/'metrics.jsonl').exists():continue
        history=[json.loads(x) for x in (out/name/'metrics.jsonl').read_text().splitlines()]
        ax=axs[0] if name.endswith('_A') else axs[1]
        ax.plot([v['epoch'] for v in history],[v['validation_a']['accuracy']*100 for v in history],label=name+' clean')
        if name.endswith('_B'):ax.plot([v['epoch'] for v in history],[v['validation_b']['accuracy']*100 for v in history],linestyle='--',label=name+' corrupted')
    for ax in axs:ax.set_xlabel('Epoch');ax.set_ylabel('Validation accuracy (%)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'validation_curves.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(3);w=.25
    for i,(key,label) in enumerate([('A_after_B','Clean'),('B_after_B','Corrupted'),('equal_domain_average','Equal-domain mean')]):
        ax.bar(x+(i-1)*w,[np.nan if row[key] is None else row[key]*100 for row in matrix],w,label=label)
    ax.set_xticks(x,[row['model']+'\n'+str(row['B_epochs_completed'])+'/'+str(row['B_epochs_planned'])+' B epochs\n'+row['training_status'] for row in matrix],fontsize=8);ax.set_ylabel('Official test accuracy (%)');ax.legend();fig.tight_layout();fig.savefig(out/'final_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
