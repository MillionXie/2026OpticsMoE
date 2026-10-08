"""CPU-only validation routing diagnostics for all locked MoE checkpoints.

No training, selection, main optical forward, or test images are used. Only the
constructor's final Module.cuda() is replaced by identity. The original encoder
and optical router run unchanged on CPU; results are diagnostics, not GPU replay.
Power variation is not, by itself, evidence of causal expert specialization.
"""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import bloodmnist_multiseed as m
from bloodmnist_multiseed import b,r,torch,np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--dataset',choices=['bloodmnist','kather2016'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.dataset=='kather2016':import kather2016_experiment
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    lock=r.read(a.run/'selection_lock.json');assert lock['sources']==m.sources() and lock['data_sha256']==r.sha(a.data)
    cfg=lock.get('config',r.read(a.run/'metadata.json').get('config'));entries=[e for e in lock['entries'] if e['result']['arch'].startswith('moe')]
    assert len(entries)==18 and len({(e['result']['arch'],e['result']['depth'],e['result']['seed']) for e in entries})==18
    a.out.mkdir(parents=True,exist_ok=False)
    with np.load(a.data,allow_pickle=False) as z:
        images=z['val_images'].copy();labels=z['val_labels'].reshape(-1).copy();ids=z['val_ids'].copy() if 'val_ids' in z else np.array([f'val_{i}' for i in range(len(labels))])
    assert images.dtype==np.uint8 and set(labels)==set(range(8))
    x=torch.from_numpy(images.transpose(0,3,1,2).copy()).float()/255
    with torch.no_grad():encoded=b.encode(x)
    del images,x
    records=[];details={}
    for entry in entries:
        v=entry['result'];folder=Path(entry['folder']);assert r.sha(folder/'best_checkpoint.pt')==v['checkpoint_sha256'];r.setseed(v['seed'])
        with patch.object(torch.nn.Module,'cuda',lambda self,*args,**kwargs:self):model=m.build(v['arch'],v['depth'],cfg)
        model.eval();router=model.net.prompt.router_network;assert router.top_k==9
        initial_phase=router.slm.raw_phase.detach().clone()
        def outputs():
            with torch.no_grad():pr=torch.cat([router(encoded[i:i+128])['probabilities'] for i in range(0,len(encoded),128)]).numpy()
            assert pr.shape==(len(labels),9) and np.isfinite(pr).all() and (pr>=0).all() and np.allclose(pr.sum(1),1,atol=1e-6)
            power=pr**2/(pr**2).sum(1,keepdims=True);return pr,power
        _,initial=outputs();ck=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);pr,power=outputs()
        mean=power.mean(0);class_means=np.stack([power[labels==i].mean(0) for i in range(8)]);dominant=np.bincount(power.argmax(1),minlength=9)/len(power)
        entropy=-(power*np.log(np.maximum(power,1e-30))).sum(1)/np.log(9)
        row=dict(model=v['name'],arch=v['arch'],depth=v['depth'],seed=v['seed'],validation_samples=len(labels),all_nine_branches_positive=bool((power>0).all()),mean_effective_branch_count=float((1/(power**2).sum(1)).mean()),mean_normalized_power_entropy=float(entropy.mean()),mean_max_branch_power=float(power.max(1).mean()),mean_l1_distance_to_mean_power=float(np.abs(power-mean).sum(1).mean()),mean_l1_power_change_from_initial=float(np.abs(power-initial).sum(1).mean()),largest_argmax_branch_fraction=float(dominant.max()),router_phase_rms_update=float((router.slm.raw_phase.detach()-initial_phase).square().mean().sqrt()))
        records.append(row);details[v['name']]=dict(summary=row,mean_power=mean.tolist(),power_std_across_images=power.std(0).tolist(),class_mean_power=class_means.tolist(),largest_power_branch_frequencies=dominant.tolist(),checkpoint_sha256=v['checkpoint_sha256'])
        np.savez_compressed(a.out/(v['name']+'.npz'),sample_ids=ids,labels=labels,probabilities=pr,branch_power=power,initial_branch_power=initial)
        print(json.dumps(row),flush=True);del model,router,ck
    assert not torch.cuda.is_initialized()
    with (a.out/'routing_summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    r.save(a.out/'routing_details.json',details)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,2,figsize=(9,10),layout='constrained');vmax=max(max(max(row) for row in d['class_mean_power']) for d in details.values() if d['summary']['depth']==6)
    for i,seed in enumerate([17,27,37]):
        for j,arch in enumerate(['moe_nooeo','moe']):
            ax=axes[i,j];d=details[f'{arch}_L6_seed{seed}'];im=ax.imshow(d['class_mean_power'],vmin=0,vmax=vmax,cmap='viridis',aspect='auto');ax.set_xticks(range(9),range(1,10));ax.set_yticks(range(8),cfg['classes']);ax.set_title(('MoE + OEO' if arch=='moe' else 'MoE')+f', seed {seed}');ax.set_xlabel('Physical branch (all active)')
    fig.colorbar(im,ax=axes,shrink=.7,label='Mean input power fraction');fig.suptitle('Six layers: validation class-conditional routing\nDescriptive diagnostic; does not establish expert specialization')
    for ext in ['png','pdf','svg']:fig.savefig(a.out/('routing_class_means_L6.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    r.save(a.out/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),auditor_sha256=r.sha(__file__),sources=m.sources(),selection_lock_sha256=r.sha(a.run/'selection_lock.json'),data_sha256=r.sha(a.data),scope=__doc__,device='CPU',torch=torch.__version__,visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),test_read=False))
    r.save(a.out/'status.json',dict(state='complete',models=len(records),time=r.now()))


if __name__=='__main__':main()
