"""Frozen-checkpoint routing distributions and controlled inference interventions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import torch
from models import PhaseOnly,encode


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))


def distribution(q):
    entropy=-(q*np.log(np.maximum(q,1e-30))).sum(1)
    return dict(n=len(q),mean_power=q.mean(0).tolist(),std_power=q.std(0).tolist(),
                quantiles=np.quantile(q,[0,.05,.25,.5,.75,.95,1],axis=0).tolist(),
                largest_share=np.bincount(q.argmax(1),minlength=4).tolist(),
                fraction_positive=(q>0).mean(0).tolist(),fraction_over_half=(q>.5).mean(0).tolist(),
                entropy_normalized_mean=float(entropy.mean()/np.log(4)),
                effective_experts_mean=float(np.exp(entropy).mean()))


@torch.no_grad()
def routes(model,x,batch):
    return np.concatenate([model.route(encode(torch.from_numpy(x[i:i+batch]).cuda(),1.))[0].cpu().numpy()
                           for i in range(0,len(x),batch)])


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    summary=json.loads((args.source_run/'dynamic_four/summary.json').read_text())
    checkpoint_path=args.source_run/'dynamic_four/best_checkpoint.pt'
    assert sha(checkpoint_path)==summary['checkpoint_sha256']
    source_meta=json.loads((args.source_run/'metadata.json').read_text())
    assert sha(args.data)==source_meta['data_sha256']
    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False);cfg=ck['config']
    model=PhaseOnly('dynamic_four',cfg).cuda().eval();model.load_state_dict(ck['model'])
    before={n:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for n,v in model.state_dict().items()}
    data=np.load(args.data,allow_pickle=False)
    x=data['validation_images'];labels=data['validation_labels'];domains=data['validation_domains'];ids=data['validation_ids']
    assert len(labels)==2000
    train_q=routes(model,data['train_images'],cfg['batch_size'])
    q=routes(model,x,cfg['batch_size']);native_route=model.route
    assert np.all(q>0) and np.allclose(q.sum(1),1,atol=1e-6)
    distributions={str(d):distribution(q[domains==d]) for d in (0,1)}
    for d in (0,1):
        assert np.allclose(distributions[str(d)]['mean_power'],summary['validation']['routing'][str(d)]['mean_power'],atol=1e-6)
        assert distributions[str(d)]['largest_share']==summary['validation']['routing'][str(d)]['largest_expert_counts']
    class_stats={f'{d}:{c}':distribution(q[(domains==d)&(labels==c)]) for d in (0,1) for c in range(10)}
    identity={str(v):i for i,v in enumerate(ids)}
    swap=np.array([identity[str(v).rsplit(':',1)[0]+':'+str(1-int(d))] for v,d in zip(ids,domains)])
    assert np.array_equal(labels[swap],labels) and np.array_equal(swap[swap],np.arange(len(ids)))
    pair_a=np.flatnonzero(domains==0);pair_b=swap[pair_a]
    paired=dict(n=len(pair_a),same_largest_fraction=float((q[pair_a].argmax(1)==q[pair_b].argmax(1)).mean()),
                total_variation_mean=float((abs(q[pair_a]-q[pair_b]).sum(1)/2).mean()))
    policies={'native':q,'uniform':np.full_like(q,.25),
              'fixed_train_mean':np.broadcast_to(train_q.mean(0),(len(q),4)).copy(),
              'paired_other_domain':q[swap]}
    rng=np.random.default_rng(20260916)
    for mode in ['within_domain','within_domain_class']:
        index=np.arange(len(q))
        for d in (0,1):
            for c in (range(10) if mode.endswith('_class') else [None]):
                subset=np.flatnonzero((domains==d)&((labels==c) if c is not None else np.ones(len(q),dtype=bool)))
                index[subset]=rng.permutation(subset)
        policies['shuffle_'+mode]=q[index]
    for e in range(4):
        one=np.zeros_like(q);one[:,e]=1.;policies[f'only_E{e+1}']=one
        dropped=q.copy();dropped[:,e]=0.;dropped/=dropped.sum(1,keepdims=True)
        policies[f'drop_E{e+1}']=dropped
    metadata=dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
                  source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('models.py')]},
                  source_training_commit=source_meta['git_commit'],checkpoint_sha256=sha(checkpoint_path),
                  data_sha256=sha(args.data),selected_epoch=ck['epoch'],config=cfg,python=sys.version,
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  seed=20260916,scope='frozen validation diagnostics; no training or test data',
                  interventions=list(policies),interpretation='coherent-field interventions; not additive causal expert contributions')
    save(args.out/'metadata.json',metadata)
    save(args.out/'routing_distribution.json',dict(domains=distributions,domain_class=class_stats,paired=paired,
         train_mean_power=train_q.mean(0).tolist(),initial_domains=summary['initial_validation']['routing']))
    with (args.out/'sample_routes.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['sample_id','domain','label','largest_expert']+[f'q_E{i+1}' for i in range(4)])
        writer.writerows([sid,int(d),int(c),int(v.argmax()+1),*v.tolist()] for sid,d,c,v in zip(ids,domains,labels,q))
    results={}
    for name,assigned in policies.items():
        assert np.allclose(assigned.sum(1),1,atol=1e-6)
        ps=[];capture=[];max_power_error=0.
        for start in range(0,len(x),cfg['batch_size']):
            images=torch.from_numpy(x[start:start+cfg['batch_size']]).cuda()
            weights=torch.from_numpy(assigned[start:start+cfg['batch_size']]).cuda()
            if name=='native':model.route=native_route
            else:model.route=types.MethodType(lambda self,amplitude,w=weights:(w,None),model)
            output=model(images)
            assert torch.allclose(output['route_power'],weights,atol=1e-7)
            max_power_error=max(max_power_error,float((output['input_power']-1).abs().max()),float((output['output_power']-1).abs().max()))
            ps.append(output['probabilities'].cpu().numpy());capture.append(output['detector_capture'].cpu().numpy())
        probs=np.concatenate(ps);captured=np.concatenate(capture);pred=probs.argmax(1)
        def metrics(mask):
            return dict(n=int(mask.sum()),accuracy=float((pred[mask]==labels[mask]).mean()),
                        nll=float(-np.log(probs[mask,labels[mask]]).mean()),detector_capture_mean=float(captured[mask].mean()))
        result=dict(overall=metrics(np.ones(len(labels),dtype=bool)),domains={str(d):metrics(domains==d) for d in (0,1)},
                    domain_class={f'{d}:{c}':metrics((domains==d)&(labels==c)) for d in (0,1) for c in range(10)},
                    max_power_error=max_power_error)
        assert max_power_error<1e-5
        if name=='native':
            assert result['overall']['accuracy']==summary['validation']['accuracy']
            with (args.source_run/'dynamic_four/validation_predictions.csv').open() as f:prior=list(csv.DictReader(f))
            expected=np.array([[float(r[f'p{k}']) for k in range(10)] for r in prior],dtype=np.float32)
            assert np.array_equal(probs,expected)
        with (args.out/(name+'_predictions.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['sample_id','domain','label','prediction','detector_capture']+[f'p{k}' for k in range(10)])
            writer.writerows([sid,int(d),int(c),int(p.argmax()),float(power),*p.tolist()] for sid,d,c,p,power in zip(ids,domains,labels,probs,captured))
        results[name]=result;save(args.out/'results.json',results)
        print(json.dumps(dict(policy=name,overall=result['overall'],domains=result['domains'])),flush=True)
    assert before=={n:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for n,v in model.state_dict().items()}
    save(args.out/'status.json',dict(state='complete',weights_unchanged=True,native_predictions_bitwise_match=True,
                                   policies=len(policies),validation_samples=len(labels)))


if __name__=='__main__':main()
