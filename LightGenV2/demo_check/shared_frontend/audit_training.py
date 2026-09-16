"""Read-only convergence, per-expert gradient and routing counterfactual audit."""
import argparse
import hashlib
import importlib.util
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frontend_training_runner',HERE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


@torch.no_grad()
def collect(model,data,cfg):
    probabilities=[];routes=[]
    for x,y,d,indices in r.batches(*data,torch.arange(len(data[1])),cfg['batch_size']):
        output=model(x);probabilities.append(output['probabilities'].cpu().numpy())
        if output['route_power'] is not None:routes.append(output['route_power'].cpu().numpy())
    p=np.concatenate(probabilities);y=data[1].numpy();d=data[2].numpy();pred=p.argmax(1)
    return dict(accuracy=float((pred==y).mean()),domain_accuracy={str(k):float((pred[d==k]==y[d==k]).mean()) for k in (0,1)},
                nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())),p,np.concatenate(routes) if routes else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    source=json.loads((a.run/'metadata.json').read_text());cfg=source['config'];ocfg=source['optical_config']
    assert r.sha(a.data)==source['data_sha256']
    with np.load(a.data,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    assert not any(k.startswith('test') for k in arrays)
    train=tuple(torch.from_numpy(arrays['train_'+k]) for k in ['images','labels','domains'])
    val=tuple(torch.from_numpy(arrays['validation_'+k]) for k in ['images','labels','domains'])
    frontend_path=a.run/'frontend/best_checkpoint.pt'
    frontend=r.SharedFrontend(torch.load(frontend_path,map_location='cpu',weights_only=False)['model']).cuda()
    r.save(a.out/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_run=str(a.run),source_training_commit=source['git_commit'],config=cfg,operation='read-only fixed-best-checkpoint audit',
        python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(),data_sha256=r.sha(a.data),
        frontend_checkpoint_sha256=r.sha(frontend_path),script_sha256=r.sha(Path(__file__)),test_set_used=False))
    reports=[]
    for architecture in cfg['architectures']:
        path=a.run/architecture/'best_checkpoint.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        model=r.FrontendOptics(frontend,architecture,ocfg).cuda().eval();model.optical.load_state_dict(ck['model'])
        before=r.tensors_sha(model.state_dict());initial=r.PhaseOnly(architecture,ocfg).cuda();phase={}
        for name,param in model.optical.named_parameters():
            init=dict(initial.named_parameters())[name];s=param.sigmoid();delta=(2*torch.pi*(s-init.sigmoid()))
            phase[name]=dict(raw_delta_rms=float((param-init).square().mean().sqrt()),
                circular_phase_delta_rms_rad=float(torch.atan2(delta.sin(),delta.cos()).square().mean().sqrt()),
                sigmoid_saturated_fraction=float(((s<.01)|(s>.99)).float().mean()),
                sigmoid_derivative_mean=float((s*(1-s)).mean()))
        del initial
        baseline,prob,q=collect(model,val,cfg)
        assert baseline['accuracy']==ck['validation']['accuracy']
        r.write_predictions(a.out/(architecture+'_validation.csv'),arrays,prob)
        rng=np.random.default_rng(42);indices=[]
        for d in (0,1):
            for y in range(10):indices.extend(rng.choice(np.flatnonzero((arrays['train_domains']==d)&(arrays['train_labels']==y)),16,replace=False).tolist())
        indices=np.array(indices);rng.shuffle(indices);gradient_rows=[]
        for images,labels,_,_ in r.batches(*train,torch.from_numpy(indices),cfg['batch_size']):
            model.zero_grad(set_to_none=True);loss=r.objective(model(images),labels);loss.backward()
            row={n:float(v.grad.norm()) for n,v in model.optical.named_parameters()}
            if architecture=='dynamic_four':row['expert_norms']=model.optical.first_phase.grad.flatten(1).norm(dim=1).tolist()
            assert all(v.grad is None for v in frontend.parameters());gradient_rows.append(row)
        gradient={name:dict(mean=float(np.mean([x[name] for x in gradient_rows])),minimum=float(np.min([x[name] for x in gradient_rows]))) for name in gradient_rows[0] if name!='expert_norms'}
        if architecture=='dynamic_four':gradient['experts']=dict(mean=np.mean([x['expert_norms'] for x in gradient_rows],axis=0).tolist(),minimum=np.min([x['expert_norms'] for x in gradient_rows],axis=0).tolist())
        report=dict(architecture=architecture,selected_epoch=ck['epoch'],checkpoint_sha256=r.sha(path),validation=baseline,phase=phase,
                    gradient_samples=320,gradient_subset_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),gradients=gradient)
        if q is not None:
            _,_,train_q=collect(model,train,cfg);train_mean=train_q.mean(0)
            entropy=-(q*np.log(np.maximum(q,1e-12))).sum(1)/np.log(4)
            report['routing']=dict(mean=q.mean(0).tolist(),std=q.std(0).tolist(),largest_expert_counts=np.bincount(q.argmax(1),minlength=4).tolist(),
                normalized_entropy_mean=float(entropy.mean()),mean_largest_share=float(q.max(1).mean()),
                largest_share_above_09_fraction=float((q.max(1)>.9).mean()),train_mean=train_mean.tolist(),
                per_class={str(k):q[arrays['validation_labels']==k].mean(0).tolist() for k in range(10)})
            original_route=model.optical.route;ablation={}
            permutation=np.random.default_rng(4242).permutation(len(q))
            np.savez_compressed(a.out/'routing.npz',probabilities=q,train_mean=train_mean,validation_ids=arrays['validation_ids'],shuffle_permutation=permutation)
            for mode,routing in [('uniform',np.full_like(q,.25)),('train_mean',np.broadcast_to(train_mean,q.shape).copy()),('globally_shuffled',q[permutation])]:
                cursor=[0]
                def replacement(amplitude):
                    _,capture=original_route(amplitude);start=cursor[0];cursor[0]+=len(amplitude)
                    return torch.from_numpy(routing[start:cursor[0]]).to(amplitude.device),capture
                model.optical.route=replacement
                metrics,predictions,_=collect(model,val,cfg);assert cursor[0]==len(q)
                r.write_predictions(a.out/('dynamic_four_'+mode+'.csv'),arrays,predictions)
                ablation[mode]=metrics
            model.optical.route=original_route;report['routing_counterfactuals']=ablation
            report['counterfactual_scope']='same learned phase plates; no retraining; not a separately trained fixed-routing baseline'
        assert before==r.tensors_sha(model.state_dict());report['state_unchanged']=True
        reports.append(report);del model;torch.cuda.empty_cache()
    r.save(a.out/'result.json',reports);r.save(a.out/'status.json',dict(state='complete'))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
