"""Matched phase-only pilot: training/validation only, no test-set selection."""
import argparse
import csv
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from models import PhaseOnly, objective, ARCHIVE

HERE = Path(__file__).resolve().parent


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors_sha(state):
    return {n:hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest() for n,x in state.items()}


def batches(images, labels, domains, order, batch_size, augment_seed=None):
    generator = torch.Generator().manual_seed(augment_seed) if augment_seed is not None else None
    for indices in order.split(batch_size):
        x = images[indices].cuda()
        if generator is not None:
            flipped = torch.rand(len(indices), generator=generator).cuda() < .5
            x = torch.where(flipped[:,None,None,None], x.flip(2), x)
        yield x, labels[indices].cuda(), domains[indices], indices


@torch.no_grad()
def evaluate(model, data, cfg):
    model.eval()
    x,y,d = data
    predictions, powers, routes, losses = [], [], [], []
    for images, labels, _, _ in batches(x,y,d,torch.arange(len(y)),cfg['batch_size']):
        output = model(images)
        losses.append((float(objective(output,labels)),len(labels)))
        predictions.append(output['probabilities'].cpu().numpy())
        powers.append(output['detector_capture'].cpu().numpy())
        if output['route_power'] is not None:
            routes.append(output['route_power'].cpu().numpy())
    prob = np.concatenate(predictions);target=y.numpy();domain=d.numpy();pred=prob.argmax(1)
    confusion = np.zeros((10,10),dtype=int)
    np.add.at(confusion,(target,pred),1)
    metrics = dict(accuracy=float((pred==target).mean()),
                   domain_accuracy={str(k):float((pred[domain==k]==target[domain==k]).mean()) for k in (0,1)},
                   loss=sum(v*n for v,n in losses)/len(y), confusion_matrix=confusion.tolist(),
                   detector_capture_mean=float(np.concatenate(powers).mean()))
    if routes:
        q=np.concatenate(routes)
        metrics['routing']={str(k):dict(mean_power=q[domain==k].mean(0).tolist(),
                                       std_power=q[domain==k].std(0).tolist(),
                                       largest_expert_counts=np.bincount(q[domain==k].argmax(1),minlength=4).tolist()) for k in (0,1)}
    return metrics,prob


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','train'],required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    sources={str(p.relative_to(HERE)):sha(p) for p in HERE.glob('*') if p.suffix in {'.py','.json'}}
    sources['archived_optics.py']=sha(ARCHIVE/'optical_reference/optics.py')
    metadata=dict(config=cfg,command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
                  source_sha256=sources,python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  environment=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),scope='validation-only pilot')
    save(args.out/'metadata.json',metadata)
    if args.phase=='smoke':
        images=torch.randint(1,255,(8,56,56,3),dtype=torch.uint8,device='cuda')
        results=[];initial_main=None
        for architecture in cfg['architectures']:
            model=PhaseOnly(architecture,cfg).cuda()
            assert all(name in {'first_phase','global_phase','router_phase'} for name,_ in model.named_parameters())
            hashes=tensors_sha(dict(model.named_parameters()))
            if architecture=='dynamic_four':initial_main={n:h for n,h in hashes.items() if n!='router_phase'}
            if architecture=='fixed_four':assert hashes==initial_main
            output=model(images);loss=objective(output,torch.arange(8,device='cuda'));loss.backward()
            gradients={n:float(p.grad.norm()) for n,p in model.named_parameters()}
            assert all(np.isfinite(x) and x>0 for x in gradients.values())
            assert torch.allclose(output['input_power'],torch.ones(8,device='cuda'),atol=1e-5)
            assert torch.allclose(output['output_power'],output['input_power'],atol=1e-5)
            if output['route_power'] is not None:
                assert torch.allclose(output['route_power'].sum(1),torch.ones(8,device='cuda'),atol=1e-6)
            results.append(dict(architecture=architecture,parameters=sum(p.numel() for p in model.parameters()),
                                loss=float(loss),gradients=gradients,
                                max_power_error=float((output['output_power']-output['input_power']).abs().max())))
            del model,output,loss;torch.cuda.empty_cache()
        save(args.out/'smoke.json',results);print(json.dumps(results),flush=True)
    else:
        assert args.data is not None
        dataset_manifest=json.loads(args.data.with_name('manifest.json').read_text())
        assert sha(args.data)==dataset_manifest['data_sha256']
        for key in ['train_pairs_per_class','validation_pairs_per_class','protocol']:
            assert dataset_manifest['protocol'][key]==cfg[key]
        with np.load(args.data,allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in z.files}
        assert not any(k.startswith('test') for k in arrays)
        train=tuple(torch.from_numpy(arrays['train_'+k]) for k in ['images','labels','domains'])
        val=tuple(torch.from_numpy(arrays['validation_'+k]) for k in ['images','labels','domains'])
        for split,data in [('train',train),('validation',val)]:
            for domain in (0,1):
                assert torch.bincount(data[1][data[2]==domain],minlength=10).tolist()==[cfg[split+'_pairs_per_class']]*10
        metadata.update(data_sha256=sha(args.data),data_manifest_sha256=sha(args.data.with_name('manifest.json')),
                        split_sha256=dataset_manifest['original_split_sha256'])
        save(args.out/'metadata.json',metadata)
        summaries=[]
        for architecture in cfg['architectures']:
            dest=args.out/architecture;dest.mkdir()
            model=PhaseOnly(architecture,cfg).cuda()
            initial_hashes=tensors_sha(dict(model.named_parameters()))
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=cfg['learning_rate']*cfg['minimum_lr_ratio'])
            best=(-1.,float('-inf'));history=[];started=time.perf_counter()
            initial_metrics,_=evaluate(model,val,cfg);save(dest/'initial_validation.json',initial_metrics)
            for epoch in range(1,cfg['epochs']+1):
                model.train();total=0.;correct=0;n=0
                order=torch.randperm(len(train[1]),generator=torch.Generator().manual_seed(cfg['seed']*1000003+epoch))
                for images,labels,domains,indices in batches(*train,order,cfg['batch_size'],augment_seed=cfg['seed']*99991+epoch):
                    optimizer.zero_grad(set_to_none=True)
                    output=model(images);loss=objective(output,labels)
                    assert bool(torch.isfinite(loss));loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();total+=float(loss)*len(labels);correct+=int((output['probabilities'].argmax(1)==labels).sum());n+=len(labels)
                validation,_=evaluate(model,val,cfg)
                row=dict(epoch=epoch,train_loss=total/n,train_accuracy=correct/n,validation=validation,
                         learning_rate=optimizer.param_groups[0]['lr'],order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
                         seconds=time.perf_counter()-started)
                history.append(row)
                key=(validation['accuracy'],-validation['loss'])
                if key>best:
                    best=key
                    torch.save(dict(model=model.state_dict(),epoch=epoch,config=cfg,architecture=architecture,validation=validation),dest/'best_checkpoint.pt')
                scheduler.step()
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,config=cfg),dest/'last_checkpoint.pt')
                save(dest/'history.json',history)
                save(args.out/'status.json',dict(state='training',architecture=architecture,epoch=epoch,epochs=cfg['epochs']))
                print(json.dumps(dict(architecture=architecture,**row)),flush=True)
            checkpoint=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['model']);metrics,prob=evaluate(model,val,cfg)
            with (dest/'validation_predictions.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['sample_id','domain','label','prediction']+[f'p{i}' for i in range(10)])
                writer.writerows([str(i),int(d),int(y),int(p.argmax()),*p.tolist()] for i,d,y,p in zip(arrays['validation_ids'],arrays['validation_domains'],arrays['validation_labels'],prob))
            end_hashes=tensors_sha(dict(model.named_parameters()))
            assert all(end_hashes[k]!=v for k,v in initial_hashes.items())
            summary=dict(architecture=architecture,parameters=sum(p.numel() for p in model.parameters()),selected_epoch=checkpoint['epoch'],
                         initial_validation=initial_metrics,validation=metrics,seconds=time.perf_counter()-started,
                         checkpoint_sha256=sha(dest/'best_checkpoint.pt'),initial_parameters_sha256=initial_hashes)
            save(dest/'summary.json',summary);summaries.append(summary)
            del model,optimizer,scheduler,checkpoint,output,loss;torch.cuda.empty_cache()
        assert len({tuple(x['order_sha256'] for x in json.loads((args.out/a/'history.json').read_text())) for a in cfg['architectures']})==1
        save(args.out/'results.json',summaries)
    save(args.out/'status.json',dict(state='complete',phase=args.phase))


if __name__=='__main__':
    main()
