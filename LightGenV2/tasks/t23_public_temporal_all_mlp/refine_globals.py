"""Continue TRAIN-only distillation of the selected MLP globals and readout."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t22_public_temporal_mlp.tools.data import load_inputs, groups, batch
from LightGenV2.tasks.t22_public_temporal_mlp.tools.run import write, save_checkpoint, correlation
from .checkpoints import BASE_SHA256, checkpoint, load_lightgen2, sha256
from .evaluation import evaluate, qualifies, feature_noise


ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','teacher-cache','paths','config','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    config=json.loads(args.config.read_text('utf-8'))
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    committed=subprocess.check_output(['git','show',args.source_commit+':'+Path(__file__).relative_to(ROOT).as_posix()],cwd=ROOT)
    if hashlib.sha256(committed).hexdigest()!=sha256(Path(__file__)):
        raise ValueError('Recorded refinement source differs')
    write(args.out/'status.json',dict(state='loading_inputs'))
    payload=load_inputs(json.loads(args.paths.read_text('utf-8')))
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model,_=load_lightgen2(args.checkpoint,args.device)
    model.requires_grad_(False)
    names=[name for name,split in zip(payload['ids'],payload['splits']) if split=='train']
    cached=torch.load(args.teacher_cache,map_location='cpu',weights_only=True)
    if cached['sample_ids']!=names or len(names)!=2250 or cached['parent_sha256']!=BASE_SHA256:
        raise ValueError('TRAIN teacher identity differs')
    if cached['manifest_sha256']!=payload['identity']['manifest']['sha256']:
        raise ValueError('Teacher manifest differs')
    teacher={name:cached[name].to(args.device) for name in ('vision_global','language_global','prediction')}
    train_indices=torch.tensor([index for index,split in enumerate(payload['splits']) if split=='train'])
    lookup=torch.full((len(payload['ids']),),-1,dtype=torch.long)
    lookup[train_indices]=torch.arange(len(train_indices))
    scale=payload['target'][train_indices].std(unbiased=False).to(args.device)
    prefixes=('vision_global.','language_global.','readout.')
    frozen={name:value.cpu().clone() for name,value in model.state_dict().items() if not name.startswith(prefixes)}
    global_parameters=[]
    for name in ('vision_global','language_global'):
        module=getattr(model,name)
        module.requires_grad_(True)
        global_parameters.extend(module.parameters())
    model.readout.requires_grad_(True)
    parameters=global_parameters+list(model.readout.parameters())
    optimizer=torch.optim.AdamW([
        dict(params=global_parameters,lr=config['global_learning_rate']),
        dict(params=model.readout.parameters(),lr=config['readout_learning_rate'])],
        weight_decay=config['weight_decay'])
    protocol=dict(original['training']['protocol'])
    feature_scales=protocol['feature_scales']
    noise_reference=protocol['feature_noise_reference']
    protocol.update(source_commit=args.source_commit,stage='global_and_readout_refinement',configuration=config,
                    initial_checkpoint_sha256=sha256(args.checkpoint),
                    supervised_split='train',selection_split='original_TEST_development',
                    trainable_components=['vision_global','language_global','readout'],
                    python_version=platform.python_version(),torch_version=str(torch.__version__))
    write(args.out/'protocol.json',protocol)
    write(args.out/'configuration.json',config)
    history=[]
    best=None
    best_epoch=None
    eligible_count=0
    batch_fields=config['batch_size']//16
    for epoch in range(1,config['epochs']+1):
        indices,valid=groups(payload,'train',seed=config['seed']+epoch)
        losses=[]
        for start in range(0,len(indices),batch_fields):
            ii=indices[start:start+batch_fields]
            vv=valid[start:start+batch_fields]
            positions=lookup[ii].to(args.device)
            if bool((positions<0).any()):
                raise ValueError('A non-TRAIN example reached the training loss')
            item=batch(payload,ii,vv,args.device)
            keep=item['valid']
            captured={}
            handles=[]
            for name in ('vision_global','language_global'):
                def capture(module,arguments,output,name=name):captured[name]=output
                handles.append(getattr(model,name).register_forward_hook(capture))
            optimizer.zero_grad(set_to_none=True)
            try:
                prediction=model.predict_from_embeddings(item['vision'])['prediction'][keep]
            finally:
                for handle in handles:handle.remove()
            target=item['target'][keep]
            desired=teacher['prediction'][positions][keep]
            features=torch.stack([F.mse_loss(captured[name][keep],teacher[name][positions][keep])/feature_scales[name]
                                  for name in ('vision_global','language_global')]).mean()
            loss=(config['feature_weight']*features
                  +config['teacher_weight']*F.mse_loss(prediction/scale,desired/scale)
                  +config['supervised_weight']*F.mse_loss(prediction/scale,target/scale)
                  +config['correlation_weight']*(1-correlation(prediction,target)))
            if float(torch.rand(()))<config['noise_probability']:
                with feature_noise(model,config['noise_fraction'],noise_reference,config['seed']+epoch*100+start):
                    noisy=model.predict_from_embeddings(item['vision'])['prediction'][keep]
                loss=loss+config['noise_weight']*F.mse_loss(noisy/scale,target/scale)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite refinement loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics,_=evaluate(model,payload,args.device)
        eligible=qualifies(metrics,config['minimum_srcc'],config['maximum_srcc'])
        if eligible:
            eligible_count+=1
            if best is None or metrics['srcc']>best:
                best=metrics['srcc']
                best_epoch=epoch
                save_checkpoint(args.out/'best_checkpoint.pt',checkpoint(model,original,
                    dict(protocol=protocol,epoch=epoch,metrics=metrics)))
                write(args.out/'best_metrics.json',metrics)
        save_checkpoint(args.out/'last_checkpoint.pt',checkpoint(model,original,dict(protocol=protocol,epoch=epoch)))
        row=dict(epoch=epoch,loss=float(np.mean(losses)),srcc=metrics['srcc'],eligible=eligible)
        history.append(row)
        write(args.out/'history.json',history)
        write(args.out/'status.json',dict(state='training',epoch=epoch,best_epoch=best_epoch,best_srcc=best))
        print(json.dumps(row),flush=True)
        if epoch>=config['minimum_epochs'] and eligible_count>=config['eligible_evaluations']:
            break
    for name,value in frozen.items():
        if not torch.equal(value,model.state_dict()[name].cpu()):
            raise ValueError('A frozen component changed '+name)
    write(args.out/'status.json',dict(state='completed',epoch=epoch,best_epoch=best_epoch,best_srcc=best,
                                    frozen_components_verified=True))


if __name__=='__main__':main()
