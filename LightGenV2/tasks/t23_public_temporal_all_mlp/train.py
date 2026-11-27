"""Train the MLP global replacement against the approved release on TRAIN."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t22_public_temporal_mlp.tools.build_release import import_public_module
from LightGenV2.tasks.t22_public_temporal_mlp.tools.data import load_inputs,groups,batch
from LightGenV2.tasks.t22_public_temporal_mlp.tools.run import (save_checkpoint,write,correlation,
                                                               router_regularization,update_router_bias)
from .checkpoints import BASE_SHA256,sha256,inherit_parent,checkpoint
from .evaluation import evaluate,qualifies,feature_noise

load_parent=import_public_module('model').load_lightgen2


ROOT=Path(__file__).resolve().parents[3]
TASK=Path(__file__).resolve().parent


def collect_teacher(teacher,payload,device,directory,source=None):
    path=source if source is not None else directory/'teacher_train.pt'
    if source is not None and not path.is_file():
        raise FileNotFoundError(path)
    if path.exists():
        cached=torch.load(path,map_location='cpu',weights_only=True)
        if cached['parent_sha256']!=BASE_SHA256 or cached['manifest_sha256']!=payload['identity']['manifest']['sha256']:
            raise ValueError('Teacher cache identity differs')
        expected=[name for name,split in zip(payload['ids'],payload['splits']) if split=='train']
        if len(expected)!=2250 or cached['sample_ids']!=expected:
            raise ValueError('Teacher cache must contain exactly the original TRAIN videos')
        return cached
    indices,valid=groups(payload,'train')
    values={name:[] for name in ('vision_global','language_global','prediction')}
    identities=[]
    captured={}
    handles=[]
    for name in ('vision_global','language_global'):
        def capture(module,arguments,output,name=name):captured[name]=output
        handles.append(getattr(teacher,name).register_forward_hook(capture))
    try:
        for start in range(0,len(indices),4):
            item=batch(payload,indices[start:start+4],valid[start:start+4],device)
            result=teacher.predict_from_embeddings(item['vision'])
            keep=item['valid']
            for name in ('vision_global','language_global'):
                values[name].append(captured[name][keep].float().cpu().clone())
            values['prediction'].append(result['prediction'][keep].float().cpu().clone())
            identities.extend(payload['ids'][index] for index in item['indices'][keep.cpu()].tolist())
    finally:
        for handle in handles:handle.remove()
    expected=[name for name,split in zip(payload['ids'],payload['splits']) if split=='train']
    if identities!=expected or len(identities)!=2250:raise ValueError('Teacher TRAIN ordering differs')
    cached={name:torch.cat(pieces) for name,pieces in values.items()}
    cached.update(sample_ids=identities,parent_sha256=BASE_SHA256,manifest_sha256=payload['identity']['manifest']['sha256'])
    save_checkpoint(path,cached)
    return cached


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--teacher-cache',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text('utf-8'))
    write(args.out/'configuration.json',config)
    write(args.out/'status.json',dict(state='loading_inputs'))
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if sha256(args.parent)!=BASE_SHA256:raise ValueError('Expected the originally approved public checkpoint')
    committed=subprocess.check_output(['git','show',args.source_commit+':'+Path(__file__).relative_to(ROOT).as_posix()],cwd=ROOT)
    if hashlib.sha256(committed).hexdigest()!=sha256(Path(__file__)):
        raise ValueError('Training source and recorded commit differ')
    payload=load_inputs(json.loads(args.paths.read_text('utf-8')))
    base_payload=torch.load(args.parent,map_location='cpu',weights_only=True)
    parent,_=load_parent(args.parent,args.device)
    parent.requires_grad_(False)
    write(args.out/'status.json',dict(state='collecting_TRAIN_teacher'))
    cached=collect_teacher(parent,payload,args.device,args.out,args.teacher_cache)
    model=inherit_parent(parent.cpu(),config['global_hidden']).to(args.device)
    del parent
    model.requires_grad_(False)
    global_parameters=list(model.vision_global.parameters())+list(model.language_global.parameters())
    for value in global_parameters:value.requires_grad_(True)
    readout=list(model.readout.parameters())
    bias=model.language_router.fc2.bias
    optimizer=torch.optim.AdamW([dict(params=global_parameters,lr=config['global_learning_rate']),
        dict(params=readout,lr=config['readout_learning_rate']),
        dict(params=[bias],lr=config['router_bias_learning_rate'])],weight_decay=config['weight_decay'])
    teacher={name:cached[name].to(args.device) for name in ('vision_global','language_global','prediction')}
    feature_scales={}
    noise_reference={}
    with torch.no_grad():
        for name in ('vision_global','language_global'):
            value=teacher[name]
            flat=value.reshape(-1,*value.shape[-2:])
            average=flat.mean(0)
            getattr(model,name).fc2.bias.copy_(average.flatten())
            torch.nn.init.normal_(getattr(model,name).fc2.weight,std=.001)
            feature_scales[name]=float((flat-average).square().mean().clamp_min(1e-6))
            noise_reference[name.split('_')[0]]=float(value.square().mean().sqrt())
    train_indices=torch.tensor([i for i,s in enumerate(payload['splits']) if s=='train'])
    lookup=torch.full((len(payload['ids']),),-1,dtype=torch.long)
    lookup[train_indices]=torch.arange(len(train_indices))
    target_scale=payload['target'][train_indices].std(unbiased=False).to(args.device)
    protocol=dict(source_commit=args.source_commit,parent_checkpoint_sha256=BASE_SHA256,
                  initialization='Inherited non-global tensors; fresh MLP globals with TRAIN mean output biases',
                  config=config,model_config=asdict(model.cfg),dataset=payload['identity'],
                  supervised_split='train',selection_split='original_TEST_development',
                  python_version=platform.python_version(),torch_version=str(torch.__version__),
                  feature_scales=feature_scales,feature_noise_reference=noise_reference)
    write(args.out/'protocol.json',protocol)
    history=[]
    best=-float('inf')
    best_epoch=None
    eligible_evaluations=0
    batch_fields=config['batch_videos']//16
    for epoch in range(1,config['epochs']+1):
        model.train()
        if epoch==config['warmup_epochs']+1:
            for value in readout:value.requires_grad_(True)
            bias.requires_grad_(True)
        indices,valid=groups(payload,'train',seed=config['seed']+epoch)
        losses=[]
        for start in range(0,len(indices),batch_fields):
            ii=indices[start:start+batch_fields]
            vv=valid[start:start+batch_fields]
            item=batch(payload,ii,vv,args.device)
            positions=lookup[ii].to(args.device)
            if bool((positions<0).any()):raise ValueError('A non-TRAIN example reached the training loss')
            keep=item['valid']
            captured={}
            handles=[]
            for name in ('vision_global','language_global'):
                def capture(module,arguments,output,name=name):captured[name]=output
                handles.append(getattr(model,name).register_forward_hook(capture))
            optimizer.zero_grad(set_to_none=True)
            try:result=model.predict_from_embeddings(item['vision'])
            finally:
                for handle in handles:handle.remove()
            prediction=result['prediction'][keep]
            target=item['target'][keep]
            desired=teacher['prediction'][positions][keep]
            features=torch.stack([F.mse_loss(captured[name][keep],teacher[name][positions][keep])/feature_scales[name]
                                  for name in ('vision_global','language_global')]).mean()
            score_loss=F.mse_loss(prediction/target_scale,desired/target_scale)
            supervised=F.mse_loss(prediction/target_scale,target/target_scale)
            routes={'frame':result['routing']['vision'],'video':result['routing']['language']}
            regularization,_=router_regularization(routes,keep)
            loss=(config['feature_weight']*features+config['teacher_weight']*score_loss
                  +config['supervised_weight']*supervised
                  +config['correlation_weight']*(1-correlation(prediction,target))
                  +config['router_importance_weight']*regularization['importance'])
            if epoch>config['warmup_epochs'] and float(torch.rand(()))<config['noise_probability']:
                with feature_noise(model,config['training_noise_fraction'],noise_reference,config['seed']+epoch*100+start):
                    observed=model.predict_from_embeddings(item['vision'])['prediction'][keep]
                loss=loss+config['noise_loss_weight']*F.mse_loss(observed/target_scale,target/target_scale)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite MLP global training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(global_parameters+readout+[bias],1.)
            optimizer.step()
            if bias.requires_grad:
                update_router_bias(model.language_router,result['routing']['language']['selected'],keep,
                                   config['router_bias_correction_rate'])
            losses.append(float(loss.detach()))
        row=dict(epoch=epoch,loss=float(np.mean(losses)))
        if epoch==1 or epoch%config['evaluate_every']==0 or epoch==config['epochs']:
            metrics,_=evaluate(model,payload,args.device)
            eligible=qualifies(metrics,config['minimum_srcc'],config['maximum_srcc'])
            row.update(srcc=metrics['srcc'],eligible=eligible,
                       vision_share=metrics['vision_routing']['selection_share'],
                       language_share=metrics['language_routing']['selection_share'],
                       language_maximum_slot_deviation=metrics['language_routing']['maximum_slot_share_deviation'])
            if eligible:eligible_evaluations+=1
            if eligible and metrics['srcc']>best:
                best=metrics['srcc'];best_epoch=epoch
                save_checkpoint(args.out/'best_checkpoint.pt',checkpoint(model,base_payload,
                                dict(protocol=protocol,epoch=epoch,metrics=metrics)))
                write(args.out/'best_metrics.json',metrics)
        save_checkpoint(args.out/'last_checkpoint.pt',checkpoint(model,base_payload,dict(protocol=protocol,epoch=epoch)))
        history.append(row)
        write(args.out/'history.json',history)
        write(args.out/'status.json',dict(state='training',epoch=epoch,best_epoch=best_epoch,
              best_srcc=best if best_epoch is not None else None,eligible_evaluations=eligible_evaluations))
        print(json.dumps(row),flush=True)
        if epoch>=config['minimum_epochs'] and eligible_evaluations>=config['stop_after_eligible_epochs']:
            break
    write(args.out/'status.json',dict(state='completed',epoch=epoch,best_epoch=best_epoch,
                                     best_srcc=best if best_epoch is not None else None))


if __name__=='__main__':main()
