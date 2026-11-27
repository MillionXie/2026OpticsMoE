"""Refine the existing five-grade readout using the original TRAIN videos."""
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
from .evaluation import evaluate, qualifies


ROOT=Path(__file__).resolve().parents[3]


@torch.no_grad()
def collect_features(model,payload,device):
    indices,valid=groups(payload,'train')
    pieces=[]
    identities=[]
    captured=[]
    def capture(module,arguments,output):captured.append(output)
    handle=model.language_global.register_forward_hook(capture)
    try:
        for start in range(0,len(indices),8):
            item=batch(payload,indices[start:start+8],valid[start:start+8],device)
            model.predict_from_embeddings(item['vision'])
            pieces.append(captured.pop()[item['valid']].cpu().clone())
            identities.extend(payload['ids'][index] for index in item['indices'][item['valid'].cpu()].tolist())
    finally:
        handle.remove()
    expected=[name for name,split in zip(payload['ids'],payload['splits']) if split=='train']
    if identities!=expected or len(identities)!=2250:raise ValueError('TRAIN feature identities differ')
    return torch.cat(pieces)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--teacher-cache',type=Path,required=True)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
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
    train_indices=torch.tensor([index for index,split in enumerate(payload['splits']) if split=='train'])
    names=[payload['ids'][index] for index in train_indices.tolist()]
    teacher=torch.load(args.teacher_cache,map_location='cpu',weights_only=True)
    if teacher['sample_ids']!=names or teacher['parent_sha256']!=BASE_SHA256:
        raise ValueError('Teacher identity or split differs')
    if teacher['manifest_sha256']!=payload['identity']['manifest']['sha256']:
        raise ValueError('Teacher manifest differs')
    write(args.out/'status.json',dict(state='collecting_TRAIN_features'))
    features=collect_features(model,payload,args.device).to(args.device)
    targets=payload['target'][train_indices].to(args.device)
    desired=teacher['prediction'].to(args.device)
    scale=targets.std(unbiased=False)
    frozen={name:value.cpu().clone() for name,value in model.state_dict().items() if not name.startswith('readout.')}
    model.readout.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.readout.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    protocol=dict(original['training']['protocol'])
    protocol.update(source_commit=args.source_commit,stage='readout_refinement',configuration=config,
                    initial_checkpoint_sha256=sha256(args.checkpoint),
                    supervised_split='train',selection_split='original_TEST_development',
                    trainable_components=['readout'],noise_location='language_global_output',
                    python_version=platform.python_version(),torch_version=str(torch.__version__))
    write(args.out/'protocol.json',protocol)
    write(args.out/'configuration.json',config)
    history=[]
    best=None
    best_epoch=None
    eligible_count=0
    noise_scale=protocol['feature_noise_reference']['language']
    for epoch in range(1,config['epochs']+1):
        order=torch.randperm(len(targets),device=args.device)
        losses=[]
        for start in range(0,len(order),config['batch_size']):
            positions=order[start:start+config['batch_size']]
            sample=features[positions]
            gold=targets[positions]
            soft=desired[positions]
            optimizer.zero_grad(set_to_none=True)
            prediction=model.readout(sample).softmax(-1)@model.grade_scores
            loss=(config['teacher_weight']*F.mse_loss(prediction/scale,soft/scale)
                  +config['supervised_weight']*F.mse_loss(prediction/scale,gold/scale)
                  +config['correlation_weight']*(1-correlation(prediction,gold)))
            if float(torch.rand(()))<config['noise_probability']:
                noisy=sample+config['noise_fraction']*noise_scale*torch.randn_like(sample)
                observed=model.readout(noisy).softmax(-1)@model.grade_scores
                loss=loss+config['noise_weight']*F.mse_loss(observed/scale,gold/scale)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite readout loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.readout.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics,_=evaluate(model,payload,args.device)
        eligible=qualifies(metrics,config['minimum_srcc'],config['maximum_srcc'])
        row=dict(epoch=epoch,loss=float(np.mean(losses)),srcc=metrics['srcc'],eligible=eligible)
        if eligible:
            eligible_count+=1
            if best is None or metrics['srcc']>best:
                best=metrics['srcc']
                best_epoch=epoch
                save_checkpoint(args.out/'best_checkpoint.pt',checkpoint(model,original,
                    dict(protocol=protocol,epoch=epoch,metrics=metrics)))
                write(args.out/'best_metrics.json',metrics)
        save_checkpoint(args.out/'last_checkpoint.pt',checkpoint(model,original,dict(protocol=protocol,epoch=epoch)))
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
