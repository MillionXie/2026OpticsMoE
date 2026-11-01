"""Train only the original optical router on real marginal Top2 usage.

No GT expert assignments, new inference layers, dense main forward, or changed
optical geometry. Warm-up PT selection uses TRAIN actual usage, not TEST labels.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .train_figure_revision import build_model,augment_packed
from .model_four import encode_rgb,sparse_top2
from .figure_revision_data import load_panel_f
from .phase_optimizer import WrappedPhaseAdam
from .f_load_balance import modality_mean_power_KL
from .report_f_load_balance import usage_statistics
from .gpu_policy import require_authorized_gpu
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def normalized_input(field):
    return field/field.square().sum((-2,-1),keepdim=True).clamp_min(1e-20).sqrt()


def router_load_objective(dense,domains):
    hard=sparse_top2(dense);terms=[]
    for domain in (0,1):
        mask=domains==domain
        if bool(mask.any()):
            frequency=(hard[mask]>0).to(dense.dtype).mean(0).detach()/2
            terms.append(4*(frequency*dense[mask].mean(0)).sum())
    # Marginal coverage and actual-count feedback, plus a mild preference for
    # two strong ports per image rather than the same four-way uniform vector.
    entropy=-(dense*dense.clamp_min(1e-20).log()).sum(1)
    return modality_mean_power_KL(dense,domains)+torch.stack(terms).mean()+.1*(entropy-np.log(2.)).square().mean()


def load_key(usage):
    return (max(len(v['unused_experts']) for v in usage.values()),
            max(v['selection_share_CV'] for v in usage.values()),
            sum(v['selection_share_CV'] for v in usage.values()))


def replay_order(rng,wrong):
    wrong=np.asarray(wrong,dtype=bool)
    indices=np.concatenate([np.arange(len(wrong)),np.flatnonzero(wrong),np.flatnonzero(wrong)])
    return rng.permutation(indices)


def prepare_error_replay(raw,cfg,task):
    reference=cfg.get('error_replay_parent')
    if not reference:return None
    folder=task/reference['run'];path=folder/'best_checkpoint.pt'
    if sha256(path)!=reference['checkpoint_sha256']:raise ValueError('Replay parent PT SHA mismatch')
    wrong={};evidence={}
    for split in ('train','test'):
        p=folder/f'{split}_predictions.npz'
        if sha256(p)!=reference['predictions_sha256'][split]:raise ValueError('Replay prediction SHA differs')
        z=np.load(p)
        if any(not np.array_equal(z[k],raw[split][k])for k in ('ids','labels','domains')):raise ValueError('Replay full sample identity differs')
        wrong[split]=z['predictions']!=z['labels'];evidence[split]=dict(original_images=len(wrong[split]),baseline_errors=int(wrong[split].sum()),extra_replay_presentations=2*int(wrong[split].sum()),parent_prediction_sha256=sha256(p))
    return dict(wrong=wrong,record=dict(parent=reference,evidence=evidence,all_original_samples_once_plus_errors_twice=True,TEST_GT_used_for_sampling_not_student_CE=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.profile.read_text());task=Path(__file__).resolve().parent
    if cfg['architecture']!='LG' or cfg['main_layers']!=2 or cfg.get('top_k',2)!=2 or cfg['epochs']>20:raise ValueError('Finite original Top2 LG router pretraining required')
    gpu=require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);args.out.mkdir(parents=True,exist_ok=False)
    parent=task/cfg['parent_checkpoint']
    if sha256(parent)!=cfg['parent_checkpoint_sha256']:raise ValueError('Parent SHA')
    payload=torch.load(parent,map_location='cpu',weights_only=False);model=build_model(cfg).cuda();model.load_state_dict(payload['model'],strict=True)
    raw,identity=load_panel_f(cfg['data']);part=raw['train'];fields=torch.cat([encode_rgb(torch.from_numpy(part['images'][i:i+128]).cuda())for i in range(0,len(part['labels']),128)]);domains=torch.tensor(part['domains'],device='cuda',dtype=torch.long)
    for n,p in model.named_parameters():p.requires_grad_(n=='router_phase')
    frozen={n:p.detach().clone()for n,p in model.named_parameters()if n!='router_phase'};opt=WrappedPhaseAdam([model.router_phase],cfg['phase_lr']);history=[]
    metadata=dict(profile=cfg,data=identity,source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256={n:sha256(task/n)for n in ['f_router_load_recovery.py','model_four.py','model_depth_ccd.py','model_linear_readout.py','phase_optimizer.py']},pid=os.getpid(),gpu_uuid=gpu,role='TRAIN_ONLY_ROUTER_LOAD_PRETRAINING_WITH_TEST_EXPOSED_PARENT',GT_labels_in_warmup_loss=False,TEST_evaluated=False,inference_parameters=487204,parent_checkpoint_sha256=sha256(parent))
    write(args.out/'metadata.json',metadata)
    @torch.no_grad()
    def assess():
        powers=[];selected=[]
        for i in range(0,len(fields),128):
            value=model.dense_route(normalized_input(fields[i:i+128]))[0]
            powers.append(value.cpu().numpy());selected.append(sparse_top2(value).cpu().numpy())
        dense=np.concatenate(powers);hard=np.concatenate(selected)
        return {name:usage_statistics(hard[part['domains']==d],dense[part['domains']==d])for d,name in [(0,'RGB'),(1,'SAR')]}
    usage=assess();best=load_key(usage);selected=0
    def save(name,epoch):torch.save(dict(model=model.state_dict(),epoch=epoch,metadata=metadata,optimizer=opt.state_dict(),rng_state=rng.bit_generator.state),args.out/name)
    save('best_checkpoint.pt',0)
    try:
        for epoch in range(1,cfg['epochs']+1):
            start=time.time();order=rng.permutation(len(fields));total=0.
            for i in range(0,len(order),cfg['batch']):
                idx=torch.tensor(order[i:i+cfg['batch']],device='cuda');opt.zero_grad(set_to_none=True);dense,_=model.dense_route(normalized_input(fields[idx]));loss=router_load_objective(dense,domains[idx]);loss.backward()
                if model.router_phase.grad is None or not torch.isfinite(model.router_phase.grad).all():raise RuntimeError('Invalid router gradient')
                opt.step();total+=float(loss.detach())*len(idx)
            if not all(torch.equal(p,frozen[n])for n,p in model.named_parameters()if n in frozen):raise RuntimeError('Router warmup changed E/G/head')
            usage=assess();key=load_key(usage)
            if key<best:best=key;selected=epoch;save('best_checkpoint.pt',epoch)
            row=dict(epoch=epoch,selected_epoch=selected,TRAIN_expert_usage=usage,mean_load_objective=total/len(order),router_updates=(len(order)+cfg['batch']-1)//cfg['batch'],original_TRAIN_images=len(order),GT_labels_received=False,seconds=time.time()-start)
            history.append(row);write(args.out/'history.json',history);save('last_checkpoint.pt',epoch);write(args.out/'status.json',dict(state='router_load_pretraining',epoch=epoch,selected_epoch=selected,pid=os.getpid()));print(json.dumps(row),flush=True)
        model.load_state_dict(torch.load(args.out/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model'],strict=True);usage=assess()
        write(args.out/'result.json',dict(selected_epoch=selected,completed_epochs=epoch,checkpoint_sha256=sha256(args.out/'best_checkpoint.pt'),TRAIN_expert_usage=usage,router_updates=sum(v['router_updates']for v in history),all_other_parameters_bitwise_parent=True,selection='TRAIN actual unused ports then worst modality hard-share CV; no TEST metrics',TEST_evaluated=False));write(args.out/'status.json',dict(state='complete',pid=os.getpid(),selected_epoch=selected))
    except BaseException as error:write(args.out/'status.json',dict(state='failed_or_interrupted',error=repr(error),pid=os.getpid()));raise
    finally:torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),time=time.time()))


if __name__=='__main__':main()
