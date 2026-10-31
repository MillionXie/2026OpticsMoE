"""Read-only complete original-TRAIN gradients and one reversible router probe."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_figure_revision import build_model,training_forward
from .figure_revision_data import load_panel_f
from .model_four import encode_rgb
from .original_f_teacher import teacher_KL
from .gpu_policy import require_authorized_gpu
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    task=Path(__file__).resolve().parent;cfg=json.loads(args.profile.read_text());gpu=require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4)
    path=task/cfg['parent_checkpoint']
    if sha256(path)!=cfg['parent_checkpoint_sha256']:raise ValueError('Exact parent SHA required')
    ck=torch.load(path,map_location='cpu',weights_only=False);profile=ck['metadata']['profile'];model=build_model(profile).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    raw,_=load_panel_f(profile['data']);part=raw['train'];z=np.load(path.parent/'TRAIN_teacher_logits.npz')
    if not all(np.array_equal(z[k],part[k]) for k in ('ids','labels','domains')):raise ValueError('TRAIN teacher identities')
    # Parent records a completed, separately audited teacher cache SHA.
    expected=ck['metadata']['teacher_cache_SHA256']['TRAIN_teacher_logits.npz']
    if sha256(path.parent/'TRAIN_teacher_logits.npz')!=expected:raise ValueError('TRAIN teacher SHA')
    targets=torch.from_numpy(z['logits'][:,0]).cuda();labels=torch.tensor(part['labels'],device='cuda',dtype=torch.long);domains=torch.tensor(part['domains'],device='cuda');original={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.requires_grad_(False);model.router_phase.requires_grad_(True)
    counts=torch.bincount(domains,minlength=2);size=len(labels);batch=64;forwards=0
    def evaluate_TRAIN():
        nonlocal forwards
        power=torch.zeros(2,4,device='cuda');frequency=torch.zeros_like(power);losses=torch.zeros(2,device='cuda');correct=torch.zeros_like(losses)
        with torch.no_grad():
            for start in range(0,size,batch):
                end=min(start+batch,size);out=model(encode_rgb(torch.from_numpy(part['images'][start:end]).cuda()),return_dense_route=True);forwards+=end-start
                for d in (0,1):
                    mask=domains[start:end]==d
                    if not bool(mask.any()):continue
                    n=int(mask.sum());power[d]+=out['dense_route_power'][mask].sum(0);frequency[d]+=(out['route_power'][mask]>0).sum(0)
                    logits=out['raw_logits'][mask];losses[d]+=(F.cross_entropy(logits,labels[start:end][mask],label_smoothing=.02)+.5*teacher_KL(logits,targets[start:end][mask],2.))*n
                    correct[d]+=(logits.argmax(1)==labels[start:end][mask]).sum()
        means=power/counts[:,None];balance=(-.25*(4*means.clamp_min(1e-20)).log()).sum(1)
        return dict(mean_dense=means,per_image_frequency=frequency/counts[:,None],classification_KD=losses/counts,correct=correct,balance_KL=balance)
    baseline=evaluate_TRAIN();main_grad=torch.zeros((2,)+model.router_phase.shape,device='cuda');balance_grad=torch.zeros_like(main_grad)
    # Exact chain rule through the population marginal mean; no minibatch approximation.
    derivative=-.25/baseline['mean_dense'].clamp_min(1e-20)
    model.train();gradient_calls=0
    for start in range(0,size,batch):
        end=min(start+batch,size);out=training_forward(model,encode_rgb(torch.from_numpy(part['images'][start:end]).cuda()),profile,return_dense_route=True);forwards+=end-start
        for d in (0,1):
            mask=domains[start:end]==d
            if not bool(mask.any()):continue
            n=int(mask.sum());logits=out['raw_logits'][mask]
            main=(F.cross_entropy(logits,labels[start:end][mask],label_smoothing=.02)+.5*teacher_KL(logits,targets[start:end][mask],2.))*n/counts[d]
            aux=(out['dense_route_power'][mask].sum(0)*derivative[d]).sum()/counts[d]
            main_grad[d]+=torch.autograd.grad(main,model.router_phase,retain_graph=True)[0]
            balance_grad[d]+=torch.autograd.grad(aux,model.router_phase,retain_graph=True)[0]
            gradient_calls+=2
    sigmoid=model.router_phase.detach().sigmoid();jacobian=2*torch.pi*sigmoid*(1-sigmoid);gm=main_grad/jacobian.clamp_min(1e-12);gb=balance_grad/jacobian.clamp_min(1e-12)
    main_mean=gm.mean(0);balance_mean=gb.mean(0);coefficient=float((.2*main_mean.norm()/balance_mean.norm().clamp_min(1e-20)).clamp(1e-5,1.))
    direction=main_mean+coefficient*balance_mean;theta=2*torch.pi*sigmoid
    with torch.no_grad():
        trial=(theta-cfg['maximum_probe_angle']*direction/direction.abs().max().clamp_min(1e-20)).remainder(2*torch.pi)
        unit=(trial/(2*torch.pi)).clamp(1e-7,1-1e-7);model.router_phase.copy_(torch.logit(unit))
    model.eval();trial_metrics=evaluate_TRAIN()
    model.load_state_dict(original,strict=True)
    if not all(torch.equal(v,original[k]) for k,v in model.state_dict().items()):raise RuntimeError('Read-only probe restoration failed')
    def cosine(a,b):return float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0))
    def export(v):return {k:x.cpu().tolist() for k,x in v.items()}
    report=dict(role='ORIGINAL_TRAIN_ONLY_READONLY_DIAGNOSIS',parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],source_commit=os.environ.get('T17_SOURCE_COMMIT'),GPU=gpu,pid=os.getpid(),original_TRAIN_images=size,VAL_TEST_evaluations=0,phase_updates_persisted=0,all_model_state_bitwise_restored=True,baseline=export(baseline),single_reversible_probe=export(trial_metrics),probe_maximum_angle=cfg['maximum_probe_angle'],training_route=profile.get('training_route','top2'),calibrated_auxiliary_coefficient=coefficient,calibration='20% norm of mean physical router TRAIN classification/KD gradient; clipped [1e-5,1]',domain_gradient_cosines=[cosine(gm[d],gb[d]) for d in (0,1)],main_domain_gradient_cosine=cosine(gm[0],gm[1]),original_TRAIN_forward_presentations=forwards,router_autograd_evaluations=gradient_calls,interpretation='One fixed code0 reversible probe, not new accuracy or proof of convergence/capacity; real hard usage and both classification losses must be checked separately.')
    args.out.mkdir(parents=True,exist_ok=False);write(args.out/'diagnosis.json',report)
    np.savez_compressed(args.out/'physical_router_gradients.npz',main=gm.cpu().numpy(),balance=gb.cpu().numpy())
    write(args.out/'manifest.json',dict(parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],files={p.name:sha256(p) for p in args.out.iterdir() if p.is_file()}));print(json.dumps(report))


if __name__=='__main__':main()
