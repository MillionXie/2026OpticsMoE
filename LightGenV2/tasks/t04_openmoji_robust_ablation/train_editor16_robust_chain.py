"""Three cumulative robust continuations of the sealed editor16 architecture.

Same clean initial PT, ranks, seed, complete TRAIN and budget; only training
perturbations differ. Alpha stays at the original validated G2 values. Existing
optical/electronic parameters learn, so every selected PT needs its own CCD.
TEST checkpoint selection is explicitly authorized DEVELOPMENT, never gradients.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import train_preserved_upstream as architecture
from .profiles import PROFILES, bounded, grid_roundtrip
from .train import sha, metric
from .train_modality_fusion import settings
from LightGenV2.tasks.t04_semantic_interaction import training as t

SOURCE_SHA='01f7fc4a8de4f20901fae09e8be55c67fce2177db0a7ed3975a7fc90853eb05a'
GROUPS=('r1_ccd','r2_ccd_dc30','r3_ccd_dc30_grid')
NOISE_SCALE=5.


def install_training_profile(model,group):
    """Clean eval uses exactly r0 bounded propagation, including G5.

    Historical grid wrapper was always active; here grid is TRAIN-only, as the
    requested intervention. DC is coherent30% expert/global leakage using the
    existing backend, noise offset/read fractions15%/5% of frame reference.
    Router noise follows the historical additive mean/read proxy. No claim of
    calibrated sensor equivalence or true8um propagation.
    """
    assert group in GROUPS
    profile=PROFILES[group]
    for path in model._optical_paths():
        path.gain_min=path.gain_max=1.
        path.offset_fraction=.03*NOISE_SCALE
        path.read_noise_fraction=.01*NOISE_SCALE
        path.ccd_noise_distribution='none'
        path.input_shift_pixels=path.phase_shift_pixels=path.ccd_shift_pixels=0
        path.zero_order_enabled=profile['dc30']
        path.amplitude_zero_order_intensity_min=path.amplitude_zero_order_intensity_max=0.
        path.phase_zero_order_intensity_min=path.phase_zero_order_intensity_max=.30 if profile['dc30'] else 0.
        original=path._simulate_detector_roi
        def detector(field,modulation,shifts,*,phase_support=None,original=original):
            field=bounded(field)
            if model.training and profile['grid']:
                field=grid_roundtrip(field,1101 if field.shape[-1]==518 else 1016)
            return original(field,modulation,shifts,phase_support=phase_support)
        path._simulate_detector_roi=detector
        router=path.core.router
        router.input_shift_pixels=router.phase_shift_pixels=router.ccd_shift_pixels=0
        original_router=router._simulate
        def route(fields,original=original_router):
            fields=bounded(fields)
            if model.training and profile['grid']: fields=grid_roundtrip(fields,1016)
            intensity=original(fields)
            if model.training:
                ref=intensity.mean((-2,-1),keepdim=True).detach()
                intensity=(intensity+.03*NOISE_SCALE*ref+
                           .01*NOISE_SCALE*ref*torch.randn_like(intensity)).clamp_min(0)
            return intensity
        router._simulate=route


def gate_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if 'optical_fusion_logit' in n}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--group',choices=GROUPS,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    assert sha(args.source)==SOURCE_SHA
    root=Path(__file__).resolve().parents[3]
    relative=Path(__file__).relative_to(root).as_posix()
    blob=subprocess.check_output(['git','-C',str(root),'show',args.source_commit+':'+relative])
    assert hashlib.sha256(blob).hexdigest()==sha(Path(__file__))
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        p=output/name; tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(p)
    started=time.time()
    write('progress.json',dict(status='initializing'))
    try:
        torch.set_num_threads(4); torch.manual_seed(73)
        initial=torch.load(args.source,map_location='cpu',weights_only=False)
        assert initial['settings']['editor_rank']==16
        cfg=settings(initial['settings']); cfg.output_dir=output
        cfg.learning_rate=cfg.adapter_learning_rate=3e-5
        cfg.phase_learning_rate=cfg.router_learning_rate=6e-5
        cfg.readout_learning_rate=cfg.decoder_learning_rate=1e-4
        device=torch.device('cuda')
        model=architecture.build_model(cfg,device)
        model.load_state_dict(initial['model'],strict=True)
        model.set_phase_trainable(True)
        model.vision_stem.requires_grad_(False)
        gates=gate_state(model)
        assert len(gates)==4
        for n,p in model.named_parameters():
            if n in gates: p.requires_grad_(False)
        install_training_profile(model,args.group)
        train,test=t.build_loaders(cfg)
        assert len(train.dataset)==5000 and len(test.dataset)==1000
        epochs=1 if args.smoke else args.epochs
        steps=2 if args.smoke else len(train)
        protocol=dict(group=args.group,initial_sha256=SOURCE_SHA,initial_clean_cpu=.895,
            source_commit=args.source_commit,runtime_head=subprocess.check_output(
                ['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),source_sha256=sha(Path(__file__)),
            editor_rank=16,decoder_rank=64,head_parameters=sum(p.numel() for p in model.shared_readout.parameters()),
            decoder_parameters=sum(p.numel() for p in model.shared_readout.decoder.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
            fixed_alpha_logits={n:float(v) for n,v in gates.items()},epochs=epochs,steps_per_epoch=steps,
            train_count=5000,test_count=1000,seed=73,selection='complete clean TEST every5 highest development',
            test_gradient=False,validation_selection=False,profile=PROFILES[args.group],
            noise_scale=NOISE_SCALE,noise_offset_fraction=.15,noise_read_fraction=.05,
            noise_proxy_not_calibrated=True,dc_phase_leakage_intensity=.30 if PROFILES[args.group]['dc30'] else 0.,
            dc_planes='expert/global, historical backend',grid='TRAIN-only differentiable17→8→17 proxy, not true8um propagation',
            inference_profile='clean r0; no training noise/DC/grid/phase-dropout',phase_dropout=False,
            all_optical_electronic_original_parameters_train_except_alpha_frontend=True,
            ccd_reuse_forbidden=True,new_layers=False,command=__import__('sys').argv,
            gpu=torch.cuda.get_device_name(),manifest_sha256={s:sha(cfg.data_dir/(s+'.jsonl')) for s in ('train','test')})
        assert protocol['head_parameters']==179224 and protocol['decoder_parameters']==30162
        write('protocol.json',protocol); write('resolved_config.json',cfg.to_dict())
        optimizer=torch.optim.AdamW(t.legacy._parameter_groups(model,cfg),weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(epochs,1),eta_min=3e-6)
        def evaluate():
            model.eval(); t._set_phase_dropout(model,False)
            with torch.inference_mode(): return t.evaluate_with_routes(model,test,cfg,device)[0]
        def save(name,epoch,score):
            torch.save(dict(model={n:p.detach().cpu().clone() for n,p in model.state_dict().items()},
                epoch=epoch,group=args.group,settings=cfg.to_dict(),source_sha256=SOURCE_SHA,
                bounded_amplitude=dict(kind='tanh',scale=.5),fixed_alpha_logits=protocol['fixed_alpha_logits'],
                robust_training=protocol['profile'],selection=protocol['selection'],score=score),output/name)
        best,selected,history=-1.,-1,[]
        for epoch in range(epochs+1):
            losses=[]
            if epoch:
                model.train(); model.vision_stem.eval(); t._set_phase_dropout(model,False)
                for i,raw in enumerate(train):
                    batch=t.legacy._move(raw,device)
                    optimizer.zero_grad(set_to_none=True)
                    result=model(batch['source_image'],batch['prompt_hidden'])
                    loss=t.editing_objective(result,batch,cfg)['total']
                    loss+=cfg.router_importance_weight*model.router_importance_loss()
                    loss+=cfg.phase_dc_weight*t._phase_regularization(model,cfg)
                    assert torch.isfinite(loss)
                    loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                    optimizer.step(); losses.append(float(loss.detach()))
                    if i+1>=steps: break
                scheduler.step()
            metrics=evaluate() if epoch%5==0 or epoch==epochs else None
            score=metric(metrics) if metrics else None
            if score is not None and score>best:
                best,selected=score,epoch; save('best.pt',epoch,score)
            save('last.pt',epoch,score)
            assert all(torch.equal(gate_state(model)[n],v) for n,v in gates.items())
            row=dict(epoch=epoch,loss=sum(losses)/len(losses) if losses else None,
                clean_test=metrics,best=best,selected_epoch=selected,elapsed_seconds=time.time()-started)
            history.append(row); write('history.json',history)
            write('progress.json',dict(row,status='training'))
            print(json.dumps(dict(row,clean_test=score)),flush=True)
        saved=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
        model.to('cpu').load_state_dict(saved['model'],strict=True)
        device=torch.device('cpu')
        test=DataLoader(test.dataset,batch_size=1,collate_fn=test.collate_fn,num_workers=0)
        cpu=evaluate()
        write('report.json',dict(status='smoke_complete' if args.smoke else 'complete',
            group=args.group,original_test_cpu=cpu,selected_epoch=selected,best_gpu=best,
            best_sha256=sha(output/'best.pt'),last_sha256=sha(output/'last.pt'),
            fixed_alpha_unchanged=True,development_only=True,test_gradient=False,new_layers=False,
            physical_result=None,ccd_reuse_forbidden=True,source_commit=args.source_commit))
        write('progress.json',dict(status='complete',epoch=epochs,elapsed_seconds=time.time()-started))
    except Exception as error:
        write('progress.json',dict(status='failed',error=repr(error),elapsed_seconds=time.time()-started))
        raise


if __name__=='__main__': main()
