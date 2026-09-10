"""Same-graph LSP continuation: joint control versus staged optimization."""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router import training as base
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.protocol import build_periodic_test_protocol, persist_protocol
from .modeling import architecture_label, architecture_report, build_student, load_vision_backbone, optimizer, sha256_file
from .run import _seed
from .settings import load_settings, save_resolved_config
from .training import _bind

TASK = Path(__file__).resolve().parent
PROFILES = ('joint', 'staged', 'staged_heatmap', 'alpha50', 'alpha40')
HIGH_SOURCE_REL = TASK/'runs/simulation/refinement_20260909/staged_heatmap/best_checkpoint.pt'
HIGH_SOURCE_SHA = '495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518'
SOURCE_REL = TASK / 'runs/simulation/moe_router_scale_dc20_no_shift_warmstart0713_seed42/best_checkpoint.pt'
SOURCE_SHA = 'fc1c6be4196593d7f27d83097c5bfd53b6e0511e84726fd55d712af0a4cd7741'
RATES = {'electronic': 2e-5, 'router': 3e-4, 'feature_phase': 3e-3,
         'ccd_readout': 5e-5, 'pose_head': 1e-4}


def stage_spec(profile: str, epoch: int) -> tuple[str, dict[str, float]]:
    if profile not in PROFILES or not 1 <= epoch <= 60:
        raise ValueError('Unknown profile or epoch outside 1..60')
    if profile == 'alpha40':
        if epoch <= 3:
            return 'alpha40_readout_adaptation', {k:(1e-4 if k in ('pose_head','ccd_readout') else 0.) for k in RATES}
        if epoch <= 50:
            scale = .2 + .8*.5*(1+math.cos(math.pi*(epoch-4)/46))
            rates = {'electronic': 2e-5, 'router': 3e-4, 'feature_phase': 2e-3,
                     'ccd_readout': 1e-4, 'pose_head': 1e-4}
            return 'alpha40_joint', {k:v*scale for k,v in rates.items()}
        return 'alpha40_fixed_optics_polish', {k:(2e-5 if k in ('pose_head','ccd_readout') else 0.) for k in RATES}
    if profile == 'alpha50':
        if epoch <= 10:
            return 'high_alpha_optics_adaptation', {'electronic': 0., 'router': 1e-3,
                'feature_phase': 6e-3, 'ccd_readout': 1e-4, 'pose_head': 2e-4}
        if epoch <= 50:
            scale = .2 + .8*.5*(1+math.cos(math.pi*(epoch-11)/39))
            rates = {'electronic': 1e-5, 'router': 3e-4, 'feature_phase': 3e-3,
                     'ccd_readout': 1e-4, 'pose_head': 1e-4}
            return 'high_alpha_joint', {k:v*scale for k,v in rates.items()}
        return 'high_alpha_fixed_optics_polish', {k:(2e-5 if k in ('pose_head','ccd_readout') else 0.) for k in RATES}
    if profile == 'joint':
        scale = .1 + .9 * .5 * (1 + math.cos(math.pi * (epoch-1)/59))
        return 'joint_control', {k: v*scale for k,v in RATES.items()}
    if epoch <= 5:
        return 'head_calibration', {k: (v if k == 'pose_head' else 0.) for k,v in RATES.items()}
    if epoch <= 15:
        return 'optics_readout', {k: (0. if k == 'electronic' else (6e-3 if k == 'feature_phase' else v)) for k,v in RATES.items()}
    if epoch <= 50:
        scale = .2 + .8 * .5 * (1+math.cos(math.pi*(epoch-16)/34))
        return 'joint_refinement', {k: v*scale for k,v in RATES.items()}
    return 'fixed_optics_polish', {k: (2e-5 if k in ('pose_head','ccd_readout') else 0.) for k in RATES}


def apply_stage(opt: torch.optim.Optimizer, rates: dict[str, float]) -> dict:
    if {g['name'] for g in opt.param_groups} != set(rates):
        raise ValueError('Stage must specify every optimizer group exactly')
    report = {}
    for group in opt.param_groups:
        rate = float(rates[group['name']])
        if rate < 0: raise ValueError('Negative learning rate')
        group['lr'] = rate
        for p in group['params']:
            p.requires_grad_(rate > 0)
            p.grad = None
        report[group['name']] = {'lr': rate, 'trainable_parameters': sum(p.numel() for p in group['params'] if p.requires_grad)}
    return report


def phase_delta(model, reference):
    values = {}
    for name,p in model.core.named_parameters():
        if name not in reference: continue
        raw = p.detach().float().cpu()
        diff = 2*math.pi*(raw.sigmoid()-reference[name].sigmoid())
        values[name] = {'raw_rms_change': float((raw-reference[name]).square().mean().sqrt()),
                        'physical_phase_rms_change_rad': float(diff.square().mean().sqrt())}
    return values


def checked_fusion(model, settings):
    values = {s:float(getattr(model.core.hybrid,s).detach())
              for s in ('block1_optical_fusion','block2_optical_fusion')}
    if not all(math.isfinite(v) and settings.fusion_alpha_min <= v <= settings.fusion_alpha_max+1e-7
               for v in values.values()):
        raise RuntimeError(f'Fusion violates configured range: {values}')
    return values


def run(args):
    _bind()
    config = TASK/'configs/moe_optical_router_scale_matched_dc20_no_shift_warmstart.yaml'
    source_settings = load_settings(config)
    high_alpha = args.profile in ('alpha50', 'alpha40')
    settings = load_settings(TASK/f'configs/moe_{args.profile}.yaml') if high_alpha else source_settings
    source_sha = HIGH_SOURCE_SHA if high_alpha else SOURCE_SHA
    if args.source is None:
        args.source = HIGH_SOURCE_REL if high_alpha else SOURCE_REL
    if sha256_file(args.source) != source_sha:
        raise RuntimeError('Source checkpoint differs from the profile-pinned candidate')
    payload = torch.load(args.source,map_location='cpu',weights_only=False)
    if payload.get('checkpoint_architecture') != architecture_label(source_settings) or payload.get('router_contract_sha256') != settings.router_contract_sha256:
        raise RuntimeError('Source architecture or optical router contract mismatch')
    if payload.get('weight_variant') != 'ema': raise RuntimeError('Source must be EMA')
    settings.output_dir=args.run_dir.resolve()
    settings.output_dir.mkdir(parents=True,exist_ok=False)
    settings.data_root=args.data_root.resolve()
    settings.download=False
    settings.cache_dir=args.cache_dir.resolve()
    settings.student_epochs=60
    settings.student_batch_size=args.batch_size
    settings.inference_batch_size=args.batch_size
    settings.num_workers=args.workers
    settings.random_seed=args.seed
    settings.visualization_sample_count=0
    settings.log_interval_batches=100
    _seed(args.seed)
    bundle=build_periodic_test_protocol(prepare_lsp(settings,persist=False))
    persist_protocol(bundle,settings.output_dir)
    if args.smoke:
        bundle=copy.copy(bundle)
        # Protocol dataclass may be frozen; replace its lists without changing the original split.
        import dataclasses
        bundle=dataclasses.replace(bundle,train=bundle.train[:4],test=bundle.test[:4])
    device=torch.device(args.device)
    loaded=load_vision_backbone(settings,device)
    model=build_student(loaded,settings)
    model.core.load_state_dict(payload['core'],strict=True)
    model.head.load_state_dict(payload['head'],strict=True)
    if high_alpha:
        model.core.hybrid.reset_fusion_logits(settings.fusion_alpha_initial)
    initial_fusion = checked_fusion(model,settings)
    opt=optimizer(model,settings)
    parameter_budget = {g['name']:sum(p.numel() for p in g['params']) for g in opt.param_groups}
    if high_alpha and parameter_budget != {'electronic':616423, 'router':50176,
            'feature_phase':429188, 'ccd_readout':86400, 'pose_head':133425}:
        raise RuntimeError(f'High-alpha continuation changed the source parameter budget: {parameter_budget}')
    ema=base.ModelEMA(model.core,model.head,settings.ema_decay)
    reference={k:p.detach().float().cpu().clone() for k,p in model.core.named_parameters()
               if k.endswith(('raw_phase', 'raw_router_phase'))}
    out=settings.output_dir
    def write(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    manifest={'source':str(args.source.resolve()),'source_sha256':source_sha,'source_epoch':payload['epoch'],
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip(),
              'args':vars(args),'architecture_unchanged':not high_alpha,'topology_unchanged':True,'new_layers':0,
              'parameter_budget':parameter_budget,
              'fusion_contract':{'min':settings.fusion_alpha_min,'max':settings.fusion_alpha_max,
                                 'initial_actual':initial_fusion,'reset_from_source':high_alpha},
              'train_samples':len(bundle.train),'test_samples':len(bundle.test),
              'test_selected':True,'coordinate_loss_train_weight':0. if args.profile in ('staged_heatmap','alpha50','alpha40') else settings.coordinate_loss_weight,
              'target_pck_at_0.2':0.73 if args.profile == 'alpha40' else None,
              'schedule':[{'epoch':e,'stage':stage_spec(args.profile,e)[0],'lr':stage_spec(args.profile,e)[1]} for e in range(1,61)],
              'torch':torch.__version__,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'}
    write('run_manifest.json',manifest)
    write('student_architecture.json',architecture_report(model,settings))
    train_loader=base._loader(bundle.train,settings,training=True)
    test_loader=base._loader(bundle.test,settings,training=False)
    eval_settings=copy.copy(settings)
    if args.profile in ('staged_heatmap','alpha50','alpha40'): settings.coordinate_loss_weight=0.
    save_resolved_config(settings)
    def evaluate(epoch,phase):
        return base.evaluate_model(model,'student',test_loader,loaded.processor,device,eval_settings,phase=phase,epoch=epoch,save_outputs=False,tta=False)[0]
    def save(path,epoch,train,test,variant='ema',selected=True):
        checked_fusion(model,settings)
        base._save_checkpoint(path,model,settings,epoch=epoch,train_metrics=train,periodic_test_metrics=test,
                              initialization_report=manifest,weight_variant=variant,selected_by_periodic_test=selected)
    history=[]; best_epoch=0; last_test={}
    model.core.set_phase_dropout_active(True)
    try:
        anchor=evaluate(0,'source_anchor_test')
        write('source_anchor_test.json',anchor)
        print('SOURCE_ANCHOR',json.dumps(anchor),'FUSION',initial_fusion,flush=True)
        best_key=base._selection_key(anchor,0)
        save(out/'best_checkpoint.pt',0,{},anchor)
        total=1 if args.smoke else 60
        for epoch in range(1,total+1):
            started=time.perf_counter()
            name,rates=stage_spec(args.profile,epoch)
            if args.smoke and not high_alpha: name,rates=stage_spec('joint',1)
            group_report=apply_stage(opt,rates)
            model.core.router.set_noise_std(settings.router_noise_std*(1-epoch/60))
            train=base._train_epoch(model,'student',train_loader,loaded.processor,device,opt,settings,epoch,ema=ema)
            row={'epoch':epoch,'stage':name,'optimizer_groups':group_report,'train':train,
                 'live_phase_delta':phase_delta(model,reference),'live_fusion':checked_fusion(model,settings)}
            if epoch==1 or epoch%5==0 or epoch==total:
                with ema.applied():
                    last_test=evaluate(epoch,'refinement_periodic_test')
                    key=base._selection_key(last_test,epoch)
                    if key<best_key:
                        best_key=key; best_epoch=epoch
                        save(out/'best_checkpoint.pt',epoch,train,last_test)
                    row['test']=last_test
                    row['ema_phase_delta']=phase_delta(model,reference)
                    row['fusion']=checked_fusion(model,settings)
            save(out/'last_checkpoint.pt',epoch,train,last_test,'live_last',False)
            row['seconds']=time.perf_counter()-started
            row['best_epoch']=best_epoch
            history.append(row); write('training_history.json',history)
            print('EPOCH',epoch,name,'train_loss',train['loss'],'test_PCK',row.get('test',{}).get('pck_at_0.2_torso'),'best_epoch',best_epoch,flush=True)
        selected=torch.load(out/'best_checkpoint.pt',map_location=device,weights_only=False)
        model.core.load_state_dict(selected['core'],strict=True); model.head.load_state_dict(selected['head'],strict=True)
        normal=evaluate(best_epoch,'selected_best_test')
        model.core.hybrid.set_fusion_ablation('remove_optical')
        off=evaluate(best_epoch,'selected_best_optical_off_counterfactual')
        model.core.hybrid.set_fusion_ablation('none')
        write('final_report.json',{'best_epoch':best_epoch,'source_test':anchor,'test':normal,'optical_off':off,
                                  'fusion':checked_fusion(model,settings),'fusion_contract':manifest['fusion_contract'],
                                  'phase_change':phase_delta(model,reference),'checkpoint_sha256':sha256_file(out/'best_checkpoint.pt'),
                                  'optical_off_is_retraining':False,'smoke_only':args.smoke,
                                  'target_pck_at_0.2':manifest['target_pck_at_0.2'],
                                  'target_met':(normal['pck_at_0.2_torso'] >= .73 and not args.smoke) if args.profile == 'alpha40' else None})
    finally:
        model.core.set_phase_dropout_active(False); model.restore_native()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=PROFILES,required=True)
    p.add_argument('--source',type=Path,default=None)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    run(p.parse_args())


if __name__=='__main__': main()
