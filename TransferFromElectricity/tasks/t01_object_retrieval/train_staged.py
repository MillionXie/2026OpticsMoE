"""Validation-selected staged training with an enforced optical fusion floor."""
from __future__ import annotations
import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .run import (ROOT, TASK, write_json, sha256, git, cpu_state, grad_norm,
    load_settings, save_resolved_config, prepare_caltech101_subset, load_backbone,
    build_student, initialize_student, resolve_cached_model_source, seed_everything,
    environment_report, GroceryRetrievalDataset, collate_grocery, PKBatchSampler,
    preprocess_images, move_inputs, validate_token_budgets, student_embeddings,
    supervised_contrastive_loss, episodic_prototype_retrieval_loss,
    evaluate_student_split, initialize_parameter_ema, update_parameter_ema, use_parameter_ema,
    phase_dc_loss)
from .models.generator import StaticGenerator, LoRALinear
from .models.injection import ExpertInjection, expert_planes
from .protocol import common_anchor, split_train_validation, stage_at, active_groups, phase_summary, physical_phase


def create_generator(method, source, cfg, device):
    generator = StaticGenerator(method, source, device, cfg['generator']['rank'], seed=cfg['seed'])
    # Identical decoder weights for frozen-Qwen and LoRA-Qwen, independent of
    # random numbers consumed while constructing LoRA matrices.
    seed_everything(cfg['seed'] + 71)
    for module in generator.decoder.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
        elif isinstance(module, torch.nn.LayerNorm):
            module.reset_parameters()
    with torch.no_grad():
        generator.decoder.positions.normal_(std=.02)
        generator.decoder.decode[-1].weight.normal_(std=cfg['generator']['decoder_final_init_std'])
        generator.decoder.decode[-1].bias.zero_()
        for module in generator.encoder.modules():
            if isinstance(module, LoRALinear):
                module.lora_a.data = module.lora_a.data.float()
                module.lora_b.data = module.lora_b.data.float()
        generator.initial_reference.copy_(generator.decode_raw())
        generator.anchor.copy_(common_anchor(cfg['seed'], dc_power=cfg['initial_expert_dc_power']).to(device))
    return generator


def parameter_groups(replacement, readout, generator, method, cfg):
    grouped = {}
    for modality, surrogate in (('vision',replacement.vision_surrogate), ('language',replacement.language_surrogate)):
        for name, p in surrogate.named_parameters():
            if not p.requires_grad:
                continue
            if '.expert_layers.' in name and 'raw_phase' in name:
                if method == 'fixed':
                    p.requires_grad_(False)
                    continue
                category = 'expert'
            elif '.router.' in name:
                category = 'router'
            elif '.global_phase.' in name:
                category = 'global'
            elif 'optical_fusion_logit' in name:
                category = 'fusion'
            else:
                category = 'electronic'
            grouped.setdefault(category, []).append((f'{modality}.{name}', p))
    grouped['readout'] = [(f'readout.{n}', p) for n,p in readout.named_parameters() if p.requires_grad]
    if generator:
        for name,p in generator.named_parameters():
            if p.requires_grad:
                group = 'generator_decoder' if name.startswith('decoder.') else 'generator_context'
                grouped.setdefault(group, []).append((f'generator.{name}',p))
    groups = [{'params':[p for _,p in rows], 'parameter_names':[n for n,_ in rows],
               'group_name':name, 'lr':cfg['learning_rates'][name],
               'weight_decay': .001 if name in {'electronic','readout'} else 0.0}
              for name,rows in grouped.items() if rows]
    params = [p for g in groups for p in g['params']]
    if len(params) != len({id(p) for p in params}):
        raise RuntimeError('Duplicate optimizer parameters')
    return groups, params


def execute(args, cfg, output):
    seed_everything(cfg['seed'])
    torch.set_num_threads(4)
    device = torch.device('cuda')
    settings = load_settings(ROOT / cfg['backend_profile'])
    settings.output_dir = output
    settings.num_workers = cfg['num_workers']
    settings.router_optimization_seed = cfg['seed']
    settings.fusion_alpha_min = cfg['fusion']['minimum']
    settings.fusion_alpha_initial = cfg['fusion']['initial']
    settings.fusion_alpha_max = cfg['fusion']['maximum']
    if not .4 <= settings.fusion_alpha_min < settings.fusion_alpha_initial < settings.fusion_alpha_max <= 1:
        raise ValueError('Both modalities require an optical coefficient floor >=0.4')
    settings.epochs = sum(s['epochs'] for s in cfg['stages'])
    settings.evaluate_test_each_epoch = False
    settings.lightgen_test_selected = False
    if any((settings.lambda_kd, settings.lambda_relational_kd, settings.lambda_teacher_gallery)):
        raise ValueError('This experiment does not use a teacher loss')
    bundle = prepare_caltech101_subset(settings, persist=True)
    training, validation = split_train_validation(bundle.train_samples, cfg['seed'], cfg['validation_per_class'])
    partitions = {'train':training, 'validation':validation, 'gallery':bundle.gallery_samples, 'test':bundle.test_samples}
    idsets = [set(s.sample_id for s in rows) for rows in partitions.values()]
    if any(idsets[i] & idsets[j] for i in range(4) for j in range(i)):
        raise RuntimeError('Data partitions overlap')
    write_json(output/'split.json', {'original_manifest_sha256':bundle.manifest_digest,
        **{name:[s.manifest_record() for s in rows] for name,rows in partitions.items()}})
    write_json(output/'data_hashes.json', {str(s.image_path):sha256(s.image_path) for rows in partitions.values() for s in rows})
    loaded = load_backbone(settings, device)
    replacement, readout = build_student(loaded, settings)
    write_json(output/'initialization.json', initialize_student(settings,replacement,readout))
    save_resolved_config(settings)
    # The inherited serializer describes LightGen's historical test selection;
    # explicitly replace that metadata with this task's actual validation policy.
    resolved = yaml.safe_load((output/'config.yaml').read_text())
    resolved['lightgen']['selection'] = {'use_periodic_test':False, 'criterion':cfg['selection'], 'test_metrics_used_for_selection':False}
    resolved['static_expert_protocol'] = cfg
    (output/'config.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False), encoding='utf-8')
    planes = expert_planes(replacement)
    initial = common_anchor(cfg['seed'],dc_power=cfg['initial_expert_dc_power']).to(device)
    with torch.no_grad():
        for plane,raw in zip(planes,initial.flatten(0,1)):
            plane.raw_phase.copy_(raw)
    generator = injection = None
    if args.method.startswith('qwen'):
        source = args.generator_source or resolve_cached_model_source(cfg['generator']['model_id'],settings.cache_dir)
        generator = create_generator(args.method,source,cfg,device)
        injection = ExpertInjection(planes)
        source_path = Path(source)
        write_json(output/'generator_source_manifest.json', {'path':str(source_path), 'snapshot':source_path.name,
            'files':{p.name:sha256(p) for p in sorted(source_path.iterdir()) if p.is_file() and p.suffix in {'.json','.safetensors'}}})
    def bind(override=None):
        raw = override if override is not None else generator() if generator else torch.stack([p.raw_phase for p in planes]).reshape_as(initial)
        if injection:
            injection.bind(raw)
        elif override is not None:
            with torch.no_grad():
                for plane,value in zip(planes,raw.flatten(0,1)):
                    plane.raw_phase.copy_(value)
        return raw
    with torch.no_grad():
        if float((bind()-initial).abs().max()) > 1e-7:
            raise RuntimeError('Unpaired expert initialization')
    groups, parameters = parameter_groups(replacement,readout,generator,args.method,cfg)
    optimizer = torch.optim.AdamW(groups)
    ema = initialize_parameter_ema(parameters)
    write_json(output/'architecture.json', {'method':args.method,'base':replacement.student_architecture_report(),
        'generator_trainable':sum(p.numel() for p in generator.parameters() if p.requires_grad) if generator else 0,
        'initial_phase':phase_summary(initial,initial),'lora_modules':generator.lora_modules if generator else [],
        'optimizer_groups':[{**{k:v for k,v in g.items() if k!='params'},'parameter_count':sum(p.numel() for p in g['params'])} for g in groups]})
    current_sha = git('rev-parse','HEAD')
    write_json(output/'environment.json', {**environment_report(),'git_sha':current_sha,'git_status':git('status','--short'),
        'command':sys.argv,'device':torch.cuda.get_device_name(),'config_sha256':sha256(args.config),'split_sha256':sha256(output/'split.json')})
    seed_everything(cfg['seed']+1000)
    dataset = GroceryRetrievalDataset(training, settings.image_size, augment=settings.augmentation_enabled,
        crop_scale_min=settings.crop_scale_min,brightness_jitter=settings.brightness_jitter,
        contrast_jitter=settings.contrast_jitter,rotation_degrees=settings.rotation_degrees)
    sampler = PKBatchSampler(training,settings.pk_skus_per_batch,settings.pk_images_per_sku,cfg['seed'],args.steps_per_epoch)
    loader = DataLoader(dataset,batch_sampler=sampler,num_workers=settings.num_workers,collate_fn=collate_grocery)
    history, step_history, start_epoch, updates = [], [], 1, 0
    best_key = (-1.,-1.)
    started = time.perf_counter()
    def checkpoint(epoch, variant='live'):
        return {'schema_version':2,'git_sha':current_sha,'epoch':epoch,'weight_variant':variant,'config':cfg,
            'vision':cpu_state(replacement.vision_surrogate),'language':cpu_state(replacement.language_surrogate),'readout':cpu_state(readout),
            'generator':generator.compact_state() if generator else None,'generator_source':generator.source if generator else None,
            'expert_raw':bind().detach().cpu(),'optimizer':optimizer.state_dict(),'ema':[x.cpu() for x in ema],
            'history':history,'step_history':step_history,'best_key':best_key,'optimizer_updates':updates,
            'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
    def load_weights(payload):
        replacement.vision_surrogate.load_state_dict(payload['vision'])
        replacement.language_surrogate.load_state_dict(payload['language'])
        readout.load_state_dict(payload['readout'])
        if generator:
            generator.load_compact_state(payload['generator'])
        with torch.no_grad():
            if float((bind().cpu()-payload['expert_raw'].cpu()).abs().max()) > 1e-6:
                raise RuntimeError('Checkpoint expert reconstruction mismatch')
    if args.resume:
        payload = torch.load(output/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        if payload['git_sha'] != current_sha or payload['config'] != cfg:
            raise ValueError('Resume requires identical code and configuration')
        load_weights(payload)
        optimizer.load_state_dict(payload['optimizer'])
        ema = [x.to(device) for x in payload['ema']]
        history,step_history = payload['history'],payload['step_history']
        start_epoch,updates,best_key = payload['epoch']+1,payload['optimizer_updates'],tuple(payload['best_key'])
        random.setstate(payload['rng_python']); np.random.set_state(payload['rng_numpy'])
        torch.set_rng_state(payload['rng_torch']); torch.cuda.set_rng_state_all(payload['rng_cuda'])
        del payload
    def metrics(samples):
        with torch.no_grad():
            bind()
            return evaluate_student_split(loaded,replacement,readout,samples,bundle.gallery_samples,bundle.class_names,settings)
    def fusion_values():
        values = {m:[float(s.core.block1_optical_fusion),float(s.core.block2_optical_fusion)] for m,s in
                  [('vision',replacement.vision_surrogate),('language',replacement.language_surrogate)]}
        if min(v for row in values.values() for v in row) < .4:
            raise RuntimeError('Optical coefficient below required floor')
        return values
    try:
        if not args.resume:
            write_json(output/'initial_validation.json',metrics(validation))
        for epoch in range(start_epoch,settings.epochs+1):
            stage,relative_epoch = stage_at(epoch,cfg['stages'])
            enabled = active_groups(stage['name'])
            for group in groups:
                for p in group['params']:
                    p.requires_grad_(group['group_name'] in enabled)
            active = [p for p in parameters if p.requires_grad]
            frozen_before = [(p,p.detach().clone()) for p in parameters if not p.requires_grad]
            sampler.set_epoch(epoch)
            # Expert isolation uses deterministic optics, no optical noise/dropout.
            dataset.augment = settings.augmentation_enabled and stage['name'] != 'experts'
            if stage['name'] == 'experts':
                replacement.vision_surrogate.eval(); replacement.language_surrogate.eval(); readout.eval()
            else:
                replacement.set_student_train_mode()
                readout.train(stage['name']=='joint')
            if generator:
                generator.eval()
            counts = {'vision':[0]*4,'language':[0]*4}
            rows = []
            for step,batch in enumerate(loader):
                inputs = move_inputs(preprocess_images(loaded.processor,batch['images'],settings.instruction),device)
                validate_token_budgets(inputs,settings)
                labels = torch.tensor([s.sku_index for s in batch['samples']],device=device)
                n = relative_epoch*len(loader)+step
                total_steps = stage['epochs']*len(loader)
                scale = cfg['stage_cosine_minimum'] + (1-cfg['stage_cosine_minimum'])*.5*(1+math.cos(math.pi*n/max(total_steps-1,1)))
                scale *= min(1.,(n+1)/cfg['warmup_steps'])
                for group in groups:
                    group['lr'] = cfg['learning_rates'][group['group_name']]*scale if group['group_name'] in enabled else 0.
                optimizer.zero_grad(set_to_none=True)
                raw = bind()
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=settings.amp_enabled):
                    embedding,_ = student_embeddings(loaded.model,replacement,readout,inputs)
                    ret = supervised_contrastive_loss(embedding,labels,settings.temperature)
                    proto,_,_ = episodic_prototype_retrieval_loss(embedding,labels,settings.gallery_temperature)
                    task_loss = settings.lambda_ret*ret+settings.lambda_gallery*proto
                    routing = replacement.router_losses()
                    hard = replacement.router_hard_load_balance_loss()
                    regularizer = settings.lambda_router_balance*(routing['vision_balance']+routing['language_balance'])/2
                    regularizer += settings.lambda_router_importance*(routing['vision_importance']+routing['language_importance'])/2
                    regularizer += settings.lambda_router_hard_load_balance*(hard['vision']+hard['language'])/2
                    ccd = replacement.auxiliary_losses()['ccd_operating_point']
                    dc = phase_dc_loss(replacement)
                    loss = task_loss+regularizer+settings.lambda_ccd_operating_point*ccd+settings.lambda_phase_dc*dc
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                chain = {}
                if step == 0 and args.method != 'fixed':
                    targets = [raw] if generator else [p.raw_phase for p in planes]
                    grads = torch.autograd.grad(task_loss,targets,retain_graph=True)
                    chain['task_to_expert'] = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
                    if args.method == 'qwen_lora':
                        grads = torch.autograd.grad(task_loss,[p for n,p in generator.named_parameters() if n.endswith('lora_b')],retain_graph=True)
                        chain['task_to_lora_b'] = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
                    if min(chain.values()) <= 0:
                        raise RuntimeError('Broken task gradient chain')
                norms = {}
                if active:
                    loss.backward()
                    for group in groups:
                        norms[group['group_name']] = grad_norm(group['params'])
                        torch.nn.utils.clip_grad_norm_(group['params'],cfg['gradient_clip_per_group'],error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    update_parameter_ema(ema,parameters,cfg['ema_decay'])
                for m,s in [('vision',replacement.vision_surrogate),('language',replacement.language_surrogate)]:
                    selected = s.core.last_routing['selected_mask'].sum(0).cpu().tolist()
                    counts[m] = [a+int(b) for a,b in zip(counts[m],selected)]
                row = {'epoch':epoch,'stage':stage['name'],'step':step+1,'task_loss':float(task_loss.detach()),
                    'total_loss':float(loss.detach()),'phase_dc_loss':float(dc.detach()),'optimizer_updates':updates,
                    'gradient_norms':norms,'task_gradient_chain':chain,'elapsed_seconds':time.perf_counter()-started}
                rows.append(row); step_history.append(row)
                if step == 0 or step+1 == len(loader):
                    print(row,flush=True)
            live_val = metrics(validation)
            with use_parameter_ema(parameters,ema):
                ema_val = metrics(validation)
            with torch.no_grad():
                raw = bind()
                phase = phase_summary(raw,initial)
                frozen_delta = max((float((p-before).abs().max()) for p,before in frozen_before),default=0.)
                if frozen_delta != 0:
                    raise RuntimeError('A frozen parameter changed')
                stage_end_validation = None
                if relative_epoch+1 == stage['epochs']:
                    learned = raw.detach().clone()
                    bind(initial)
                    stage_end_validation = evaluate_student_split(loaded,replacement,readout,validation,bundle.gallery_samples,bundle.class_names,settings)
                    bind(learned)
            record = {'epoch':epoch,'stage':stage['name'],'mean_task_loss':sum(r['task_loss'] for r in rows)/len(rows),
                'live_validation':live_val,'ema_validation':ema_val,'expert_phase':phase,'fusion':fusion_values(),
                'expert_selection_counts':counts,'task_gradient_chain':rows[0]['task_gradient_chain'],
                'frozen_parameter_max_change':frozen_delta,'stage_end_validation_with_initial_experts':stage_end_validation,
                'optimizer_updates':updates,'elapsed_seconds':time.perf_counter()-started}
            history.append(record)
            key = (live_val['top1_retrieval_accuracy'],live_val['mrr'])
            if key > best_key:
                best_key = key
                torch.save(checkpoint(epoch),output/'best_checkpoint.pt')
            torch.save(checkpoint(epoch),output/'last_checkpoint.pt')
            write_json(output/'history.json',history)
            write_json(output/'steps.json',step_history)
            write_json(output/'status.json',{'status':'running','epoch':epoch,'stage':stage['name'],'epochs':settings.epochs})
            print({'epoch_result':record},flush=True)
        with use_parameter_ema(parameters,ema):
            final_ema_test = metrics(bundle.test_samples)
        payload = torch.load(output/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        load_weights(payload)
        selected_epoch = payload['epoch']
        del payload
        selected_test = metrics(bundle.test_samples)
        with torch.no_grad():
            learned = bind().detach().clone()
            selected_phase = phase_summary(learned,initial)
            def intervention(bank):
                bind(bank)
                return evaluate_student_split(loaded,replacement,readout,bundle.test_samples,bundle.gallery_samples,bundle.class_names,settings)
            ablations = {'initial_experts':intervention(initial),'flat_pi_experts':intervention(torch.zeros_like(initial))}
            bind(learned)
            if generator and args.method=='qwen_lora':
                lora_b = [p for n,p in generator.named_parameters() if n.endswith('lora_b')]
                backups = [p.clone() for p in lora_b]
                for p in lora_b: p.zero_()
                no_lora = generator().detach()
                ablations['lora_disabled_same_decoder'] = intervention(no_lora)
                ablations['lora_phase_effect'] = phase_summary(no_lora,learned)
                for p,b in zip(lora_b,backups): p.copy_(b)
                bind(learned)
            torch.save({'raw_phase':learned.cpu(),'phase_rad':physical_phase(learned).cpu(),
                'selected_epoch':selected_epoch,'git_sha':current_sha},output/'expert_bank.pt')
            # Verify deployed materialized masks agree with the generated path.
            audit = collate_grocery([dataset[i] for i in range(3)])
            inp = move_inputs(preprocess_images(loaded.processor,audit['images'],settings.instruction),device)
            replacement.vision_surrogate.eval(); replacement.language_surrogate.eval(); readout.eval()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=settings.amp_enabled):
                before = student_embeddings(loaded.model,replacement,readout,inp)[0].clone()
                if injection: injection.materialize()
                after = student_embeddings(loaded.model,replacement,readout,inp)[0].clone()
            export_error = float((before-after).abs().max())
            if export_error > 1e-5: raise RuntimeError('Materialization changed outputs')
        report = {'method':args.method,'protocol':'staged_alpha40','git_sha':current_sha,'split_sha256':sha256(output/'split.json'),
            'counts':{n:len(r) for n,r in partitions.items()},'selected_epoch':selected_epoch,'selection':cfg['selection'],
            'test_used_for_selection':False,'selected_live_test':selected_test,'final_ema_test':final_ema_test,
            'selected_expert_phase':selected_phase,'fusion':fusion_values(),'ablations':ablations,
            'optimizer_updates':updates,'training_batch_opportunities':settings.epochs*len(loader),
            'export_max_error':export_error,'expert_bank_sha256':sha256(output/'expert_bank.pt'),
            'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,'elapsed_seconds':time.perf_counter()-started}
        write_json(output/'final_report.json',report)
        print({'final_report':report},flush=True)
    finally:
        replacement.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--method',required=True,choices=['fixed','direct','qwen_frozen','qwen_lora'])
    parser.add_argument('--config',default=str(TASK/'configs/staged_alpha40.yaml'))
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--generator-source')
    parser.add_argument('--steps-per-epoch',type=int)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    cfg.update(method=args.method,smoke=args.smoke,steps_per_epoch=args.steps_per_epoch)
    if args.smoke:
        cfg['stages'] = [{'name':name,'epochs':1} for name in ('experts','optics','joint')]
        args.steps_per_epoch = cfg['steps_per_epoch'] = 2
    output = Path(args.run_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        import json
        if json.loads((output/'protocol.json').read_text()) != cfg:
            raise ValueError('Resume configuration differs')
    write_json(output/'protocol.json',cfg)
    write_json(output/'status.json',{'status':'running'})
    try:
        execute(args,cfg,output)
        write_json(output/'status.json',{'status':'complete'})
    except BaseException as exc:
        write_json(output/'status.json',{'status':'failed','error':repr(exc)})
        raise


if __name__ == '__main__':
    main()
