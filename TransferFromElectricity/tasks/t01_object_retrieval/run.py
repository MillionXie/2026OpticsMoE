"""Bounded, reproducible fixed-expert pilot using the unchanged LightGen graph."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from LightGenV2.tasks.t01_object_retrieval.settings import load_settings, save_resolved_config
from LightGenV2.tasks.t01_object_retrieval.modeling import build_student, initialize_student, load_backbone
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import prepare_caltech101_subset
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import preprocess_images, move_inputs, student_embeddings, validate_token_budgets
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import seed_everything, environment_report
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import resolve_cached_model_source
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import phase_dc_loss
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import GroceryRetrievalDataset, collate_grocery
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    PKBatchSampler, _build_optimizer, supervised_contrastive_loss, episodic_prototype_retrieval_loss,
    evaluate_student_split, initialize_parameter_ema, update_parameter_ema, use_parameter_ema,
    _learning_rate_scale, _apply_learning_rate_scale)
from .models.generator import StaticGenerator
from .models.injection import ExpertInjection, expert_planes

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def grad_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--method', choices=['direct', 'small_hyper', 'qwen_frozen', 'qwen_lora'], required=True)
    parser.add_argument('--config', default=str(TASK / 'configs/pilot.yaml'))
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--generator-source')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--steps-per-epoch', type=int)
    parser.add_argument('--train-per-class', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    for key in ('epochs', 'train_per_class', 'seed'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg.update(method=args.method, steps_per_epoch=args.steps_per_epoch)
    output = Path(args.run_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and json.loads((output / 'pilot_config.json').read_text()) != cfg:
        raise ValueError('Resume configuration differs from original run')
    write_json(output / 'pilot_config.json', cfg)
    write_json(output / 'status.json', {'status': 'running'})
    try:
        execute(args, cfg, output)
        write_json(output / 'status.json', {'status': 'complete'})
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


def execute(args, cfg, output):
    seed_everything(cfg['seed'])
    torch.set_num_threads(4)
    settings = load_settings(ROOT / cfg['backend_profile'])
    settings.output_dir = output
    settings.epochs = cfg['epochs']
    settings.num_workers = cfg['num_workers']
    settings.router_optimization_seed = cfg['seed']
    settings.optimizer_steps_per_epoch = args.steps_per_epoch
    if not settings.episodic_prototype_loss_enabled or any((settings.lambda_kd, settings.lambda_relational_kd, settings.lambda_teacher_gallery)):
        raise ValueError('Pilot requires the teacher-free episodic retrieval profile')
    if not torch.cuda.is_available():
        raise RuntimeError('Real-data pilot requires a CUDA GPU; use unittest for CPU smoke')
    device = torch.device('cuda')
    bundle = prepare_caltech101_subset(settings, persist=True)
    # Select only from the existing train split; preserve gallery and test.
    selected = []
    selection_rng = random.Random(cfg['seed'])
    for cls in range(len(bundle.class_names)):
        candidates = sorted((s for s in bundle.train_samples if s.sku_index == cls), key=lambda s: s.sample_id)
        selection_rng.shuffle(candidates)
        selected.extend(candidates[:cfg['train_per_class']])
    manifest = {'original_manifest_sha256': bundle.manifest_digest,
                'train': [s.manifest_record() for s in selected],
                'gallery': [s.manifest_record() for s in bundle.gallery_samples],
                'test': [s.manifest_record() for s in bundle.test_samples]}
    write_json(output / 'pilot_split.json', manifest)
    write_json(output / 'data_hashes.json', {str(s.image_path): sha256(s.image_path)
        for s in (*selected, *bundle.gallery_samples, *bundle.test_samples)})
    loaded = load_backbone(settings, device)
    replacement, readout = build_student(loaded, settings)
    write_json(output / 'initialization.json', initialize_student(settings, replacement, readout))
    save_resolved_config(settings)
    planes = expert_planes(replacement)
    seed_everything(cfg['seed'])
    generator = None
    injection = None
    if args.method != 'direct':
        source = args.generator_source or resolve_cached_model_source(cfg['generator']['model_id'], settings.cache_dir)
        generator = StaticGenerator(args.method, source, device, cfg['generator']['rank'], seed=cfg['seed'])
        if args.method.startswith('qwen'):
            source_path = Path(source)
            write_json(output / 'generator_source_manifest.json', {
                'path': str(source_path), 'snapshot': source_path.name,
                'files': {p.name: sha256(p) for p in sorted(source_path.iterdir())
                          if p.is_file() and (p.suffix in {'.json', '.safetensors'})}})
        injection = ExpertInjection(planes)
        initial = generator.anchor
    else:
        initial = torch.randn(2, 4, 224, 224, generator=torch.Generator().manual_seed(cfg['seed'])).to(device) * 0.02
        with torch.no_grad():
            for plane, value in zip(planes, initial.flatten(0, 1)):
                plane.raw_phase.copy_(value)
    optimizer, parameters = _build_optimizer(replacement, readout, settings)
    if generator is not None:
        for name, values, lr in (
            ('generator_context', [p for n, p in generator.named_parameters() if p.requires_grad and not n.startswith('decoder.')], cfg['generator']['learning_rate']),
            ('generator_decoder', list(generator.decoder.parameters()), cfg['generator']['decoder_learning_rate'])):
            if values:
                optimizer.add_param_group({'params': values, 'lr': lr, 'configured_lr': lr, 'group_name': name})
                parameters.extend(values)
    if len({id(p) for p in parameters}) != len(parameters):
        raise RuntimeError('Optimizer duplicate parameters')
    def bind():
        raw = generator() if generator else torch.stack([p.raw_phase for p in planes]).reshape(2, 4, 224, 224)
        if injection:
            injection.bind(raw)
        return raw
    with torch.no_grad():
        raw = bind()
        initial_error = float((raw - initial).abs().max())
    if initial_error > 1e-7:
        raise RuntimeError(f'Unpaired initialization: {initial_error}')
    write_json(output / 'architecture.json', {
        'base': replacement.student_architecture_report(), 'method': args.method,
        'static_bank_shape': list(raw.shape), 'initial_max_error': initial_error,
        'generation_condition': 'fixed task/modality/expert descriptors; no samples or labels',
        'generator_source': generator.source if generator else None,
        'lora_modules': generator.lora_modules if generator else [],
        'generator_trainable': sum(p.numel() for p in generator.parameters() if p.requires_grad) if generator else 0,
        'total_trainable': sum(p.numel() for p in parameters),
        'optimizer_groups': [{k: v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups]})
    write_json(output / 'environment.json', {**environment_report(), 'git_sha': git('rev-parse', 'HEAD'),
        'git_status': git('status', '--short'), 'command': sys.argv, 'device': torch.cuda.get_device_name(),
        'source_config_sha256': sha256(args.config), 'split_sha256': sha256(output / 'pilot_split.json')})
    # Reset augmentation/optics RNG after constructing different generators.
    seed_everything(cfg['seed'] + 1000)
    dataset = GroceryRetrievalDataset(selected, settings.image_size, augment=settings.augmentation_enabled,
        crop_scale_min=settings.crop_scale_min, brightness_jitter=settings.brightness_jitter,
        contrast_jitter=settings.contrast_jitter, rotation_degrees=settings.rotation_degrees)
    sampler = PKBatchSampler(selected, settings.pk_skus_per_batch, settings.pk_images_per_sku, cfg['seed'], args.steps_per_epoch)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=settings.num_workers, collate_fn=collate_grocery)
    ema = initialize_parameter_ema(parameters)
    start_epoch = 1
    history = []
    route_counts = {'vision': [0]*4, 'language': [0]*4}
    if args.resume:
        payload = torch.load(output / 'last_checkpoint.pt', map_location=device, weights_only=False)
        if payload['git_sha'] != git('rev-parse', 'HEAD'):
            raise ValueError('Resume requires the same code commit')
        replacement.vision_surrogate.load_state_dict(payload['vision'])
        replacement.language_surrogate.load_state_dict(payload['language'])
        readout.load_state_dict(payload['readout'])
        if generator:
            generator.load_compact_state(payload['generator'])
        optimizer.load_state_dict(payload['optimizer'])
        ema = [t.to(device) for t in payload['ema']]
        start_epoch = payload['epoch'] + 1
        history = payload['history']
        route_counts = payload['route_counts']
        random.setstate(payload['rng_python'])
        np.random.set_state(payload['rng_numpy'])
        torch.set_rng_state(payload['rng_torch'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in payload['rng_cuda']])
    started = time.perf_counter()
    def checkpoint(epoch):
        return {'schema_version': 1, 'git_sha': git('rev-parse', 'HEAD'), 'epoch': epoch, 'config': cfg,
            'vision': cpu_state(replacement.vision_surrogate), 'language': cpu_state(replacement.language_surrogate),
            'readout': cpu_state(readout), 'generator': generator.compact_state() if generator else None,
            'generator_source': generator.source if generator else None,
            'optimizer': optimizer.state_dict(), 'ema': [t.cpu() for t in ema], 'history': history,
            'route_counts': route_counts,
            'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
            'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}
    audit_inputs = None
    try:
        for epoch in range(start_epoch, cfg['epochs'] + 1):
            sampler.set_epoch(epoch)
            replacement.set_student_train_mode()
            readout.train()
            if generator:
                generator.eval()  # deterministic static bank with autograd enabled
            for step, batch in enumerate(loader, 1):
                inputs = move_inputs(preprocess_images(loaded.processor, batch['images'], settings.instruction), device)
                validate_token_budgets(inputs, settings)
                if audit_inputs is None:
                    audit_inputs = inputs
                labels = torch.tensor([s.sku_index for s in batch['samples']], device=device)
                _apply_learning_rate_scale(optimizer, _learning_rate_scale(settings, (epoch-1)*len(loader)+step-1, cfg['epochs']*len(loader)), False)
                optimizer.zero_grad(set_to_none=True)
                raw = bind()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=settings.amp_enabled):
                    embedding, _ = student_embeddings(loaded.model, replacement, readout, inputs)
                    ret = supervised_contrastive_loss(embedding, labels, settings.temperature)
                    gallery, _, _ = episodic_prototype_retrieval_loss(embedding, labels, settings.gallery_temperature)
                    task_loss = settings.lambda_ret * ret + settings.lambda_gallery * gallery
                    routing = replacement.router_losses()
                    hard = replacement.router_hard_load_balance_loss()
                    ccd = replacement.auxiliary_losses()['ccd_operating_point']
                    total = task_loss + settings.lambda_router_balance * (routing['vision_balance'] + routing['language_balance']) / 2
                    total = total + settings.lambda_router_importance * (routing['vision_importance'] + routing['language_importance']) / 2
                    total = total + settings.lambda_router_hard_load_balance * (hard['vision'] + hard['language']) / 2
                    total = total + settings.lambda_ccd_operating_point * ccd + settings.lambda_phase_dc * phase_dc_loss(replacement)
                if not torch.isfinite(total):
                    raise RuntimeError('Nonfinite loss')
                for name, surrogate in [('vision', replacement.vision_surrogate), ('language', replacement.language_surrogate)]:
                    counts = surrogate.core.last_routing['selected_mask'].sum(0).cpu().tolist()
                    route_counts[name] = [a+int(b) for a,b in zip(route_counts[name], counts)]
                if not history:
                    targets = [p.raw_phase for p in planes] if generator is None else [raw]
                    task_grads = torch.autograd.grad(task_loss, targets, retain_graph=True)
                    diagnostic = {'task_to_expert_grad_norm': float(torch.stack([x.float().square().sum() for x in task_grads]).sum().sqrt())}
                    if args.method == 'qwen_lora':
                        lora_b = [p for n, p in generator.named_parameters() if n.endswith('lora_b')]
                        grads = torch.autograd.grad(task_loss, lora_b, retain_graph=True)
                        diagnostic['task_to_lora_b_grad_norm'] = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
                        if diagnostic['task_to_lora_b_grad_norm'] <= 0:
                            raise RuntimeError('Task loss does not reach LoRA')
                    write_json(output / 'gradient_chain.json', diagnostic)
                total.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise RuntimeError('Nonfinite gradient')
                norm = grad_norm(parameters)
                if settings.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip_norm)
                optimizer.step()
                update_parameter_ema(ema, parameters, cfg['ema_decay'])
                row = {'epoch': epoch, 'step': step, 'loss': float(total), 'task_loss': float(task_loss),
                    'gradient_norm': norm, 'elapsed_seconds': time.perf_counter()-started,
                    'peak_memory_gib': torch.cuda.max_memory_allocated()/1024**3}
                history.append(row)
                with (output / 'train_log.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
                if step == 1 or step == len(loader):
                    print(json.dumps(row), flush=True)
            torch.save(checkpoint(epoch), output / 'last_checkpoint.pt')
        with use_parameter_ema(parameters, ema), torch.no_grad():
            raw = bind().detach()
            metrics = evaluate_student_split(loaded, replacement, readout, bundle.test_samples, bundle.gallery_samples, bundle.class_names, settings)
            best = checkpoint(cfg['epochs'])
            best['weight_variant'] = 'ema_final_epoch'
            best['metrics'] = metrics
            torch.save(best, output / 'best_checkpoint.pt')
            bank = {'raw_phase': raw.cpu(), 'phase_rad': (2*torch.pi*torch.sigmoid(raw)).cpu(), 'shape': list(raw.shape)}
            torch.save(bank, output / 'expert_bank.pt')
            phase_motion = float((2*torch.pi*(torch.sigmoid(raw)-torch.sigmoid(initial))).square().mean().sqrt())
            # Export agreement: consume plain phase Parameters with no generator.
            replacement.vision_surrogate.eval(); replacement.language_surrogate.eval(); readout.eval()
            if audit_inputs is None:
                batch = collate_grocery([dataset[i] for i in range(min(3, len(dataset)))])
                audit_inputs = move_inputs(preprocess_images(loaded.processor, batch['images'], settings.instruction), device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=settings.amp_enabled):
                expected, _ = student_embeddings(loaded.model, replacement, readout, audit_inputs)
                expected = expected.clone()
                if injection:
                    injection.materialize()
                actual, _ = student_embeddings(loaded.model, replacement, readout, audit_inputs)
            export_error = float((actual-expected).abs().max())
            if export_error > 1e-5:
                raise RuntimeError(f'Export disagreement: {export_error}')
            # Real task batch-independence check using separately processed images.
            check_images = [dataset[i]['image'] for i in range(3)]
            def audit_encode(images):
                x = move_inputs(preprocess_images(loaded.processor, images, settings.instruction), device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=settings.amp_enabled):
                    return student_embeddings(loaded.model, replacement, readout, x)[0].clone()
            together = audit_encode(check_images)
            alone = audit_encode(check_images[:1])
            reversed_embeddings = audit_encode(check_images[::-1])
            batch_error = float((together[:1]-alone).abs().max())
            order_error = float((together-reversed_embeddings.flip(0)).abs().max())
            # BF16 task adapters can differ slightly with GEMM batch shape.
            if max(batch_error, order_error) > 5e-3:
                raise RuntimeError(f'Batch-dependence audit failed: {batch_error}, {order_error}')
            route_report = {}
            for name, surrogate in [('vision', replacement.vision_surrogate), ('language', replacement.language_surrogate)]:
                routing = surrogate.core.last_routing
                route_report[name] = routing['selected_mask'].sum(0).cpu().tolist()
            write_json(output / 'final_report.json', {'metrics': metrics, 'method': args.method,
                'selection': cfg['selection'], 'test_used_for_selection': False,
                'train_samples': len(selected), 'test_samples': len(bundle.test_samples), 'gallery_samples': len(bundle.gallery_samples),
                'optimizer_steps': len(history), 'export_max_error': export_error,
                'batch_max_error': batch_error, 'order_max_error': order_error,
                'physical_phase_rms_change_rad': phase_motion,
                'train_expert_selection_counts': route_counts,
                'last_audit_batch_expert_counts': route_report,
                'expert_bank_sha256': sha256(output/'expert_bank.pt'),
                'peak_memory_gib': torch.cuda.max_memory_allocated()/1024**3,
                'elapsed_seconds': time.perf_counter()-started, 'git_sha': git('rev-parse','HEAD')})
            print(json.dumps({'final_metrics': metrics, 'export_max_error': export_error}), flush=True)
    finally:
        replacement.close()


if __name__ == '__main__':
    main()
