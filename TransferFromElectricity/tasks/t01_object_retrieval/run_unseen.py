"""Class-disjoint CIFAR transfer: frozen-bank evaluation and expert-only adaptation.

Source training began with zero raw phases. Adaptation explicitly CONTINUES its
learned phases; it is not a new zero-initialized source training experiment.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from . import train_staged as base
from .datasets import prepare_cifar100
from .models.injection import ExpertInjection, expert_planes, global_planes
from .models.spatial_generator import SpatialGenerator, reference_images
from .unseen_protocol import validate_classes, select_support
from .run_spatial_suite import inventory, processes
from .launch_rtx import validate_device


def execute(args, cfg, output):
    expected = validate_device(args.gpu_uuid, inventory())
    if processes().get(args.gpu_uuid):
        raise RuntimeError('Assigned GPU already has a compute process')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from .deterministic_ops import install_deterministic_pooling
    install_deterministic_pooling()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = torch.device('cuda')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != expected:
        raise RuntimeError('GPU binding mismatch')
    base.seed_everything(args.support_seed)
    validate_classes(cfg['source_class_ids'], cfg['novel_class_sets'])
    source_run = base.TASK / cfg['source_runs'][args.method]
    source_protocol = json.loads((source_run / 'protocol.json').read_text())
    if source_protocol['dataset']['class_ids'] != cfg['source_class_ids']:
        raise RuntimeError('Source training class identity differs')
    if source_protocol['phase_initialization'] != 'zero_raw_all_optical':
        raise RuntimeError('Source experiment did not start with zero phases')
    source_path = source_run / 'best_checkpoint.pt'
    payload = torch.load(source_path, map_location='cpu', weights_only=False)
    if payload['git_sha'] != 'c024b9280433f6e7fe31fc0122a1b8aadf342b38':
        raise RuntimeError('Unexpected source checkpoint provenance')
    settings = base.load_settings(base.ROOT / cfg['backend_profile'])
    settings.output_dir = output
    settings.num_workers = 0
    settings.router_optimization_seed = 42
    settings.instruction = 'Represent this image for cifar100 image-to-image retrieval.'
    settings.fusion_alpha_min = cfg['fusion']['minimum']
    settings.fusion_alpha_initial = cfg['fusion']['initial']
    settings.fusion_alpha_max = cfg['fusion']['maximum']
    data_cfg = {**cfg['dataset'], 'class_ids': cfg['novel_class_sets'][args.class_set], 'gallery_per_class': 0}
    bundle = prepare_cifar100(data_cfg, base.ROOT, output, 42)
    support = select_support(bundle.train_samples, args.shots, args.support_seed)
    support_ids = {s.sample_id for s in support}
    queries = list(bundle.test_samples)
    if args.smoke:
        queries = [next(s for s in bundle.train_samples if s.sku_index == i and s.sample_id not in support_ids)
                   for i in range(10)]
    assert not support_ids.intersection(s.sample_id for s in queries)
    settings.selected_skus = bundle.class_names
    base.write_json(output / 'split.json', {'source_class_ids': cfg['source_class_ids'],
        'novel_class_ids': data_cfg['class_ids'], 'class_names': bundle.class_names,
        'support': [s.manifest_record() for s in support], 'query': [s.manifest_record() for s in queries],
        'gallery_policy': 'All K support images per class; same images used for gradient adaptation and gallery; no additional labels',
        'query_policy': 'disjoint official_train smoke' if args.smoke else 'all selected official_test images'})
    base.write_json(output / 'data_hashes.json', {str(s.image_path): base.sha256(s.image_path) for s in support + queries})
    base.write_json(output / 'source_checkpoint.json', {'run_id': source_run.name, 'path': str(source_path),
        'sha256': base.sha256(source_path), 'git_sha': payload['git_sha'], 'selected_epoch': payload['epoch'],
        'initialization': 'Source began with raw=0; adaptation retains learned source phases and other source weights'})
    loaded = base.load_backbone(settings, device)
    replacement, readout = base.build_student(loaded, settings)
    generator = injection = None
    planes = expert_planes(replacement)
    if args.method == 'qwen_vision_lora':
        images, refs = reference_images(support, args.support_seed)
        generator = SpatialGenerator(args.method, payload['generator_source'], images, device, rank=8, seed=42)
        # Source buffers include its original reference pixels and centering offset.
        novel_pixels = generator.pixels.detach().clone()
        novel_grid = generator.grid_thw.detach().clone()
        generator.load_compact_state(payload['generator'])
        injection = ExpertInjection(planes)
        base.write_json(output / 'fixed_references.json', {'references': refs, 'policy': 'one fixed novel support image per class'})
    replacement.vision_surrogate.load_state_dict(payload['vision'], strict=True)
    replacement.language_surrogate.load_state_dict(payload['language'], strict=True)
    readout.load_state_dict(payload['readout'], strict=True)
    source_raw = payload['expert_raw'].to(device)
    source_global = payload['global_raw'].to(device)
    del payload

    def bind():
        raw = generator() if generator else torch.stack([p.raw_phase for p in planes]).reshape(2, 4, 224, 224)
        if injection:
            injection.bind(raw)
        return raw

    def evaluate(label):
        from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import encode_student_samples
        from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.retrieval_metrics import evaluate_embeddings
        with torch.no_grad():
            bind()
            gallery = encode_student_samples(loaded, replacement, readout, support, settings)
            query = encode_student_samples(loaded, replacement, readout, queries, settings)
            result = evaluate_embeddings(query, queries, gallery, support, bundle.class_names,
                                         settings.gallery_aggregation, system_name=label)
            base.write_json(output / f'{label}_predictions.json', result.rows)
            base.write_json(output / f'{label}_confusion.json', result.confusion.tolist())
            return result.metrics

    try:
        replacement.vision_surrogate.eval(); replacement.language_surrogate.eval(); readout.eval()
        with torch.no_grad():
            reconstruction = float((bind() - source_raw).abs().max())
            assert reconstruction <= 1e-6, reconstruction
            assert float((torch.stack([p.raw_phase for p in global_planes(replacement)]) - source_global).abs().max()) <= 1e-6
        frozen_metrics = evaluate('frozen_source_bank')
        regenerated_metrics = None
        regenerated_phase = None
        if generator:
            with torch.no_grad():
                generator.pixels.copy_(novel_pixels); generator.grid_thw.copy_(novel_grid)
                regenerated_phase = base.phase_summary(bind(), source_raw)
            # Explicit diagnostic: change only reference inputs, keep source offset.
            regenerated_metrics = evaluate('qwen_reference_swap_only')
            with torch.no_grad():
                # Continue the same source mask under new fixed conditioning, without a phase jump.
                generator.initial_reference.copy_(generator.decode_raw() - source_raw)
                assert float((bind() - source_raw).abs().max()) <= 1e-6
        replacement.vision_surrogate.requires_grad_(False)
        replacement.language_surrogate.requires_grad_(False)
        readout.requires_grad_(False)
        if generator:
            for n, p in generator.named_parameters():
                p.requires_grad_(n in generator._trainable_names)
            groups = [{'params': [p for n,p in generator.named_parameters() if p.requires_grad and n.startswith('decoder.')], 'lr': cfg['learning_rates']['generator_decoder']},
                      {'params': [p for n,p in generator.named_parameters() if p.requires_grad and not n.startswith('decoder.')], 'lr': cfg['learning_rates']['generator_context']}]
        else:
            for p in planes: p.raw_phase.requires_grad_(True)
            groups = [{'params': [p.raw_phase for p in planes], 'lr': cfg['learning_rates']['expert']}]
        for g in groups: g['initial_lr'] = g['lr']
        optimizer = torch.optim.AdamW(groups, weight_decay=0.)
        frozen = [(p, p.detach().clone()) for m in (replacement.vision_surrogate, replacement.language_surrogate, readout)
                  for p in m.parameters() if not p.requires_grad]
        epochs = 1 if args.smoke else cfg['adaptation_epochs']
        steps = 2 if args.smoke else cfg['adaptation_steps_per_epoch']
        dataset = base.GroceryRetrievalDataset(support, settings.image_size, augment=False)
        sampler = base.PKBatchSampler(support, 10, 3, args.support_seed, steps)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=base.collate_grocery)
        history = []
        base.write_json(output / 'initial_metrics.json', {'frozen_source_bank': frozen_metrics,
            'qwen_reference_swap_only': regenerated_metrics, 'reference_swap_phase': regenerated_phase,
            'source_reconstruction_max_error': reconstruction})
        base.write_json(output / 'environment.json', {**base.environment_report(), 'git_sha': base.git('rev-parse','HEAD'),
            'git_status': base.git('status','--short'), 'device': torch.cuda.get_device_name(), 'gpu_uuid': args.gpu_uuid,
            'command': sys.argv, 'source_checkpoint_sha256': base.sha256(source_path)})
        for epoch in range(1, epochs + 1):
            base.seed_everything(args.support_seed + 10000 + epoch)
            sampler.set_epoch(epoch)
            torch.cuda.synchronize(); started = time.perf_counter()
            losses = []; chain = {}
            for step, batch in enumerate(loader):
                inputs = base.move_inputs(base.preprocess_images(loaded.processor, batch['images'], settings.instruction), device)
                base.validate_token_budgets(inputs, settings)
                labels = torch.tensor([s.sku_index for s in batch['samples']], device=device)
                n = (epoch - 1) * steps + step
                scale = .15 + .85 * .5 * (1 + math.cos(math.pi * n / max(epochs * steps - 1, 1)))
                scale *= min(1., (n + 1) / 10)
                for g in groups: g['lr'] = g['initial_lr'] * scale
                optimizer.zero_grad(set_to_none=True)
                raw = bind()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=settings.amp_enabled):
                    embedding, _ = base.student_embeddings(loaded.model, replacement, readout, inputs)
                    ret = base.supervised_contrastive_loss(embedding, labels, settings.temperature)
                    proto, _, _ = base.episodic_prototype_retrieval_loss(embedding, labels, settings.gallery_temperature)
                    task_loss = settings.lambda_ret * ret + settings.lambda_gallery * proto
                    loss = task_loss + settings.lambda_phase_dc * base.phase_dc_loss(replacement)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite adaptation loss')
                if step == 0:
                    targets = [raw] if generator else [p.raw_phase for p in planes]
                    grads = torch.autograd.grad(task_loss, targets, retain_graph=True)
                    chain['task_to_expert'] = float(sum(g.float().square().sum() for g in grads).sqrt())
                    if generator:
                        grads = torch.autograd.grad(task_loss, [p for n,p in generator.named_parameters() if n.endswith('lora_b')], retain_graph=True)
                        chain['task_to_lora_b'] = float(sum(g.float().square().sum() for g in grads).sqrt())
                    if min(chain.values()) <= 0: raise RuntimeError('Broken task gradient')
                loss.backward()
                for g in groups: torch.nn.utils.clip_grad_norm_(g['params'], 1., error_if_nonfinite=True)
                optimizer.step(); losses.append(float(task_loss.detach()))
            torch.cuda.synchronize(); train_end = time.perf_counter()
            with torch.no_grad():
                raw = bind().detach()
                delta = max((float((p - before).abs().max()) for p,before in frozen), default=0.)
                assert delta == 0
                phase = base.phase_summary(raw, source_raw)
            record = {'epoch': epoch, 'steps': steps, 'mean_task_loss': sum(losses)/len(losses),
                      'task_gradient_chain': chain, 'expert_phase': phase, 'frozen_parameter_max_change': delta}
            history.append(record)
            checkpoint = {'git_sha': base.git('rev-parse','HEAD'), 'epoch': epoch, 'config': cfg,
                'vision': base.cpu_state(replacement.vision_surrogate), 'language': base.cpu_state(replacement.language_surrogate),
                'readout': base.cpu_state(readout), 'generator': generator.compact_state() if generator else None,
                'expert_raw': raw.cpu(), 'global_raw': source_global.cpu(), 'optimizer': optimizer.state_dict(), 'history': history}
            torch.save(checkpoint, output / 'last_checkpoint.pt')
            if epoch == epochs: torch.save(checkpoint, output / 'best_checkpoint.pt')
            torch.cuda.synchronize(); finished = time.perf_counter()
            record.update(train_loop_seconds=train_end-started, epoch_wall_seconds=finished-started)
            base.write_json(output / 'history.json', history)
            base.write_json(output / 'status.json', {'status': 'running', 'epoch': epoch})
            print(record, flush=True)
        final_metrics = evaluate('adapted')
        with torch.no_grad():
            raw = bind().detach()
            torch.save({'raw_phase': raw.cpu(), 'phase_rad': base.physical_phase(raw).cpu(),
                        'global_raw_phase': source_global.cpu(), 'global_phase_rad': base.physical_phase(source_global).cpu()}, output / 'expert_bank.pt')
            batch = base.collate_grocery([dataset[i] for i in range(3)])
            inputs = base.move_inputs(base.preprocess_images(loaded.processor, batch['images'], settings.instruction), device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=settings.amp_enabled):
                before = base.student_embeddings(loaded.model, replacement, readout, inputs)[0].clone()
                if injection: injection.materialize()
                after = base.student_embeddings(loaded.model, replacement, readout, inputs)[0].clone()
            export_error = float((before-after).abs().max())
            assert export_error <= 1e-5
        fusion = {name: [float(s.core.block1_optical_fusion), float(s.core.block2_optical_fusion)]
                  for name,s in [('vision',replacement.vision_surrogate),('language',replacement.language_surrogate)]}
        assert all(abs(v-.6)<1e-6 for row in fusion.values() for v in row)
        base.write_json(output / 'final_report.json', {'method': args.method, 'class_set': args.class_set, 'shots': args.shots,
            'support_seed': args.support_seed, 'smoke': args.smoke, 'epochs': epochs, 'steps_per_epoch': steps,
            'frozen_source_bank': frozen_metrics, 'qwen_reference_swap_only': regenerated_metrics,
            'adapted': final_metrics, 'expert_phase': base.phase_summary(raw, source_raw), 'fusion': fusion,
            'export_max_error': export_error,
            'selection': 'fixed final epoch; no novel query-based selection', 'query_used_for_gradient': False,
            'source_checkpoint_sha256': base.sha256(source_path), 'git_sha': base.git('rev-parse','HEAD')})
    finally:
        replacement.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--method', required=True, choices=['direct','qwen_vision_lora'])
    parser.add_argument('--class-set', required=True, choices=['a','b'])
    parser.add_argument('--shots', required=True, type=int, choices=[5,20])
    parser.add_argument('--support-seed', required=True, type=int)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--config', default=str(base.TASK/'configs/unseen_v1/cifar.yaml'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = base._read_config(Path(args.config))
    cfg.update(method=args.method, class_set=args.class_set, shots=args.shots, support_seed=args.support_seed, smoke=args.smoke)
    output = Path(args.run_dir).resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    base.write_json(output/'protocol.json', cfg)
    base.write_json(output/'status.json', {'status':'running'})
    try:
        execute(args, cfg, output)
        base.write_json(output/'status.json', {'status':'complete'})
    except BaseException as exc:
        base.write_json(output/'status.json', {'status':'failed', 'error':repr(exc)})
        raise


if __name__ == '__main__':
    main()
