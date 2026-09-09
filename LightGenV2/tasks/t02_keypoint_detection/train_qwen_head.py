"""Train only a smaller Qwen pose head, preserving native frozen Vision."""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import build_teacher, load_vision_backbone
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import load_settings, save_resolved_config
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.training import _train_epoch, build_loaders, evaluate_model
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.protocol import build_periodic_test_protocol, persist_protocol
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.training import _selection_key
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import resolve_cached_model_source
from .modeling import sha256_file
from .run import _seed

TASK = Path(__file__).resolve().parent


def audit_frozen_teacher(model):
    head_ids = {id(p) for p in model.head.parameters()}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if trainable_ids != head_ids or any(p.requires_grad for p in model.visual.parameters()):
        raise RuntimeError('Only the readout head may be trainable')
    return {'head': model.head.specification(),
            'native_vision_blocks': len(model.visual.blocks),
            'native_vision_blocks_executed': True, 'language_executed': False,
            'frozen_vision_parameters': sum(p.numel() for p in model.visual.parameters())}


def run(args):
    settings = load_settings(args.config)
    settings.data_root = args.data_root.resolve()
    settings.cache_dir = args.cache_dir.resolve()
    settings.local_files_only = True
    settings.download = False
    settings.output_dir = args.run_dir.resolve()
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    settings.num_workers = args.workers
    settings.visualization_sample_count = 0
    settings.log_interval_batches = 200
    if args.smoke:
        settings.teacher_batch_size = 2
        settings.teacher_epochs = 1
    out = settings.output_dir

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n', encoding='utf-8')

    write('status.json', {'status': 'initializing', 'smoke_only': args.smoke})
    model = None
    try:
        _seed(settings.random_seed)
        bundle = build_periodic_test_protocol(prepare_lsp(settings, persist=False))
        persist_protocol(bundle, out)
        if args.smoke:
            bundle = dataclasses.replace(bundle, train=bundle.train[:2], test=bundle.test[:2])
        device = torch.device(args.device)
        loaded = load_vision_backbone(settings, device)
        model = build_teacher(loaded, settings)
        audit = audit_frozen_teacher(model)
        if audit['head']['parameters'] != 138422:
            raise RuntimeError('Expected the audited 138422-parameter Deconv40 head')
        initial_head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        source = resolve_cached_model_source(settings.model_id, settings.cache_dir)
        manifest = {
            'command': sys.argv, 'args': vars(args), 'settings': settings.to_dict(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TASK, text=True).strip(),
            'model_snapshot': source,
            'model_config_sha256': sha256_file(Path(source)/'config.json'),
            'data_manifest_sha256': sha256_file(out/'pose_protocol_split.csv'),
            'architecture': audit, 'train_samples': len(bundle.train), 'test_samples': len(bundle.test),
            'head_initialization': 'random; not copied or truncated from old head',
            'selection': 'max periodic test PCK; ties NME/loss/earliest epoch; no EMA',
            'test_used_for_selection': True, 'historical_baseline_selection': 'minimum train loss',
            'environment': {'torch': torch.__version__, 'transformers': transformers.__version__,
                            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device)},
            'smoke_only': args.smoke,
        }
        write('run_manifest.json', manifest)
        save_resolved_config(settings)
        print('ARCHITECTURE', json.dumps(audit), flush=True)
        train_loader, test_loader = build_loaders(bundle, settings, training=True)
        opt = torch.optim.AdamW(model.head.parameters(), lr=settings.teacher_learning_rate,
                                weight_decay=settings.weight_decay)
        history, best_key, best_epoch = [], None, None
        for epoch in range(1, settings.teacher_epochs+1):
            started = time.perf_counter()
            train = _train_epoch(model, 'teacher', train_loader, loaded.processor, device, opt, settings, epoch)
            if any(p.grad is not None for p in model.visual.parameters()):
                raise RuntimeError('Frozen Vision unexpectedly acquired gradients')
            test, _ = evaluate_model(model, 'teacher', test_loader, loaded.processor, device, settings,
                                     phase='qwen_deconv40_periodic_test', epoch=epoch, save_outputs=False, tta=False)
            payload = {'epoch': epoch, 'head': model.head.state_dict(), 'architecture': audit,
                       'train_metrics': train, 'test_metrics': test, 'manifest': manifest}
            torch.save(payload, out/'last_checkpoint.pt')
            key = _selection_key(test, epoch)
            if best_key is None or key < best_key:
                best_key, best_epoch = key, epoch
                torch.save(payload, out/'best_checkpoint.pt')
            row = {'epoch': epoch, 'train': train, 'test': test, 'seconds': time.perf_counter()-started,
                   'best_epoch': best_epoch, 'lr': opt.param_groups[0]['lr']}
            history.append(row)
            write('training_history.json', history)
            write('status.json', {'status': 'training', 'epoch': epoch, 'best_epoch': best_epoch})
            print('EPOCH', epoch, 'PCK', test['pck_at_0.2_torso'], 'BEST', best_epoch, flush=True)
        selected = torch.load(out/'best_checkpoint.pt', map_location=device, weights_only=False)
        model.head.load_state_dict(selected['head'], strict=True)
        final, _ = evaluate_model(model, 'teacher', test_loader, loaded.processor, device, settings,
                                 phase='qwen_deconv40_selected_test', epoch=best_epoch, save_outputs=True, tta=False)
        changes = [(value.detach().float().cpu()-initial_head[name].float()).square().sum()
                   for name, value in model.head.state_dict().items() if value.is_floating_point()]
        write('final_report.json', {'test': final, 'best_epoch': best_epoch,
                                  'checkpoint_sha256': sha256_file(out/'best_checkpoint.pt'),
                                  'head_parameter_count': audit['head']['parameters'],
                                  'head_squared_change_from_initialization': float(sum(changes)),
                                  'backbone_trainable_parameters': 0, 'smoke_only': args.smoke})
        write('status.json', {'status': 'complete', 'best_epoch': best_epoch, 'smoke_only': args.smoke})
    except Exception as error:
        write('status.json', {'status': 'failed', 'error': repr(error)})
        raise
    finally:
        if model is not None:
            model.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=TASK/'configs/qwen_deconv40.yaml')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
