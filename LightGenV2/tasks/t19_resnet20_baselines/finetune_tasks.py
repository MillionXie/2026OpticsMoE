"""Explicit fixed-budget post-30 fine tuning; parent runs remain immutable.

The approved LSP stage changes only the loss to Gaussian heatmap MSE. Its
augmentation workers restart with seed42, so it is a new fine-tuning stage,
not a claim of bitwise uninterrupted 60-epoch training. Deterministic OpenMoji
and editor data retain the original sampler order by replaying index loaders.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .continue_lgvq50 import replay_index_loader, shutdown_loader, structured_sha256
from .frozen_resnet18 import assert_encoder_unchanged, encoder_state_sha256, file_sha256
from .models import build_model, parameter_counts
from .task_problems import Problem
from .train_easy100 import TASK, git, require_published_source, save_checkpoint, wait_for_gpu, write_json
from .train_tasks import CachedIndexBatches

FINAL_EPOCHS = {'lsp': 60, 'openmoji': 60, 'abo_conditional_edit': 50}
COUNTS = {'lsp': (10428, 1000), 'openmoji': (5000, 1000),
          'abo_conditional_edit': (20736, 2304)}
EDITOR_COUNTS = {'train': 20736, 'val': 2304, 'test': 2304}
LOSS_PROFILES = {'lsp': 'gaussian_heatmap_mse_only', 'openmoji': 'unchanged_editing_objective',
                 'abo_conditional_edit': 'unchanged_rgb_mse_gradient_l1'}
SOURCE_FILES = ('finetune_tasks.py', 'task_problems.py', 'train_tasks.py', 'train_easy100.py',
                'models.py', 'frozen_resnet18.py', 'frozen_visual_models.py',
                'frozen_multimodal_models.py', 'performance_cache.py', 'continue_lgvq50.py',
                'other_data.py', 'editor_dataset.py')


class FineTuneProblem(Problem):
    def _other_loss(self, model, batch, device):
        if self.task != 'lsp':
            return super()._other_loss(model, batch, device)
        # Approved minimal repair: preserve peak-one Gaussian targets and the
        # hardargmax evaluation; remove the incompatible temperature-one
        # softargmax coordinate term. No target, crop or model changes.
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import masked_heatmap_mse
        prediction = model(batch['image'].to(device, non_blocking=True))
        return self.settings.heatmap_loss_weight * masked_heatmap_mse(
            prediction, batch['heatmaps'].to(device), batch['visible'].to(device))


def sources():
    return {name: file_sha256(TASK / name) for name in SOURCE_FILES}


def resolve_task_path(value, category):
    path = (TASK / value).resolve()
    if not path.is_relative_to((TASK / 'runs' / category).resolve()):
        raise ValueError('Path escapes the task run category')
    return path


def load_parent(path, problem, model, optimizer):
    if path.name != 'last_checkpoint.pt':
        raise ValueError('Fine tuning must start from the fixed last checkpoint')
    content = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    original = checkpoint['manifest']
    expected = dict(task=problem.task, backbone='resnet18_imagenet_frozen', seed=42, epochs=30,
                    learning_rate=.001, effective_batch=32, microbatch=problem.batch_size,
                    schedule='constant', warmup_epochs=0)
    if any(original.get(key) != value for key, value in expected.items()):
        raise ValueError('Parent is not the approved original 30-epoch profile')
    if checkpoint['epoch'] != 30 or original['data_identity'] != problem.identity:
        raise ValueError('Parent epoch/data identity differs')
    result = json.loads((path.parent / 'result.json').read_text())
    digest = hashlib.sha256(content).hexdigest()
    if (result['status'] != 'complete' or result['selected_epoch'] != 30
            or result['checkpoint_sha256'] != digest or result.get('selection_uses_test') is not False):
        raise ValueError('Parent lacks a completed, SHA-bound fixed-last result')
    if problem.task != 'lsp' and original.get('text_encoder') != 'byte_gru':
        raise ValueError('Fine tuning cannot restore the superseded MiniLM profile')
    model.load_state_dict(checkpoint['model'], strict=True)
    assert_encoder_unchanged(model.encoder, original['encoder_state_sha256_before'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer_sha = structured_sha256(checkpoint['optimizer'])
    if structured_sha256(optimizer.state_dict()) != optimizer_sha:
        raise RuntimeError('AdamW moments were not restored exactly')
    expected_steps = math.ceil(len(problem.train) / 32) * 30
    if (any(group['lr'] != .001 or group['weight_decay'] != .0001 for group in optimizer.param_groups)
            or len(optimizer.state) != sum(len(group['params']) for group in optimizer.param_groups)
            or any(int(state['step']) != expected_steps for state in optimizer.state.values())):
        raise ValueError('Parent optimizer does not match all 30 completed epochs')
    if [audit['epoch'] for audit in original['frozen_encoder_audits']] != list(range(1, 31)):
        raise ValueError('Parent does not contain all 30 encoder audits')
    return checkpoint, {'source_run': str(path.parent), 'checkpoint_sha256': digest,
                        'completed_epoch': 30, 'optimizer_steps': expected_steps,
                        'optimizer_state_sha256': optimizer_sha, 'model_and_AdamW_restored': True}


def validate_smoke(path, parent_sha, profile_sha, source_sha, loss_profile):
    result = json.loads(path.read_text())
    if (result.get('status') != 'complete' or result.get('mode') != 'smoke'
            or result.get('test_evaluations') != 0 or not result.get('trainable_state_changed')
            or result.get('parent_checkpoint_sha256') != parent_sha
            or result.get('profile_sha256') != profile_sha
            or result.get('source_sha256') != source_sha or result.get('loss_profile') != loss_profile):
        raise ValueError('This fine-tuning profile needs its own successful matching smoke')


def run(args):
    require_published_source()
    plan_path = args.plan.resolve(strict=True)
    if not plan_path.is_relative_to((TASK / 'configs').resolve()):
        raise ValueError('Fine tuning requires a task-local published plan')
    plan = json.loads(plan_path.read_text())
    spec = plan['tasks'][args.task]
    if (spec['final_epoch'] != FINAL_EPOCHS[args.task] or spec['loss_profile'] != LOSS_PROFILES[args.task]
            or plan['seed'] != 42 or plan['learning_rate'] != .001 or plan['effective_batch'] != 32
            or args.smoke_batches < 0):
        raise ValueError('Fine-tuning budget or science profile differs from authorization')
    smoke = bool(args.smoke_batches)
    run_dir = args.run_dir.resolve()
    category = 'smoke' if smoke else 'simulation'
    if not run_dir.is_relative_to((TASK / 'runs' / category).resolve()):
        raise ValueError('Run is outside its approved task category')
    parent_path = resolve_task_path(spec['parent_checkpoint'], 'simulation')
    profile_sha, source_sha = file_sha256(plan_path), sources()
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'preflight', 'task': args.task, 'pid': os.getpid(),
                'command': [sys.executable, '-m', __spec__.name, *sys.argv[1:]],
                'mode': 'smoke' if smoke else 'formal', 'started_unix': time.time(),
                'profile_sha256': profile_sha, 'source_sha256': source_sha,
                'git_commit': git('rev-parse', 'HEAD'), 'gpu_uuid': args.gpu_uuid,
                'python': platform.python_version(), 'torch': torch.__version__,
                'selection': 'fixed_last_epoch', 'selection_uses_test': False,
                'loss_profile': spec['loss_profile']}
    write_json(run_dir / 'run_manifest.json', manifest)
    loaders = []
    model = optimizer = None
    began = time.monotonic()
    try:
        torch.set_num_threads(4)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        problem = FineTuneProblem(args.task)
        if (len(problem.train), len(problem.test)) != COUNTS[args.task]:
            raise ValueError('Fine-tuning dataset counts differ')
        if problem.batch_sampler is not None:
            raise ValueError('Unsupported fine-tuning batch sampler')
        if args.task == 'abo_conditional_edit':
            from .performance_cache import CachedEditorPairs
            cache_root = resolve_task_path(spec['editor_cache_root'], 'cache')
            cached = {split: CachedEditorPairs(cache_root / split, source_identity=problem.identity,
                                              verify_integrity=False) for split in ('train', 'val', 'test')}
            for split, dataset in cached.items():
                if (len(dataset) != EDITOR_COUNTS[split]
                        or dataset.manifest.get('dataset_split') != split
                        or not dataset.manifest.get('all_rows_exactly_verified')):
                    raise RuntimeError('Editor cache count/split/exact verification differs')
                dataset.preload()  # Full blobs are SHA checked, then restored with original FP32 math.
            problem.train, problem.test, problem.final_test = (cached[s] for s in ('train', 'val', 'test'))
            manifest['cache_manifests'] = {s: file_sha256(cache_root / s / 'manifest.json') for s in cached}
        model = build_model(args.task, backbone='resnet18_imagenet_frozen',
                            **({} if args.task == 'lsp' else {'text_encoder': 'byte_gru'}))
        trainable = [p for p in model.parameters() if p.requires_grad]
        if parameter_counts(model)['total'] >= 15000000:
            raise ValueError('Complete model must remain below 15M parameters')
        optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.0001)
        parent, continuation = load_parent(parent_path, problem, model, optimizer)
        frozen_sha = parent['manifest']['encoder_state_sha256_before']
        if not smoke:
            if args.smoke_result is None:
                raise ValueError('Formal fine tuning requires its own smoke result')
            smoke_path = args.smoke_result.resolve(strict=True)
            if not smoke_path.is_relative_to((TASK / 'runs' / 'smoke').resolve()):
                raise ValueError('Smoke result must be task-local')
            validate_smoke(smoke_path, continuation['checkpoint_sha256'], profile_sha, source_sha,
                           spec['loss_profile'])
        manifest.update(status='waiting_for_gpu', parent=continuation, first_epoch=31,
                        final_epoch=31 if smoke else spec['final_epoch'], seed=42,
                        backbone='resnet18_imagenet_frozen', epochs=31 if smoke else spec['final_epoch'],
                        text_encoder='none' if args.task == 'lsp' else 'byte_gru',
                        learning_rate=.001, schedule='constant', warmup_epochs=0,
                        effective_batch=32, microbatch=problem.batch_size,
                        optimizer='AdamW_wd0.0001_clip1', precision='FP32_TF32_disabled',
                        parameter_counts=parameter_counts(model), optimizer_parameters=sum(p.numel() for p in trainable),
                        data_identity=problem.identity, pretrained_provenance=model.encoder.pretrained_provenance,
                        encoder_state_sha256_before=frozen_sha,
                        frozen_encoder_audits=list(parent['manifest']['frozen_encoder_audits']),
                        training_scope='decoder_only' if args.task == 'lsp' else 'frozen_visual_trainable_text_and_decoder',
                        parent_loss_profile='original_gaussian_mse_plus_coordinate' if args.task == 'lsp' else 'unchanged',
                        sampler_policy='fresh_seed42_augmented_finetune_stage_not_uninterrupted_rng' if args.task == 'lsp'
                        else 'original_persistent2_index_sampler_replayed_30_epochs')
        write_json(run_dir / 'run_manifest.json', manifest)
        device = torch.device(args.device)
        if args.device == 'cuda':
            wait_for_gpu(args)
        model.to(device)
        # Move restored AdamW moments with the parameters without changing their values.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step':
                    state[key] = value.to(device)
        if structured_sha256(optimizer.state_dict()) != continuation['optimizer_state_sha256']:
            raise RuntimeError('AdamW state changed during device transfer')
        before_trainable = structured_sha256({n: p for n, p in model.named_parameters() if p.requires_grad})
        options = dict(batch_size=problem.batch_size, num_workers=2, pin_memory=args.device == 'cuda',
                       multiprocessing_context='spawn', persistent_workers=True)
        if args.task == 'lsp':
            train = DataLoader(problem.train, shuffle=True, generator=torch.Generator().manual_seed(42), **options)
            loaders.append(train)
        else:
            indices, replay = replay_index_loader(len(problem.train), problem.batch_size, 30,
                                                  pin_memory=args.device == 'cuda')
            loaders.append(indices)
            manifest['sampler_replay'] = replay
            train = CachedIndexBatches(problem.train, indices)
        parent_history = json.loads((parent_path.parent / 'history.json').read_text())
        if [e['epoch'] for e in parent_history] != list(range(1, 31)):
            raise ValueError('Parent history is incomplete')
        history = list(parent_history)
        manifest['status'] = 'running'
        write_json(run_dir / 'run_manifest.json', manifest)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        final_epoch = manifest['final_epoch']
        for epoch in range(31, final_epoch + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = count = 0
            order = []
            accumulation = max(1, 32 // problem.batch_size)
            steps = min(len(train), args.smoke_batches) if smoke else len(train)
            sample_limit = min(len(problem.train), steps * problem.batch_size) if smoke else len(problem.train)
            for step, batch in enumerate(itertools.islice(iter(train), steps)):
                n = len(batch['sample_id'])
                group_start = (step // accumulation) * accumulation * problem.batch_size
                group_samples = min(32, sample_limit - group_start)
                loss = problem.loss(model, batch, device)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite fine-tuning loss')
                (loss * n / group_samples).backward()
                if (step + 1) % accumulation == 0 or step + 1 == steps:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                total += float(loss.detach()) * n; count += n
                order.extend(batch['sample_id'])
                if (step + 1) % 25 == 0:
                    write_json(run_dir / 'progress.json', dict(status='training', epoch=epoch, step=step + 1,
                               train_loss=total / count, session_elapsed_seconds=time.monotonic() - began))
            if count != sample_limit or len(set(order)) != count:
                raise RuntimeError('Fine-tuning epoch omitted or repeated TRAIN identities')
            audit = assert_encoder_unchanged(model.encoder, frozen_sha)
            manifest['frozen_encoder_audits'].append({'epoch': epoch, **audit})
            entry = dict(epoch=epoch, train_loss=total / count, learning_rate=.001,
                         loss_profile=spec['loss_profile'], samples=count,
                         sample_order_sha256=structured_sha256(order),
                         session_elapsed_seconds=time.monotonic() - began)
            history.append(entry)
            manifest['last_completed_epoch'] = epoch
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                           manifest=manifest, metrics=entry, torch_rng_state=torch.get_rng_state())
            save_checkpoint(run_dir / 'last_checkpoint.pt', payload)
            write_json(run_dir / 'history.json', history)
            write_json(run_dir / 'run_manifest.json', manifest)
            write_json(run_dir / 'progress.json', dict(status='training', **entry))
            print(entry, flush=True)
        last_sha = file_sha256(run_dir / 'last_checkpoint.pt')
        updates_per_epoch = math.ceil(sample_limit / 32)
        expected_steps = continuation['optimizer_steps'] + updates_per_epoch * (final_epoch - 30)
        if any(int(state['step']) != expected_steps for state in optimizer.state.values()):
            raise RuntimeError('Fine-tuning AdamW steps differ from the fixed budget')
        manifest['optimizer_steps_after'] = expected_steps
        selected = torch.load(run_dir / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        if selected['epoch'] != final_epoch:
            raise RuntimeError('Last checkpoint epoch differs from the fixed budget')
        model.load_state_dict(selected['model'], strict=True)
        changed = before_trainable != structured_sha256({n: p for n, p in model.named_parameters() if p.requires_grad})
        if not changed:
            raise RuntimeError('No trainable parameter changed')
        metrics, records, validation_metrics = {}, None, None
        if not smoke:
            if args.task == 'abo_conditional_edit':
                test = CachedIndexBatches(problem.test, DataLoader(range(len(problem.test)), batch_size=8))
            else:
                test = DataLoader(problem.test, shuffle=False, **options)
                loaders.append(test)
            metrics, records = problem.evaluate(model, test, device)
            if problem.final_test is not None:
                validation_metrics = metrics
                final_loader = CachedIndexBatches(problem.final_test,
                    DataLoader(range(len(problem.final_test)), batch_size=8))
                metrics, records = problem.evaluate(model, final_loader, device)
            write_json(run_dir / 'fixed_weight_predictions.json', records)
        if file_sha256(parent_path) != continuation['checkpoint_sha256']:
            raise RuntimeError('Immutable parent checkpoint changed')
        if file_sha256(run_dir / 'last_checkpoint.pt') != last_sha:
            raise RuntimeError('Checkpoint changed during fixed-weight evaluation')
        manifest['frozen_encoder_final_audit'] = assert_encoder_unchanged(model.encoder, frozen_sha)
        write_json(run_dir / 'result.json', dict(status='complete', mode=manifest['mode'], selected_epoch=final_epoch,
                   metrics=metrics, validation_metrics=validation_metrics, selection='fixed_last_epoch',
                   selection_uses_test=False, checkpoint='last_checkpoint.pt', checkpoint_sha256=last_sha,
                   parent_checkpoint_sha256=continuation['checkpoint_sha256'],
                   trainable_state_changed=changed, profile_sha256=profile_sha, source_sha256=source_sha,
                   loss_profile=spec['loss_profile'], test_evaluations=0 if smoke else 1,
                   session_elapsed_seconds=time.monotonic() - began))
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        manifest['finished_unix'] = time.time()
        write_json(run_dir / 'run_manifest.json', manifest)
        for loader in loaders:
            shutdown_loader(loader)
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if args.device == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(FINAL_EPOCHS), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke-batches', type=int, default=0)
    parser.add_argument('--smoke-result', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--max-physical-gpus', type=int, choices=(4,), default=4)
    parser.add_argument('--max-jobs-per-gpu', type=int, choices=(2, 3), default=2)
    parser.add_argument('--queue-status-path', type=Path)
    parser.add_argument('--shared-owner-pids', type=int, nargs='*', default=[])
    run(parser.parse_args())


if __name__ == '__main__':
    main()
