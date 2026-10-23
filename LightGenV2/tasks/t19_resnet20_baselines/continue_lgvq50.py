"""Continue sealed frozen-ResNet18 LGVQ epoch30 weights/AdamW through epoch50.

The original run is read-only. Static FP32 frame features are cached only in
memory, with all-sample direct-forward equivalence checks. A real persistent
DataLoader replays the old 30 sampler epochs before yielding epoch31 indices.
"""
from __future__ import annotations

import argparse
import hashlib
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .frozen_resnet18 import (OFFICIAL_CHECKPOINT_SHA256, assert_encoder_unchanged,
                              encoder_state_sha256, file_sha256)
from .models import build_model, parameter_counts
from .task_problems import Problem
from .train_easy100 import (TASK, git, require_published_source, save_checkpoint,
                            wait_for_gpu, write_json)

EXPECTED_COUNTS = {'train': 2250, 'test': 558}
MICROBATCH = 8
EFFECTIVE_BATCH = 32
PARENT_EPOCH = 30
FINAL_EPOCH = 50
EQUIVALENCE_POLICY = {
    'same_batch': {'rtol': 1e-5, 'atol': 1e-5},
    'cross_batch_singleton': {'rtol': 1e-4, 'atol': 1e-4,
        'reason': 'FP32 encoder/head kernels can round differently when batch shape changes; not a bitwise guarantee'},
}


def structured_sha256(value):
    """Stable identity of tensors and optimizer/cache metadata, without pickle."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for element in item:
                visit(element)
        else:
            digest.update(json.dumps([type(item).__name__, item], ensure_ascii=False,
                                     allow_nan=False).encode('utf-8'))
        digest.update(b'\0')

    visit(value)
    return digest.hexdigest()


class IndexDataset(Dataset):
    """Static indices replace expensive RGB loading without changing sampling."""
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return index


def shutdown_loader(loader, *, cleanup_errors=None):
    # Capture the in-flight error before entering an except for cleanup itself.
    # The first scientific/runtime failure must not be replaced by a prefetch
    # worker abort during early shutdown. A standalone cleanup failure still fails.
    original_error = sys.exc_info()[1]
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        try:
            iterator._shutdown_workers()
        except BaseException as cleanup_error:
            record = {'operation': 'DataLoader._shutdown_workers',
                      'error': repr(cleanup_error),
                      'original_error': repr(original_error) if original_error is not None else None}
            owner = original_error if original_error is not None else cleanup_error
            attached = list(getattr(owner, 'cleanup_errors', []))
            attached.append(record)
            owner.cleanup_errors = attached
            if cleanup_errors is not None:
                cleanup_errors.append(record)
            if original_error is None:
                raise
            if hasattr(original_error, 'add_note'):
                original_error.add_note(f'DataLoader cleanup also failed: {cleanup_error!r}')


def replay_index_loader(length, batch_size, completed_epochs, seed=42, num_workers=2, pin_memory=True):
    """Use PyTorch itself to reproduce every original generator consumption.

    In particular, persistent-worker base_seed is drawn only on first iterator
    creation, and RandomSampler's end-of-epoch generator draws are preserved.
    Keep this loader alive for subsequent epochs: recreating it would draw an
    additional base_seed and change the epoch31 sample order.
    """
    if length < 1 or batch_size < 1 or completed_epochs < 0 or num_workers < 0:
        raise ValueError('Invalid index-loader replay budget')
    generator = torch.Generator().manual_seed(seed)
    options = dict(batch_size=batch_size, shuffle=True, generator=generator,
                   num_workers=num_workers, pin_memory=pin_memory)
    if num_workers:
        options.update(multiprocessing_context='spawn', persistent_workers=True)
    loader = DataLoader(IndexDataset(length), **options)
    orders = []
    try:
        for epoch in range(1, completed_epochs + 1):
            order = torch.cat(list(loader))
            if len(order) != length or not torch.equal(order.sort().values, torch.arange(length)):
                raise RuntimeError('Sampler replay omitted or repeated an LGVQ TRAIN sample')
            orders.append({'epoch': epoch, 'sample_order_sha256': structured_sha256(order)})
    except BaseException:
        shutdown_loader(loader)
        raise
    return loader, {
        'seed': seed, 'completed_epochs_replayed': completed_epochs,
        'batch_size': batch_size, 'num_workers': num_workers, 'persistent_workers': bool(num_workers),
        'mechanism': 'actual_original_DataLoader_RandomSampler_exhausted_once_per_epoch',
        'loader_retained_for_continuation_epoch': completed_epochs + 1, 'replayed_epoch_orders': orders,
        'generator_state_sha256_after_replay': structured_sha256(generator.get_state()),
    }


def replay_training_loader(count, completed_epochs=PARENT_EPOCH):
    return replay_index_loader(count, MICROBATCH, completed_epochs, seed=42, num_workers=2)


def cached_forward(model, features):
    if features.ndim != 3 or features.shape[1:] != (4, model.feature_dim):
        raise ValueError('Expected four ordered frame features per LGVQ video')
    inputs = (features.mean(1) if model.target == 'spatial' else
              torch.cat((features.flatten(1),
                         (features[:, 1:] - features[:, :-1]).flatten(1)), 1))
    logits = model.head(inputs)
    return {'quality_logits': logits,
            'quality_score': logits.softmax(-1) @ model.quality_levels}


def cache_dataset(model, dataset, device, split, *, workers=2):
    """Cache FP32 features and compare every cached batch to direct forward."""
    options = dict(batch_size=MICROBATCH, shuffle=False, num_workers=workers,
                   pin_memory=True, generator=torch.Generator().manual_seed(42))
    if workers:
        options.update(multiprocessing_context='spawn', persistent_workers=True)
    loader = DataLoader(dataset, **options)
    features, scores, ids = [], [], []
    max_logits = max_score = singleton_logits = singleton_score = 0.
    bindings = []
    model.eval()
    try:
        with torch.no_grad():
            for batch_number, batch in enumerate(loader):
                frames = batch['image'].to(device, non_blocking=True)
                n, four, channels, height, width = frames.shape
                if (four, channels) != (4, 3):
                    raise ValueError('LGVQ frame geometry changed')
                z = model.encoder(frames.reshape(n * four, channels, height, width))
                z = z.reshape(n, four, model.feature_dim)
                if z.dtype != torch.float32 or not torch.isfinite(z).all():
                    raise RuntimeError('Frozen visual cache must contain finite FP32 features')
                direct = model(frames)
                cached = cached_forward(model, z)
                for key in ('quality_logits', 'quality_score'):
                    torch.testing.assert_close(cached[key], direct[key], **EQUIVALENCE_POLICY['same_batch'])
                max_logits = max(max_logits, float((direct['quality_logits'] - cached['quality_logits']).abs().max()))
                max_score = max(max_score, float((direct['quality_score'] - cached['quality_score']).abs().max()))
                if batch_number == 0:
                    single = model(frames[:1])
                    one_cached = cached_forward(model, z[:1])
                    for key in ('quality_logits', 'quality_score'):
                        torch.testing.assert_close(one_cached[key], single[key],
                            **{name:EQUIVALENCE_POLICY['cross_batch_singleton'][name] for name in ('rtol','atol')})
                    singleton_logits = float((single['quality_logits'] - one_cached['quality_logits']).abs().max())
                    singleton_score = float((single['quality_score'] - one_cached['quality_score']).abs().max())
                batch_ids = list(batch['sample_id'])
                bindings.append(structured_sha256({'ids': batch_ids,
                    'direct_logits': direct['quality_logits'], 'cached_logits': cached['quality_logits'],
                    'direct_score': direct['quality_score'], 'cached_score': cached['quality_score']}))
                features.append(z.cpu())
                scores.append(batch['score'].float().cpu())
                ids.extend(batch_ids)
    finally:
        shutdown_loader(loader)
    if len(ids) != len(dataset) or len(set(ids)) != len(ids):
        raise RuntimeError('Cached LGVQ sample IDs are missing or duplicated')
    result = {'features': torch.cat(features), 'scores': torch.cat(scores), 'sample_ids': ids}
    return result, {
        'split': split, 'samples': len(ids), 'features_shape': list(result['features'].shape),
        'cache_dtype': 'torch.float32', 'cache_bytes': result['features'].numel() * 4,
        'cache_binding_sha256': structured_sha256(result),
        'all_samples_checked_against_direct_forward': True,
        'batch_size': MICROBATCH, 'max_abs_logits_error': max_logits,
        'max_abs_score_error': max_score, 'singleton_samples_checked': 1,
        'singleton_max_abs_logits_error': singleton_logits,
        'singleton_max_abs_score_error': singleton_score,
        'equivalence_policy': EQUIVALENCE_POLICY,
        'output_binding_sha256': structured_sha256(bindings),
    }


@torch.no_grad()
def evaluate_cached(model, cache, device):
    from LightGenV2.tasks.t06_video_quality_assessment.quality_token_train import score_metrics
    model.eval()
    predictions = []
    for start in range(0, len(cache['sample_ids']), MICROBATCH):
        predictions.append(cached_forward(model, cache['features'][start:start + MICROBATCH].to(device))['quality_score'].cpu())
    predicted = torch.cat(predictions)
    metrics = score_metrics(cache['scores'], predicted)
    metrics = {name: value if math.isfinite(value) else None for name, value in metrics.items()}
    records = [{'sample_id': sid, 'target': float(target), 'prediction': float(value)}
               for sid, target, value in zip(cache['sample_ids'], cache['scores'], predicted)]
    return metrics, records


def load_parent(source_run, target):
    manifest = json.loads((source_run / 'run_manifest.json').read_text())
    result = json.loads((source_run / 'result.json').read_text())
    checkpoint = source_run / 'last_checkpoint.pt'
    identity = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = payload['manifest']
    for data in (manifest, saved):
        if (data.get('task') != 'lgvq_' + target or data.get('backbone') != 'resnet18_imagenet_frozen'
                or data.get('seed') != 42 or data.get('epochs') != PARENT_EPOCH
                or data.get('training_scope') != 'decoder_only'
                or data.get('learning_rate') != .001 or data.get('schedule') != 'constant'
                or data.get('warmup_epochs') != 0 or data.get('effective_batch') != EFFECTIVE_BATCH
                or data.get('microbatch') != MICROBATCH or str(data.get('torch')) != str(torch.__version__)):
            raise ValueError('Parent is not the approved same-version frozen LGVQ epoch30 profile')
    if (manifest.get('status') != 'complete' or result.get('status') != 'complete'
            or result.get('selected_epoch') != PARENT_EPOCH or result.get('selection') != 'fixed_last_epoch'
            or payload.get('epoch') != PARENT_EPOCH or result.get('checkpoint_sha256') != identity):
        raise ValueError('Parent last checkpoint is not a sealed, verified epoch30 result')
    if 'optimizer' not in payload:
        raise ValueError('Parent checkpoint has no AdamW state to continue')
    if file_sha256(checkpoint) != identity:
        raise RuntimeError('Parent checkpoint changed while being read')
    return payload, manifest, identity


def verify_official_cache():
    """Never silently download weights as part of continuation construction."""
    checkpoint = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    if not checkpoint.is_file() or file_sha256(checkpoint) != OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError('Verified local official ResNet18 cache is required; continuation does not download weights')


def restore_optimizer(model, state):
    trainable = [value for value in model.parameters() if value.requires_grad]
    if {id(value) for value in trainable} != {id(value) for value in model.head.parameters()}:
        raise RuntimeError('Only the original LGVQ head may enter AdamW')
    optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.0001)
    expected = structured_sha256(state)
    optimizer.load_state_dict(state)
    if structured_sha256(optimizer.state_dict()) != expected:
        raise RuntimeError('AdamW moments, step counters or parameter-group state were not restored exactly')
    if any(group['lr'] != .001 or group['weight_decay'] != .0001 for group in optimizer.param_groups):
        raise ValueError('Parent AdamW schedule does not match constant lr0.001/wd0.0001')
    if len(optimizer.state) != len(trainable):
        raise ValueError('Parent AdamW state is missing trained head parameters')
    expected_steps = math.ceil(EXPECTED_COUNTS['train'] / EFFECTIVE_BATCH) * PARENT_EPOCH
    if any(int(value['step']) != expected_steps for value in optimizer.state.values()):
        raise ValueError('Parent AdamW update counters do not match 30 complete LGVQ epochs')
    return optimizer, {'optimizer_state_restored': True, 'parent_optimizer_state_sha256': expected,
                       'restored_optimizer_state_sha256': expected, 'parent_optimizer_steps': expected_steps}


def run(args):
    require_published_source()
    if args.smoke_batches not in (0, 2):
        raise ValueError('Smoke uses exactly two batches; formal continues epochs31 through50')
    source = args.source_run.resolve(strict=True)
    run_dir = args.run_dir.resolve()
    category = 'smoke' if args.smoke_batches else 'simulation'
    if (not source.is_relative_to((TASK / 'runs/simulation').resolve())
            or not run_dir.is_relative_to((TASK / 'runs' / category).resolve()) or source == run_dir):
        raise ValueError('Continuation requires a distinct task-owned run and a sealed source simulation')
    payload, parent, parent_sha = load_parent(source, args.target)
    problem = Problem('lgvq_' + args.target)
    if any(len(getattr(problem, split)) != count for split, count in EXPECTED_COUNTS.items()):
        raise ValueError('Original LGVQ TRAIN2250/TEST558 counts changed')
    if problem.identity != parent['data_identity']:
        raise ValueError('LGVQ data identity changed since the sealed epoch30 run')
    verify_official_cache()
    wait_for_gpu(args)
    run_dir.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    manifest = {
        'task': 'lgvq_' + args.target, 'backbone': 'resnet18_imagenet_frozen', 'status': 'running',
        'training_scope': 'decoder_only', 'seed': 42, 'start_epoch': 31, 'end_epoch': FINAL_EPOCH,
        'additional_epochs': 20, 'epochs': FINAL_EPOCH, 'learning_rate': .001, 'schedule': 'constant',
        'warmup_epochs': 0, 'microbatch': MICROBATCH, 'effective_batch': EFFECTIVE_BATCH,
        'precision': 'FP32_TF32_disabled', 'optimizer': 'continued_AdamW_wd0.0001_clip1',
        'selection': 'fixed_last_epoch', 'parent_run': str(source), 'parent_checkpoint': 'last_checkpoint.pt',
        'continuation_numerics': 'same_FP32_graph; checked_direct_forward_tolerance; not_a_bitwise_cross_batch_kernel_guarantee',
        'cache_equivalence_policy': EQUIVALENCE_POLICY, 'cleanup_errors': [],
        'parent_checkpoint_sha256_before': parent_sha, 'parent_git_commit': parent.get('git_commit'),
        'parent_source_sha256': parent.get('source_sha256'), 'data_identity': problem.identity,
        'gpu_uuid': args.gpu_uuid, 'pid': os.getpid(), 'gpu': torch.cuda.get_device_name(),
        'max_physical_gpus': args.max_physical_gpus, 'max_jobs_per_gpu': args.max_jobs_per_gpu,
        'git_commit': git('rev-parse', 'HEAD'), 'torch': str(torch.__version__), 'python': platform.python_version(),
        'command': [sys.executable, '-m', __spec__.name, *sys.argv[1:]],
        'source_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
            ('continue_lgvq50.py', 'models.py', 'frozen_resnet18.py', 'task_problems.py', 'lgvq_data.py')},
        'smoke_batches': args.smoke_batches, 'formal_result': not bool(args.smoke_batches),
        'partial_epoch': bool(args.smoke_batches), 'started_unix': time.time(),
    }
    write_json(run_dir / 'run_manifest.json', manifest)
    loader = None
    began = time.monotonic()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    try:
        model = build_model('lgvq_' + args.target, backbone='resnet18_imagenet_frozen').to(device)
        model.load_state_dict(payload['model'], strict=True)
        frozen_sha = encoder_state_sha256(model.encoder)
        expected_sha = parent['encoder_state_sha256_before']
        assert_encoder_unchanged(model.encoder, expected_sha)
        levels = model.quality_levels.detach().clone()
        problem.configure_model(model, device)
        if not torch.equal(model.quality_levels, levels):
            raise ValueError('TRAIN MOS scale changed since epoch30')
        optimizer, optimizer_audit = restore_optimizer(model, payload['optimizer'])
        manifest.update(optimizer_audit, parameter_counts=parameter_counts(model),
            encoder_state_sha256_before=frozen_sha, head_state_sha256_before=encoder_state_sha256(model.head),
            pretrained_provenance=model.encoder.pretrained_provenance, frozen_encoder_audits=[])
        train_cache, train_audit = cache_dataset(model, problem.train, device, 'train')
        test_cache, test_audit = cache_dataset(model, problem.test, device, 'test')
        if set(train_cache['sample_ids']) & set(test_cache['sample_ids']):
            raise ValueError('TRAIN and TEST cache sample identities overlap')
        manifest['feature_cache'] = {'storage': 'FP32_RAM_only', 'train': train_audit, 'test': test_audit,
                                     'encoder_state_sha256': frozen_sha}
        assert_encoder_unchanged(model.encoder, frozen_sha)
        loader, replay_audit = replay_training_loader(len(problem.train))
        manifest['sampler_continuation'] = replay_audit
        write_json(run_dir / 'run_manifest.json', manifest)
        history = []
        last_epoch = 31 if args.smoke_batches else FINAL_EPOCH
        train_features = train_cache['features'].to(device)
        train_scores = train_cache['scores'].to(device)
        trainable = [value for value in model.parameters() if value.requires_grad]
        for epoch in range(31, last_epoch + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            count = 0
            sample_order = []
            for step, indices in enumerate(loader):
                indices = indices.to(device)
                n = len(indices)
                group_start = (step // 4) * EFFECTIVE_BATCH
                group_samples = min(EFFECTIVE_BATCH, len(problem.train) - group_start)
                if args.smoke_batches:
                    group_samples = min(EFFECTIVE_BATCH, args.smoke_batches * MICROBATCH)
                logits = cached_forward(model, train_features[indices])['quality_logits']
                labels = torch.bucketize(train_scores[indices], problem.boundaries, right=False)
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite continuation TRAIN loss')
                (loss * n / group_samples).backward()
                stop_smoke = bool(args.smoke_batches and step + 1 == args.smoke_batches)
                if (step + 1) % 4 == 0 or step + 1 == len(loader) or stop_smoke:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                total += float(loss.detach()) * n
                count += n
                sample_order.extend(indices.cpu().tolist())
                if (step + 1) % 25 == 0:
                    write_json(run_dir / 'progress.json', {'status': 'training', 'epoch': epoch,
                        'step': step + 1, 'train_loss': total / count, 'elapsed_seconds': time.monotonic() - began})
                if stop_smoke:
                    break
            metrics = {}
            if args.smoke_batches or epoch % 5 == 0:
                metrics, _ = evaluate_cached(model, test_cache, device)
            entry = {'epoch': epoch, 'train_loss': total / count, 'learning_rate': .001,
                'samples_trained': count, 'sample_order_sha256': structured_sha256(torch.tensor(sample_order)),
                'partial_epoch': bool(args.smoke_batches), 'elapsed_seconds': time.monotonic() - began, **metrics}
            history.append(entry)
            manifest['frozen_encoder_audits'].append({'epoch': epoch, **assert_encoder_unchanged(model.encoder, frozen_sha)})
            write_json(run_dir / 'history.json', history)
            write_json(run_dir / 'progress.json', {'status': 'training', **entry})
            write_json(run_dir / 'run_manifest.json', manifest)
            print(entry, flush=True)
        manifest.update(head_state_sha256_after=encoder_state_sha256(model.head),
                        encoder_state_sha256_after=encoder_state_sha256(model.encoder),
                        optimizer_state_sha256_after=structured_sha256(optimizer.state_dict()))
        if file_sha256(source / 'last_checkpoint.pt') != parent_sha:
            raise RuntimeError('Protected source checkpoint changed during continuation')
        final_payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': last_epoch,
                         'manifest': manifest, 'metrics': history[-1], 'partial_epoch': bool(args.smoke_batches),
                         'sampler_generator_state': loader.generator.get_state()}
        save_checkpoint(run_dir / 'last_checkpoint.pt', final_payload)
        save_checkpoint(run_dir / 'best_checkpoint.pt', final_payload)
        selected = torch.load(run_dir / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model'], strict=True)
        metrics, records = evaluate_cached(model, test_cache, device)
        if len(records) != EXPECTED_COUNTS['test']:
            raise RuntimeError('Final LGVQ predictions are incomplete')
        manifest['frozen_encoder_final_audit'] = assert_encoder_unchanged(model.encoder, frozen_sha)
        write_json(run_dir / 'fixed_weight_predictions.json', records)
        write_json(run_dir / 'result.json', {'status': 'complete', 'selected_epoch': last_epoch,
            'selection': 'fixed_last_epoch', 'selection_uses_test': False, 'checkpoint': 'last_checkpoint.pt',
            'checkpoint_sha256': file_sha256(run_dir / 'last_checkpoint.pt'), 'metrics': metrics,
            'prediction_count': len(records), 'predictions_binding_sha256': structured_sha256(records),
            'parent_checkpoint_sha256': parent_sha, 'optimizer_state_restored': True,
            'training_scope': 'decoder_only', 'formal_result': not bool(args.smoke_batches),
            'partial_epoch': bool(args.smoke_batches), 'frozen_encoder_audit': manifest['frozen_encoder_final_audit'],
            'development_metric': True, 'elapsed_seconds': time.monotonic() - began})
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc),
                        cleanup_errors=list(getattr(exc, 'cleanup_errors', [])))
        raise
    finally:
        cleanup_failure = None
        try:
            shutdown_loader(loader, cleanup_errors=manifest.setdefault('cleanup_errors', []))
        except BaseException as exc:
            # No earlier exception was active: a normal worker-shutdown failure
            # must mark this run failed and remain visible to the coordinator.
            manifest.update(status='failed', error=repr(exc))
            cleanup_failure = exc
        manifest['parent_checkpoint_sha256_after'] = file_sha256(source / 'last_checkpoint.pt')
        manifest['parent_checkpoint_unchanged'] = manifest['parent_checkpoint_sha256_after'] == parent_sha
        if not manifest['parent_checkpoint_unchanged']:
            manifest.update(status='failed', error='Protected source checkpoint changed')
        manifest['finished_unix'] = time.time()
        write_json(run_dir / 'run_manifest.json', manifest)
        torch.cuda.empty_cache()
        if not manifest['parent_checkpoint_unchanged']:
            raise RuntimeError('Protected source checkpoint changed during continuation')
        if cleanup_failure is not None:
            raise cleanup_failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--target', choices=['temporal', 'spatial'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--queue-status-path', type=Path)
    parser.add_argument('--shared-owner-pids', type=int, nargs='*', default=[])
    parser.add_argument('--max-physical-gpus', type=int, choices=[4], default=4)
    parser.add_argument('--max-jobs-per-gpu', type=int, choices=[2, 3], default=3)
    parser.add_argument('--smoke-batches', type=int, choices=[0, 2], default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
