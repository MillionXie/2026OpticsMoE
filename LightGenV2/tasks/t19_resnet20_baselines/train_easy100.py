"""Direction-specific easy100 training with explicit initialization/freeze profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .easy100_data import Easy100
from .models import build_model, parameter_counts
from .retrieval import cross_modal_loss, exact_sku_ranking

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
PROFILE_PATH = TASK / 'configs/execution_single_seed30_20261023.json'
PROFILE = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def require_published_source():
    relative = TASK.relative_to(ROOT).as_posix()
    if git('branch', '--show-current') != 'main':
        raise RuntimeError('Training requires the existing main branch')
    if git('status', '--porcelain', '--', relative):
        raise RuntimeError('Task source has uncommitted changes')
    head=git('rev-parse','HEAD')
    if git('merge-base',head,'origin/main')!=head:
        # Another window can publish directly from the server without updating
        # this checkout's origin/main. Read-only remote proof avoids a fetch
        # transaction inside a training worker.
        remote=git('ls-remote','origin','refs/heads/main').split()
        if not remote or remote[0]!=head:
            raise RuntimeError('HEAD must be a published commit in origin/main history')


def validate_gpu_processes(uuid, apps, allowed_pids, max_jobs_per_gpu=2):
    if max_jobs_per_gpu not in (2, 3):
        raise ValueError('GPU capacity must be two or three tasks')
    active = {int(line.split(',')[1].strip()) for line in apps.splitlines()
              if line.split(',')[0].strip() == uuid}
    if active - set(allowed_pids):
        raise RuntimeError('Requested GPU has a process outside this T19 queue')
    if len(active) >= max_jobs_per_gpu:
        capacity = 'two' if max_jobs_per_gpu == 2 else 'three'
        raise RuntimeError(f'Requested GPU already has {capacity} tasks')
    for pid in active:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if (not any(module in command for module in (
                'LightGenV2.tasks.t19_resnet20_baselines.train_easy100',
                'LightGenV2.tasks.t19_resnet20_baselines.train_tasks'))
                or uuid not in command or str(TASK / 'runs') not in command):
            raise RuntimeError('Shared process is not a verified T19 worker on this GPU')
    return active


def admit_gpu(uuid, allowed_pids=(), max_jobs_per_gpu=2):
    if not uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the single requested GPU UUID')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    active = validate_gpu_processes(uuid, apps, allowed_pids, max_jobs_per_gpu)
    state = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, total, utilization = (int(x.strip()) for x in state.strip().split(','))
    if not active and memory > 100:
        raise RuntimeError('Unattributed GPU allocation')
    utilization_limit = 50 if len(active) >= 2 else 95
    minimum_free = PROFILE['minimum_free_memory_before_second_job_mib']
    if len(active) >= 2:
        minimum_free = max(6144, minimum_free)
    if total - memory < minimum_free or utilization > utilization_limit:
        raise RuntimeError('Insufficient GPU memory or compute headroom')
    return active


def sharing_pids(args):
    if args.queue_status_path is None:
        return args.shared_owner_pids
    path = args.queue_status_path.resolve()
    if not path.is_relative_to((TASK / 'runs').resolve()) or path.name != 'queue_status.json':
        raise ValueError('Shared-worker registry must be in this task runs')
    state = json.loads(path.read_text(encoding='utf-8'))
    jobs = [job for job in state['jobs'] if job['status'] == 'running' and job['gpu_uuid'] == args.gpu_uuid]
    if (len(jobs) > getattr(args, 'max_jobs_per_gpu', 2)
            or not any(job['pid'] == os.getpid() and job['name'] == args.run_dir.name for job in jobs)):
        raise RuntimeError('This worker is not registered in the shared-GPU queue')
    # Read after data verification so either worker may finish CPU preflight
    # first; a launch-time PID snapshot alone races concurrent CUDA admission.
    return [job['pid'] for job in jobs if job['pid'] != os.getpid()]


def wait_for_gpu(args):
    """Wait without CUDA if another job arrives during CPU data verification."""
    resource_errors = {'Requested GPU has a process outside this T19 queue',
        'Requested GPU already has two tasks', 'Requested GPU already has three tasks', 'Unattributed GPU allocation',
        'Insufficient GPU memory or compute headroom'}
    while True:
        allowed = sharing_pids(args)
        try:
            cap = getattr(args, 'max_jobs_per_gpu', 2)
            if cap > 2:
                admit_gpu(args.gpu_uuid, allowed, cap)
            else:
                admit_gpu(args.gpu_uuid, allowed)
            return allowed
        except RuntimeError as exc:
            if str(exc) not in resource_errors:
                raise
            print('Waiting without CUDA before admission:', str(exc), flush=True)
        except FileNotFoundError:
            # A registered peer can finish between its PID and cmdline reads.
            pass
        time.sleep(5)


@torch.inference_mode()
def evaluate(model, loader, tokens, title_skus, direction, device):
    model.eval()
    features, skus, ids = [], [], []
    for batch in loader:
        features.append(model(batch['image'].to(device)).cpu())
        skus.extend(batch['sku'])
        ids.extend(batch['sample_id'])
    images = torch.cat(features)
    titles = model.encode_text(tokens.to(device)).cpu()
    scores = images @ titles.T
    if direction == 'image_to_text':
        result = exact_sku_ranking(scores, skus, title_skus)
        result.update(query_ids=ids, candidate_ids=title_skus)
    else:
        result = exact_sku_ranking(scores.T, title_skus, skus)
        result.update(query_ids=title_skus, candidate_ids=ids)
    return result


def learning_rate(epoch, epochs, base=1e-3):
    return base


def frozen_encoder_hashes(model, include_text=True):
    """Bind only the encoders frozen by this explicit text-training profile."""
    from .frozen_resnet18 import encoder_state_sha256
    encoders = [('visual', model.encoder)]
    if include_text:
        encoders.append(('text', model.text_encoder))
    return {name: encoder_state_sha256(encoder) for name, encoder in encoders}


def audit_frozen_encoders(model, expected_hashes):
    """Reject parameter, gradient, running-statistic or training-mode drift."""
    from .frozen_resnet18 import assert_encoder_unchanged
    if set(expected_hashes) not in ({'visual'}, {'visual', 'text'}):
        raise ValueError('Frozen audit requires visual and optionally text identities')
    encoders = {'visual': model.encoder, 'text': model.text_encoder}
    return {name: assert_encoder_unchanged(encoder, expected_hashes[name])
            for name, encoder in encoders.items() if name in expected_hashes}


def run(args):
    require_published_source()
    authorization = json.loads((TASK / 'configs/execution_authorization_20261023.json').read_text())
    if not authorization['training_authorized_after_required_preflight']:
        raise RuntimeError('Training authorization missing')
    if (args.epochs < 1 or args.batch_size < 2 or args.smoke_batches < 0
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Invalid budget')
    if args.seed not in PROFILE['seeds'] or (not args.smoke_batches and args.epochs != PROFILE['epochs']):
        raise ValueError('Formal training requires the authorized single seed and 30-epoch budget')
    frozen = args.backbone == 'resnet18_imagenet_frozen'
    text_encoder = getattr(args, 'text_encoder', 'minilm')
    if text_encoder not in {'minilm', 'byte_gru'}:
        raise ValueError('Unknown text encoder profile')
    small_text = frozen and text_encoder == 'byte_gru'
    if frozen and (args.learning_rate != PROFILE['learning_rate'] or args.batch_size != PROFILE['batch_size']):
        raise ValueError('Frozen encoder profile requires fixed lr0.001 and effective batch32')
    run_dir = args.run_dir.resolve()
    category = 'smoke' if args.smoke_batches else 'simulation'
    if not run_dir.is_relative_to((TASK / 'runs' / category).resolve()):
        raise ValueError('Run must use the task runs/smoke or runs/simulation directory')
    # Verify every original image and CSV before taking a GPU.
    dataset = Easy100(args.data_root)
    shared_pids = wait_for_gpu(args)
    run_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    task_id = 'abo_' + args.direction
    model_options = {'backbone': args.backbone}
    # Keep historical builders and test doubles compatible with their original
    # two-argument API; the new override is explicit only for the small text run.
    if text_encoder == 'byte_gru':
        model_options['text_encoder'] = text_encoder
    model = build_model(task_id, **model_options)
    counts = parameter_counts(model)
    if small_text:
        if counts['total'] >= 15_000_000:
            raise ValueError('Small-text profile requires complete model strictly below 15M parameters')
        if counts['text_encoder'] != 27_104:
            raise ValueError('Small-text profile requires the complete 27,104-parameter byte-GRU')
        if any(not p.requires_grad for p in model.text_encoder.parameters()):
            raise RuntimeError('Small random text encoder must remain fully trainable')
    model = model.to(device)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=True)
    if args.workers:
        loader_options.update(multiprocessing_context='spawn', persistent_workers=True)
    train_images = dataset.images('train')
    if args.smoke_batches:
        train_images = Subset(train_images, range(min(len(train_images), args.smoke_batches * args.batch_size)))
    train = DataLoader(train_images, shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed), **loader_options)
    test = DataLoader(dataset.images('test'), shuffle=False, **loader_options)
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen_hashes = None
    if frozen:
        frozen_hashes = frozen_encoder_hashes(model, include_text=not small_text)
        audit_frozen_encoders(model, frozen_hashes)
        frozen_encoders = (model.encoder,) if small_text else (model.encoder, model.text_encoder)
        encoder_ids = {id(p) for encoder in frozen_encoders
                       for p in encoder.parameters()}
        if any(id(p) in encoder_ids for p in trainable):
            raise RuntimeError('A frozen visual or text encoder entered the optimizer')
    manifest = {
        'status': 'running', 'task': task_id, 'backbone': args.backbone, 'seed': args.seed, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'precision': 'FP32_TF32_disabled',
        'git_commit': git('rev-parse', 'HEAD'), 'command': [sys.executable, '-m', __spec__.name, *sys.argv[1:]],
        'parameters': counts, 'data_identity': dataset.identity,
        'remote_original_data_verified': True, 'train_images': len(dataset.train),
        'test_images': len(dataset.test), 'title_candidates': len(dataset.titles),
        'gpu_uuid': args.gpu_uuid, 'gpu': torch.cuda.get_device_name(),
        'pid': os.getpid(), 'torch': torch.__version__, 'python': platform.python_version(),
        'selection': 'fixed last epoch; no metric-based selection',
        'metric_name': 'Hit@1', 'schedule': 'constant', 'warmup_epochs': 0,
        'learning_rate': args.learning_rate, 'execution_profile': dict(PROFILE,
            max_physical_gpus=getattr(args,'max_physical_gpus',3),
            max_jobs_per_gpu=getattr(args,'max_jobs_per_gpu',2)),
        'optimizer': 'AdamW_constant_lr_wd0.0001_clip1',
        'initialization': ('official_pretrained_visual_random_byte_gru' if small_text else
                           'official_pretrained_visual_and_text' if frozen
                           else 'random_all_parameters_no_teacher'),
        'training_scope': ('frozen_visual_trainable_text_and_decoder' if small_text else
                           'decoder_only' if frozen else 'full_network'),
        'text_encoder': text_encoder if frozen else 'byte_gru',
        'optimizer_parameters': sum(p.numel() for p in trainable),
        'direction_independent_weights': True, 'smoke_batches': args.smoke_batches,
        'shared_owner_pids_at_launch': shared_pids,
        'queue_status_path': str(args.queue_status_path) if args.queue_status_path else None,
        'max_jobs_per_gpu': getattr(args,'max_jobs_per_gpu',2),
        'started_unix': time.time(),
    }
    if frozen:
        provenance = {'visual': model.encoder.pretrained_provenance}
        if not small_text:
            provenance['text'] = model.text_encoder.pretrained_provenance
        manifest.update(pretrained_provenance=provenance,
            encoder_state_sha256_before=frozen_hashes,
            frozen_encoder_audits=[],
            source_sha256={'train_easy100.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        if small_text:
            from .frozen_resnet18 import encoder_state_sha256
            manifest.update(text_encoder_initialization={
                'initialization': 'random_no_pretraining', 'architecture': 'UTF8_byte_embedding259x32_GRU32to64',
                'parameters': counts['text_encoder'], 'all_parameters_trainable': True},
                complete_model_parameter_limit_exclusive=15_000_000,
                trainable_text_state_sha256_before=encoder_state_sha256(model.text_encoder))
    write_json(run_dir / 'run_manifest.json', manifest)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=PROFILE['weight_decay'])
    tokens = dataset.title_tokens.to(device)
    history, selected_epoch = [], None
    began = time.monotonic()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            lr = learning_rate(epoch, args.epochs, args.learning_rate)
            for group in optimizer.param_groups:
                group['lr'] = lr
            total, count = 0., 0
            for step, batch in enumerate(train, 1):
                optimizer.zero_grad(set_to_none=True)
                loss = cross_modal_loss(model(batch['image'].to(device)), model.encode_text(tokens),
                    list(batch['sku']), dataset.title_skus, direction=args.direction)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite TRAIN loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                optimizer.step()
                total += loss.item() * len(batch['sku'])
                count += len(batch['sku'])
                if step % 25 == 0:
                    write_json(run_dir / 'progress.json', {'status': 'training', 'epoch': epoch,
                        'step': step, 'loss': total / count, 'elapsed_seconds': time.monotonic()-began})
            result = evaluate(model, test, tokens, dataset.title_skus, args.direction, device) if (
                epoch == 1 or epoch % 5 == 0 or epoch == args.epochs) else None
            entry = {'epoch': epoch, 'train_loss': total/count, 'learning_rate': lr,
                     'elapsed_seconds': time.monotonic()-began}
            if result:
                entry.update({k: v for k, v in result.items() if isinstance(v, (float, int))})
            if frozen:
                manifest['frozen_encoder_audits'].append({
                    'epoch': epoch, **audit_frozen_encoders(model, frozen_hashes)})
                write_json(run_dir / 'run_manifest.json', manifest)
            payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'epoch': epoch, 'manifest': manifest, 'metrics': entry}
            save_checkpoint(run_dir / 'last_checkpoint.pt', payload)
            if epoch == args.epochs:
                selected_epoch = epoch
                save_checkpoint(run_dir / 'best_checkpoint.pt', payload)
                write_json(run_dir / 'best_predictions.json', result)
            history.append(entry)
            write_json(run_dir / 'history.json', history)
            write_json(run_dir / 'progress.json', {'status': 'training', **entry,
                                                  'selection': 'fixed_last_epoch', 'selected_epoch': selected_epoch})
            print(json.dumps(entry), flush=True)
        selected = torch.load(run_dir / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model'])
        result = evaluate(model, test, tokens, dataset.title_skus, args.direction, device)
        if frozen:
            manifest['frozen_encoder_final_audit'] = audit_frozen_encoders(model, frozen_hashes)
            manifest['encoder_state_sha256_after'] = frozen_encoder_hashes(model, include_text=not small_text)
            if small_text:
                manifest['trainable_text_state_sha256_after'] = encoder_state_sha256(model.text_encoder)
        write_json(run_dir / 'fixed_weight_predictions.json', result)
        write_json(run_dir / 'result.json', {'status': 'complete', 'selected_epoch': selected_epoch,
            'selection': 'fixed_last_epoch', 'selection_uses_test': False, 'checkpoint': 'last_checkpoint.pt',
            'checkpoint_sha256': hashlib.sha256((run_dir / 'last_checkpoint.pt').read_bytes()).hexdigest(),
            'training_scope': manifest['training_scope'],
            'frozen_encoder_audit': manifest.get('frozen_encoder_final_audit'),
            'development_metric': True, 'metrics': {k:v for k,v in result.items() if isinstance(v,(float,int))},
            'elapsed_seconds': time.monotonic()-began})
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        manifest['finished_unix'] = time.time()
        write_json(run_dir / 'run_manifest.json', manifest)
        # Spawn workers exit at the end of each iterator; never kill unrelated PIDs.
        del model, optimizer, tokens
        for loader in (train, test):
            iterator = getattr(loader, '_iterator', None)
            if iterator is not None:
                iterator._shutdown_workers()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', choices=['image_to_text', 'text_to_image'], required=True)
    parser.add_argument('--backbone', choices=['resnet20','resnet32','resnet18_imagenet_frozen'], default='resnet20')
    parser.add_argument('--text-encoder', choices=['minilm','byte_gru'], default='minilm')
    parser.add_argument('--max-physical-gpus',type=int,choices=[3,4,6],default=3)
    parser.add_argument('--max-jobs-per-gpu',type=int,choices=[2,3],default=2)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--seed', type=int, default=PROFILE['seeds'][0])
    parser.add_argument('--epochs', type=int, default=PROFILE['epochs'])
    parser.add_argument('--learning-rate', type=float, default=PROFILE['learning_rate'])
    parser.add_argument('--batch-size', type=int, default=PROFILE['batch_size'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke-batches', type=int, default=0)
    parser.add_argument('--shared-owner-pids', type=int, nargs='*', default=[])
    parser.add_argument('--queue-status-path', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
