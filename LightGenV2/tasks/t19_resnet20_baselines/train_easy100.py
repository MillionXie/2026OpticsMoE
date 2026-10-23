"""From-scratch, direction-specific easy100 training; no teacher/cache inputs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .easy100_data import Easy100
from .models import build_model, parameter_counts
from .retrieval import cross_modal_loss, exact_sku_ranking

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]


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
    if git('rev-parse', 'HEAD') != git('rev-parse', 'origin/main'):
        raise RuntimeError('HEAD must match fetched published origin/main')


def admit_gpu(uuid):
    if not uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the single requested GPU UUID')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    if any(line.split(',')[0].strip() == uuid for line in apps.splitlines()):
        raise RuntimeError('Requested GPU already has a compute process')
    state = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, utilization = (int(x.strip()) for x in state.strip().split(','))
    if memory > 100 or utilization > 5:
        raise RuntimeError('Requested GPU is not idle')


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
    if epoch <= 5:
        return base * epoch / 5
    return base * .5 * (1 + math.cos(math.pi * (epoch - 5) / max(epochs - 5, 1)))


def run(args):
    require_published_source()
    authorization = json.loads((TASK / 'configs/execution_authorization_20261023.json').read_text())
    if not authorization['training_authorized_after_required_preflight']:
        raise RuntimeError('Training authorization missing')
    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError('Invalid budget')
    run_dir = args.run_dir.resolve()
    category = 'smoke' if args.smoke_batches else 'simulation'
    if not run_dir.is_relative_to((TASK / 'runs' / category).resolve()):
        raise ValueError('Run must use the task runs/smoke or runs/simulation directory')
    # Verify every original image and CSV before taking a GPU.
    dataset = Easy100(args.data_root)
    admit_gpu(args.gpu_uuid)
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
    model = build_model(task_id).to(device)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=True)
    if args.workers:
        loader_options.update(multiprocessing_context='spawn', persistent_workers=False)
    train = DataLoader(dataset.images('train'), shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed), **loader_options)
    test = DataLoader(dataset.images('test'), shuffle=False, **loader_options)
    manifest = {
        'status': 'running', 'task': task_id, 'seed': args.seed, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'precision': 'FP32_TF32_disabled',
        'git_commit': git('rev-parse', 'HEAD'), 'command': [sys.executable, '-m', __spec__.name, *sys.argv[1:]],
        'parameters': parameter_counts(model), 'data_identity': dataset.identity,
        'remote_original_data_verified': True, 'train_images': len(dataset.train),
        'test_images': len(dataset.test), 'title_candidates': len(dataset.titles),
        'gpu_uuid': args.gpu_uuid, 'gpu': torch.cuda.get_device_name(),
        'pid': os.getpid(), 'torch': torch.__version__, 'python': platform.python_version(),
        'selection': 'TEST Hit@1, development metric; epoch1/every5/final',
        'optimizer': 'AdamW_lr0.001_wd0.0001_warmup5_cosine_clip1',
        'initialization': 'random_all_parameters_no_teacher',
        'direction_independent_weights': True, 'smoke_batches': args.smoke_batches,
        'started_unix': time.time(),
    }
    write_json(run_dir / 'run_manifest.json', manifest)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tokens = dataset.title_tokens.to(device)
    history, best, selected_epoch = [], -float('inf'), None
    began = time.monotonic()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            lr = learning_rate(epoch, args.epochs)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += loss.item() * len(batch['sku'])
                count += len(batch['sku'])
                if step % 25 == 0:
                    write_json(run_dir / 'progress.json', {'status': 'training', 'epoch': epoch,
                        'step': step, 'loss': total / count, 'elapsed_seconds': time.monotonic()-began})
                if args.smoke_batches and step >= args.smoke_batches:
                    break
            result = evaluate(model, test, tokens, dataset.title_skus, args.direction, device) if (
                epoch == 1 or epoch % 5 == 0 or epoch == args.epochs) else None
            entry = {'epoch': epoch, 'train_loss': total/count, 'learning_rate': lr,
                     'elapsed_seconds': time.monotonic()-began}
            if result:
                entry.update({k: v for k, v in result.items() if isinstance(v, (float, int))})
            payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'epoch': epoch, 'manifest': manifest, 'metrics': entry}
            save_checkpoint(run_dir / 'last_checkpoint.pt', payload)
            if result and result['hit_at_1'] > best:
                best, selected_epoch = result['hit_at_1'], epoch
                save_checkpoint(run_dir / 'best_checkpoint.pt', payload)
                write_json(run_dir / 'best_predictions.json', result)
            history.append(entry)
            write_json(run_dir / 'history.json', history)
            write_json(run_dir / 'progress.json', {'status': 'training', **entry,
                                                  'best_hit_at_1': best, 'selected_epoch': selected_epoch})
            print(json.dumps(entry), flush=True)
        selected = torch.load(run_dir / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model'])
        result = evaluate(model, test, tokens, dataset.title_skus, args.direction, device)
        write_json(run_dir / 'fixed_weight_predictions.json', result)
        write_json(run_dir / 'result.json', {'status': 'complete', 'selected_epoch': selected_epoch,
            'development_metric': True, 'metrics': {k:v for k,v in result.items() if isinstance(v,(float,int))},
            'elapsed_seconds': time.monotonic()-began})
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['finished_unix'] = time.time()
        write_json(run_dir / 'run_manifest.json', manifest)
        # Spawn workers exit at the end of each iterator; never kill unrelated PIDs.
        del model, optimizer, tokens
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', choices=['image_to_text', 'text_to_image'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke-batches', type=int, default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
