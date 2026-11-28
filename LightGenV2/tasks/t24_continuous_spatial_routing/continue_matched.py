"""Continue an archived matched T24 run from its last optimizer and shuffle state."""
import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import train
from .replay_epoch_curves import SPLITS, check, metrics, original_files, read


def load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


@contextmanager
def writer(out):
    # Kernel locks release automatically on process termination, including an interrupted GPU job.
    with (out / '.writer.lock').open('a+b') as handle:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def key(validation):
    return validation['correct'], -validation['cross_entropy']


def verify_training_state(state, history, config, model, optimizer):
    check(state['epoch'] == len(history) and [r['epoch'] for r in history] == list(range(1, len(history) + 1)),
          'Checkpoint epoch disagrees with history')
    check(state['validation'] == history[-1]['validation'], 'Last model VAL differs from committed history')
    check(set(state['model']) == set(model.state_dict()) and len(state['model']) == 4, 'Optical model identity differs')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    n = config['expected_counts']['train']
    steps = (n + config['batch'] - 1) // config['batch']
    rng = np.random.default_rng(config['seed'])
    for row in history:
        order = rng.permutation(n).astype(np.int64)
        check(row['train_steps'] == steps and hashlib.sha256(order.tobytes()).hexdigest() == row['minibatch_order_sha256'],
              f'Archived seed, shuffle, or budget differs at E{row["epoch"]}')
    check(rng.bit_generator.state == state['shuffle_rng_state'], 'NumPy shuffle state differs')
    check(len(optimizer.state) == 4 and all(s['step'].item() == len(history) * steps for s in optimizer.state.values()),
          'Restored Adam update count differs')
    for group, lr in zip(optimizer.param_groups, (config['phase_lr'], config['head_lr'])):
        check(group['lr'] == lr and group['betas'] == (0.9, 0.999) and group['eps'] == 1e-8
              and group['weight_decay'] == 0 and not group['amsgrad'], 'Adam hyperparameters changed')
    rng.bit_generator.state = state['shuffle_rng_state']
    torch.set_rng_state(state['torch_rng_state'])
    return rng


def train_epoch(model, optimizer, inputs, config, rng, device):
    fields, labels, _ = inputs['train']
    batch = config['batch']
    model.train()
    order = rng.permutation(len(labels)).astype(np.int64)
    loss_sum = torch.zeros((), device=device)
    for start in range(0, len(order), batch):
        index = torch.from_numpy(order[start:start + batch])
        optimizer.zero_grad(set_to_none=True)
        output = model(fields[index].to(device))
        loss = F.cross_entropy(output['logits'], labels[index].to(device), label_smoothing=config['label_smoothing'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.detach() * len(index)
    return {'train_loss': loss_sum.item() / len(order), 'train_steps': (len(order) + batch - 1) // batch,
            'minibatch_order_sha256': hashlib.sha256(order.tobytes()).hexdigest()}


def materialize_best(out, best):
    path = out / 'best_checkpoint.pt'
    if path.exists():
        existing = load_pt(path)
        if existing['epoch'] == best['epoch'] and existing['mode'] == best['mode'] and all(
                torch.equal(existing['model'][k], v.cpu()) for k, v in best['model'].items()):
            return
    train.save_checkpoint(path, best)


def run(original, out, target, device):
    files, parent, _, parent_result = original_files(original)
    config = parent['config']
    parent_epoch = config['epochs']
    check(target > parent_epoch, 'Target must exceed the original training budget')
    gpu_uuid = train.require_authorized_gpu(config['authorized_gpu_uuids']) if device.type == 'cuda' else None
    check(device.type == parent['device'] and gpu_uuid == parent['gpu_uuid'] and str(torch.__version__) == parent['torch']
          and torch.version.cuda == parent['cuda'] and platform.python_version() == parent['python'],
          'Original GPU or Python/PyTorch/CUDA runtime differs')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    model = train.build_model(parent['mode'], config['num_classes'])
    check(train.state_digest(model) == parent['initial_state_sha256'], 'Original initialization differs')
    model.to(device)
    phase = [p for name, p in model.named_parameters() if name != 'shared_head.weight']
    optimizer = torch.optim.Adam([{'params': phase, 'lr': config['phase_lr']},
                                  {'params': model.shared_head.parameters(), 'lr': config['head_lr']}])
    sources = train.source_files()
    for rel, expected in parent['source_sha256'].items():
        check(sources.get(rel) == expected, f'Training dependency identity differs: {rel}')
    commit = os.environ.get('T24_SOURCE_COMMIT') or subprocess.check_output(
        ['git', '-C', str(train.REPO), 'rev-parse', 'HEAD'], text=True).strip()
    lineage = {'original_run': original.relative_to(train.TASK).as_posix(), 'parent_source_commit': parent['source_commit'],
               'original_manifest_sha256': train.sha256(original / 'manifest.json'),
               'original_last_sha256': files['last_checkpoint.pt'], 'original_best_sha256': files['best_checkpoint.pt'],
               'original_history_sha256': files['history.json'], 'trainer_sha256': train.sha256(Path(__file__))}
    meta_path = out / 'metadata.json'
    if meta_path.exists():
        previous = read(meta_path)
        check(previous.get('lineage') == lineage and previous['source_sha256'] == sources,
              'Existing output belongs to another run or training source')
        check(target >= previous['config']['epochs'], 'A resumed run cannot lower its requested epoch target')
    else:
        check(not any(p.name != '.writer.lock' for p in out.iterdir()), 'Output directory contains unrelated files')
    base_hist = read(original / 'history.json')
    last_path = out / 'last_checkpoint.pt'
    if last_path.exists():
        state = load_pt(last_path)
        check(state['lineage'] == lineage, 'Continuation checkpoint lineage differs')
        history, best = state['history'], state['selected_best']
        check(history[:parent_epoch] == base_hist, 'Original E1..E8 history was modified')
    else:
        state = load_pt(original / 'last_checkpoint.pt')
        best = load_pt(original / 'best_checkpoint.pt')
        history = base_hist
        check(state['epoch'] == parent_epoch and state['source_commit'] == parent['source_commit']
              and best['epoch'] == parent_result['selected_epoch'], 'Archived parent PT identity differs')
    check(state['mode'] == best['mode'] == parent['mode'] and state['epoch'] <= target, 'State mode or target mismatch')
    check(best['epoch'] == max(history, key=lambda r: key(r['validation']))['epoch']
          and best['validation'] == history[best['epoch'] - 1]['validation'], 'Stored best fails the original VAL rule')
    effective = {**config, 'epochs': target}
    metadata = {**parent, 'config': effective, 'parent_config': config, 'target_epoch': target, 'source_sha256': sources,
                'source_commit': commit, 'lineage': lineage, 'parent_epoch': parent_epoch, 'pid': os.getpid(),
                'command': [sys.executable, '-m', __spec__.name, *sys.argv[1:]],
                'parent_config_file_sha256': parent['config_file_sha256'],
                'TEST_use': 'post-update metrics only; no optimization, checkpoint selection or stopping'}
    # Epoch targets are CLI overrides rather than the archived eight-epoch config file.
    train.write(out / 'config.json', effective)
    metadata['config_file_sha256'] = train.sha256(out / 'config.json')
    train.write(meta_path, metadata)
    train.write(out / 'status.json', {'state': 'restoring', 'epoch': state['epoch'], 'target_epoch': target, 'pid': os.getpid()})
    inputs, identity = train.load_inputs(config, device)
    check(identity == parent['data'], 'Original pixel/data identity differs')
    for s in SPLITS:
        with np.load(original / f'predictions_{s}.npz', allow_pickle=False) as original_preds:
            check(np.array_equal(inputs[s][1].numpy(), original_preds['labels']) and np.array_equal(inputs[s][2], original_preds['ids']),
                  f'Original {s} IDs, labels, or order differ')
    rng = verify_training_state(state, history, config, model, optimizer)
    if device.type == 'cuda' and 'cuda_rng_state' in state:
        torch.cuda.set_rng_state(state['cuda_rng_state'], device)
    epoch0 = state['epoch']
    del state
    materialize_best(out, best)
    train.write(out / 'history.json', history)
    batch = config['batch']
    for epoch in range(epoch0 + 1, target + 1):
        start = time.monotonic()
        row = train_epoch(model, optimizer, inputs, config, rng, device)
        val = train.evaluate(model, inputs['val'], batch, device)
        # Preserve the same tie ordering and model-selection input as the archived trainer.
        improved = key(val) > key(best['validation'])
        snapshot = {'model': model.state_dict(), 'epoch': epoch, 'mode': parent['mode'], 'config': effective,
                    'validation': val, 'source_commit': commit}
        if improved:
            best = {**snapshot, 'model': {k: v.detach().cpu().clone() for k, v in snapshot['model'].items()}}
        gpus = [torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=gpus):
            measured = {s: metrics(train.evaluate(model, inputs[s], batch, device)) for s in ('train', 'test')}
        row.update(epoch=epoch, validation=val, **measured, elapsed_seconds=time.monotonic() - start)
        history.append(row)
        checkpoint = {**snapshot, 'optimizer': optimizer.state_dict(), 'shuffle_rng_state': rng.bit_generator.state,
                      'torch_rng_state': torch.get_rng_state(), 'selected_best': best, 'history': history, 'lineage': lineage}
        if device.type == 'cuda':
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state(device)
        # Commit the entire recoverable epoch in one atomic replace before updating views.
        train.save_checkpoint(last_path, checkpoint)
        if improved:
            train.save_checkpoint(out / 'best_checkpoint.pt', best)
        train.write(out / 'history.json', history)
        train.write(out / 'status.json', {'state': 'training', 'epoch': epoch, 'selected_epoch': best['epoch'],
                                         'target_epoch': target, 'pid': os.getpid()})
        print(f"epoch={epoch} val={val['correct']}/{val['n']} selected={best['epoch']} seconds={row['elapsed_seconds']:.1f}",
              flush=True)
    model.load_state_dict(best['model'], strict=True)
    selection = {'mode': parent['mode'], 'seed': config['seed'], 'source_commit': commit, 'selected_epoch': best['epoch'],
                 'best_checkpoint_sha256': train.sha256(out / 'best_checkpoint.pt'), 'TEST_selection_calls': 0,
                 'target_epoch': target}
    train.write(out / 'selection_lock.json', selection)
    result = {**selection, 'end_of_epoch_TEST_diagnostic_calls_since_parent': target - parent_epoch}
    for s in SPLITS:
        temp_npz = out / f'predictions_{s}.tmp.npz'
        result[s] = train.evaluate(model, inputs[s], batch, device, temp_npz)
        temp_npz.replace(out / f'predictions_{s}.npz')
    check(result['val']['correct'] == best['validation']['correct'], 'Locked model VAL prediction count mismatch')
    train.write(out / 'result.json', result)
    artifacts = ['best_checkpoint.pt', 'last_checkpoint.pt', 'history.json', 'metadata.json', 'config.json',
                 'selection_lock.json', 'result.json', *(f'predictions_{s}.npz' for s in SPLITS)]
    train.write(out / 'manifest.json', {name: train.sha256(out / name) for name in artifacts})
    train.write(out / 'status.json', {'state': 'complete', 'epoch': target, 'target_epoch': target, 'pid': os.getpid(),
                                     'selected_epoch': best['epoch']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target-epoch', type=int, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    original, out = args.run.resolve(), args.out.resolve()
    roots = (train.TASK / 'runs' / 'simulation', train.TASK / 'runs' / 'smoke')
    if (not original.is_relative_to(roots[0]) or not any(out.is_relative_to(p) and out != p for p in roots)
            or out.is_relative_to(original) or original.is_relative_to(out)):
        parser.error('Use separate original T24 simulation and new/resumed T24 run directories')
    out.mkdir(parents=True, exist_ok=True)
    with writer(out):
        run(original, out, args.target_epoch, torch.device(args.device))


if __name__ == '__main__':
    main()
