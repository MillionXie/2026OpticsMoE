"""Paired full-split experiment. Select on validation, lock, then evaluate TEST once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.demo_check.pure_optical.models import encode
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256
from LightGenV2.tasks.t17_router_classification.gpu_policy import require_authorized_gpu
from LightGenV2.tasks.t17_router_classification.readout_data import load_raw
from LightGenV2.tasks.t17_router_classification.train import write
from .model import SpatialRoutingLG


TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]


def state_digest(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def source_files():
    """Hash actual imported project sources, including the archived optical propagator."""
    sources = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == '.py' and path.is_relative_to(REPO / 'LightGenV2'):
            sources.add(path)
    return {path.relative_to(REPO).as_posix(): sha256(path) for path in sorted(sources)}


def load_inputs(config, device):
    raw, identity = load_raw(Path(config['data_root']), config)
    for field in ('manifest_sha256', 'index_sha256'):
        if identity[field] != config[field]:
            raise ValueError(f'Original Kather {field} differs from the frozen split')
    inputs = {}
    for split, part in raw.items():
        images = part.pop('images')
        # Encode once, keep whole splits on the host to permit sharing GPU memory.
        fields = torch.empty(len(images), 224, 224, dtype=torch.float32)
        for start in range(0, len(images), 64):
            rows = slice(start, start + 64)
            fields[rows] = encode(torch.from_numpy(images[rows]).to(device)).cpu()
        inputs[split] = (fields, torch.as_tensor(part['labels'], dtype=torch.long), part['ids'].astype(str))
    return inputs, identity


def route_summary(positions):
    extent = 478 - 224
    corner_distances = np.minimum(positions, extent - positions)
    offsets = np.abs(positions[:, 0] - positions[:, 1])
    overlap = np.maximum(224 - offsets, 0).prod(axis=1) / (224**2)
    return {
        'coordinate_order': 'top-left y,x in active 478x478 plane pixels',
        'mean_pair_distance_px': float(np.linalg.norm(offsets, axis=1).mean()),
        'mean_distance_to_nearest_corner_px': float(np.linalg.norm(corner_distances, axis=2).mean()),
        'mean_overlap_fraction_of_one_copy': float(overlap.mean()),
        'distinct_locations_rounded_px': int(len(np.unique(np.rint(positions.reshape(-1, 2)), axis=0))),
    }


@torch.inference_mode()
def evaluate(model, split_data, batch_size, device, prediction_path=None):
    model.eval()
    fields, labels, ids = split_data
    batches = {key: [] for key in ('logits', 'positions', 'power', 'router_capture')}
    for start in range(0, len(ids), batch_size):
        output = model(fields[start:start + batch_size].to(device))
        for key in batches:
            batches[key].append(output[key].cpu())
    logits, positions, power, capture = (torch.cat(batches[key]) for key in batches)
    scores = logits.softmax(dim=1)
    predictions = scores.argmax(dim=1)
    n_classes = scores.shape[1]
    cm = torch.bincount(labels * n_classes + predictions, minlength=n_classes**2).reshape(n_classes, n_classes)
    if not torch.isfinite(logits).all() or not torch.isfinite(capture).all():
        raise RuntimeError('Non-finite scores or capture')
    if positions.shape != (len(ids), 2, 2) or not torch.isfinite(positions).all():
        raise RuntimeError('Expected two finite 2D positions per image')
    if power.shape != (len(ids), 2) or (power <= 0).any() or not torch.allclose(power.sum(1), torch.ones(len(ids)), atol=1e-5):
        raise RuntimeError('Expected two active copies with normalized power')
    if (positions < -1e-4).any() or (positions > 254.0001).any():
        raise RuntimeError('Input extends beyond the active expert plane')
    correct = int(cm.diag().sum())
    if prediction_path is not None:
        np.savez_compressed(prediction_path, ids=ids, labels=labels.numpy(), scores=scores.numpy(),
                            predictions=predictions.numpy(), positions=positions.numpy(), power=power.numpy(),
                            router_capture=capture.numpy())
    return {
        'n': len(ids), 'correct': correct, 'accuracy': correct / len(ids),
        'cross_entropy': F.cross_entropy(logits, labels).item(), 'confusion_matrix': cm.tolist(),
        'position_diagnostics': route_summary(positions.numpy()),
    }


def save_checkpoint(path, payload):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def run(args, config):
    device = torch.device(args.device)
    gpu_uuid = require_authorized_gpu(config['authorized_gpu_uuids']) if device.type == 'cuda' else None
    torch.set_num_threads(4)
    seed = config['seed']
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpatialRoutingLG(mode=args.mode, num_classes=config['num_classes'])
    source_commit = os.environ.get('T24_SOURCE_COMMIT') or subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    metadata = {
        'mode': args.mode, 'config': config, 'config_file_sha256': sha256(args.config),
        'source_commit': source_commit, 'source_sha256': source_files(),
        'initial_state_sha256': state_digest(model), 'pid': os.getpid(),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'command': [sys.executable, '-m', __spec__.name, *sys.argv[1:]],
        'gpu_uuid': gpu_uuid, 'device': str(device),
        'gpu_model': torch.cuda.get_device_name(device) if gpu_uuid else None,
        'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'selection': 'highest original VAL correct count, then lowest VAL CE; earlier exact tie',
        'role': 'T24 TRAIN-only fit, VAL selection. T17 previously exposed original Kather TEST in development.',
    }
    write(args.out / 'metadata.json', metadata)
    inputs, identity = load_inputs(config, device)
    metadata['data'] = identity
    write(args.out / 'metadata.json', metadata)
    model.to(device)
    phase_parameters = [p for name, p in model.named_parameters() if name != 'shared_head.weight']
    optimizer = torch.optim.Adam([
        {'params': phase_parameters, 'lr': config['phase_lr']},
        {'params': model.shared_head.parameters(), 'lr': config['head_lr']},
    ])
    history = []
    best_key = None
    fields, labels, _ = inputs['train']
    batch_size = config['batch']
    for epoch in range(1, config['epochs'] + 1):
        started = time.monotonic()
        model.train()
        order = rng.permutation(len(labels)).astype(np.int64)
        loss_sum = torch.zeros((), device=device)
        for start in range(0, len(order), batch_size):
            index = torch.from_numpy(order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            output = model(fields[index].to(device))
            loss = F.cross_entropy(output['logits'], labels[index].to(device), label_smoothing=config['label_smoothing'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.detach() * len(index)
        val = evaluate(model, inputs['val'], batch_size, device)
        row = {'epoch': epoch, 'train_loss': loss_sum.item() / len(order), 'validation': val,
               'train_steps': (len(order) + batch_size - 1) // batch_size,
               'minibatch_order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
               'elapsed_seconds': time.monotonic() - started}
        history.append(row)
        key = (val['correct'], -val['cross_entropy'])
        payload = {'model': model.state_dict(), 'epoch': epoch, 'mode': args.mode,
                   'config': config, 'validation': val, 'source_commit': source_commit}
        if best_key is None or key > best_key:
            save_checkpoint(args.out / 'best_checkpoint.pt', payload)
            best_key = key
        save_checkpoint(args.out / 'last_checkpoint.pt', dict(payload, optimizer=optimizer.state_dict(),
                        shuffle_rng_state=rng.bit_generator.state, torch_rng_state=torch.get_rng_state()))
        write(args.out / 'history.json', history)
        write(args.out / 'status.json', {'state': 'training', 'epoch': epoch, 'pid': os.getpid()})
        print(f"epoch={epoch} val={val['correct']}/{val['n']} seconds={row['elapsed_seconds']:.1f}", flush=True)

    best_path = args.out / 'best_checkpoint.pt'
    selected = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model'], strict=True)
    selection = {'mode': args.mode, 'selected_epoch': selected['epoch'], 'best_checkpoint_sha256': sha256(best_path),
                 'seed': seed, 'source_commit': source_commit, 'TEST_selection_calls': 0}
    write(args.out / 'selection_lock.json', selection)
    result = dict(selection)
    for split in ('train', 'val', 'test'):
        path = args.out / f'predictions_{split}.npz'
        result[split] = evaluate(model, inputs[split], batch_size, device, path)
    write(args.out / 'result.json', result)
    artifacts = {p.name: sha256(p) for p in sorted(args.out.iterdir()) if p.is_file() and p.name != 'status.json'}
    write(args.out / 'manifest.json', artifacts)
    write(args.out / 'status.json', {'state': 'complete', 'pid': os.getpid(), 'epoch': config['epochs']})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('fixed', 'continuous'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--raw-cache', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.seed is not None:
        config['seed'] = args.seed
    if args.data_root is not None:
        config['data_root'] = str(args.data_root)
    if args.raw_cache is not None:
        config['raw_cache'] = str(args.raw_cache)
    args.out = args.out.resolve()
    if not any(args.out.is_relative_to(TASK / 'runs' / k) for k in ('smoke', 'simulation')):
        parser.error('Output must be within this T24 task runs/smoke or runs/simulation')
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'status.json', {'state': 'initializing', 'pid': os.getpid()})
    try:
        run(args, config)
    except BaseException as exc:
        write(args.out / 'status.json', {'state': 'failed', 'pid': os.getpid(), 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
