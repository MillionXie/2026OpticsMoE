"""Train and evaluate matched fixed-position optical experts with optional local fiber projections."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t24_continuous_spatial_routing import train as paired
from LightGenV2.tasks.t24_continuous_spatial_routing.continue_matched import (
    key, load_pt, materialize_best, train_epoch, verify_training_state, writer)
from LightGenV2.tasks.t24_continuous_spatial_routing.replay_epoch_curves import SPLITS, check, metrics, read
from .model import HeterogeneousFiberLG


TASK = Path(__file__).resolve().parent
PROTOCOL = TASK / 'configs' / 'matched_fiber_20261128.json'
DATA_CONFIGS = {'kather': 'kather2016.json', 'blood': 'bloodmnist128.json'}
DIAGNOSTIC_FIELDS = {'slot_indices': 'selected_slots', 'selected_gate_power': 'slot_gate_power',
                     'dense_power': 'dense_power', 'router_port_energy': 'router_port_energy',
                     'pre_fiber_power': 'pre_fiber_power', 'transmitted_power': 'transmitted_power'}


def configuration(dataset, seed, arm, epochs):
    protocol = read(PROTOCOL)
    check(dataset in protocol['datasets'] and arm in protocol['arms'] and seed in protocol['seeds'],
          'Run is absent from the matched cohort')
    parent = TASK.parent / protocol['base_configs'][dataset]
    cfg = read(parent)
    check(all(cfg[k] == protocol['training'][k] for k in ('phase_lr', 'head_lr', 'batch', 'label_smoothing')),
          'Base data config optimizer differs from the matched protocol')
    slots = protocol['fiber_slots_zero_based'][arm]
    if isinstance(slots, dict):
        slots = slots[str(seed)]
    side = protocol['fiber_model']['cores_per_expert_side']
    counts = [((side**2 // 2) if arm == 'uniform_half' else side**2) if i in slots else 0 for i in range(4)]
    check(sum(counts) == protocol['fiber_channels'][arm], 'Per-slot fiber count differs from the registered total')
    return {**cfg, 'arm': arm, 'seed': seed, 'fiber_slots': slots, 'fiber_core_counts': counts,
            'fiber_model': protocol['fiber_model'], 'epochs': epochs, 'protocol_sha256': paired.sha256(PROTOCOL),
            'assigned_gpu_uuid': protocol['gpu_uuids'][str(seed)], 'data_config_sha256': paired.sha256(parent)}


def make_model(config):
    return HeterogeneousFiberLG(config['arm'], num_classes=config['num_classes'], fiber_slots=tuple(config['fiber_slots']))


def optimizer_for(model, config):
    phase = [p for k, p in model.named_parameters() if k != 'shared_head.weight']
    return torch.optim.Adam([{'params': phase, 'lr': config['phase_lr']},
                             {'params': model.shared_head.parameters(), 'lr': config['head_lr']}])


def check_power(z, fiber_slots):
    slots, positions, gates = z['slot_indices'], z['positions'], z['selected_gate_power']
    n = len(slots)
    check(slots.shape == (n, 2) and np.issubdtype(slots.dtype, np.integer) and (slots >= 0).all()
          and (slots < 4).all() and (slots[:, 0] != slots[:, 1]).all(), 'Invalid selected expert indices')
    corners = np.array([[0, 0], [0, 254], [254, 0], [254, 254]])
    check(np.array_equal(positions, corners[slots]), 'Selected expert indices disagree with optical placement')
    for k in ('selected_gate_power', 'dense_power', 'router_port_energy', 'pre_fiber_power', 'transmitted_power'):
        check(z[k].shape == (n, 4) and np.isfinite(z[k]).all() and (z[k] >= 0.).all(), f'Invalid {k}')
    expected = np.zeros_like(gates)
    np.put_along_axis(expected, slots, z['power'], axis=1)
    check(np.array_equal(expected, gates) and ((gates > 0).sum(1) == 2).all()
          and np.allclose(gates.sum(1), 1., rtol=0., atol=1e-5), 'Invalid hard Top2 gate shares')
    raw = z['router_port_energy']
    target_dense = (raw + 1e-12) / (raw.sum(1, keepdims=True) + 4e-12)
    check(np.allclose(z['dense_power'], target_dense, rtol=2e-5, atol=1e-6), 'Invalid pre-Top2 optical power')
    selected_dense = np.take_along_axis(z['dense_power'], slots, axis=1)
    check((selected_dense.min(1) >= np.where(gates == 0, z['dense_power'], -np.inf).max(1)).all(),
          'Selected gate below a masked proposal')
    check(np.allclose(np.take_along_axis(raw, slots, axis=1).sum(1), z['router_capture'], rtol=2e-5, atol=1e-7),
          'Capture disagrees with selected physical detector integrals')
    before, after = z['pre_fiber_power'], z['transmitted_power']
    incident = before.sum(1, keepdims=True)
    check(np.all(np.isclose(incident, 1., atol=2e-4) | (incident == 0.))
          and np.allclose(before, gates * incident, rtol=2e-4, atol=3e-6) and (after <= before + 3e-6).all(),
          'Local power exceeds routed input or passive projection increases power')
    bypass = [j for j in range(4) if j not in fiber_slots]
    check(np.allclose(before[:, bypass], after[:, bypass], rtol=2e-5, atol=1e-7), 'Bypass slot changed power')


class CollectPower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.values = {k: [] for k in DIAGNOSTIC_FIELDS}

    def forward(self, fields):
        row = self.model(fields, diagnostics=True)
        for k, source in DIAGNOSTIC_FIELDS.items():
            self.values[k].append(row[source].detach().cpu())
        return row


def final_split(model, split, inputs, config, device, out):
    collector = CollectPower(model)
    temp = out / f'{split}.optical.tmp.npz'
    packed = out / f'{split}.slots.tmp.npz'
    try:
        metric = paired.evaluate(collector, inputs[split], config['batch'], device, temp)
        with np.load(temp, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        arrays.update({k: torch.cat(v).numpy() for k, v in collector.values.items()})
        check_power(arrays, config['fiber_slots'])
        np.savez_compressed(packed, **arrays)
        packed.replace(out / f'predictions_{split}.npz')
    finally:
        temp.unlink(missing_ok=True)
        packed.unlink(missing_ok=True)
    return metric


def smoke(model, config, device):
    raw, identity = paired.load_raw(Path(config['data_root']), config)
    check(all(identity[k] == config[k] for k in ('manifest_sha256', 'index_sha256')), 'Smoke split identity differs')
    part = raw['train']
    chosen = np.array([np.flatnonzero(part['labels'] == k)[0] for k in range(config['num_classes'])])
    fields = paired.encode(torch.from_numpy(part['images'][chosen]).to(device))
    labels = torch.as_tensor(part['labels'][chosen], dtype=torch.long, device=device)
    reference = paired.build_model('fixed', config['num_classes']).to(device)
    reference.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        baseline = reference(fields)
        plain = model(fields)
        diag = model(fields, diagnostics=True)
        check(torch.equal(plain['logits'], diag['logits']), 'Diagnostics alter optical logits')
        for k in ('positions', 'power', 'router_capture'):
            check(torch.equal(plain[k], baseline[k]), 'Expert change modifies upstream optical routing')
        if config['arm'] == 'diffraction':
            check(torch.equal(plain['logits'], baseline['logits']), 'Diffraction control differs from T24')
        arrays = {k: diag[src].cpu().numpy() for k, src in DIAGNOSTIC_FIELDS.items()}
        arrays.update({k: plain[k].cpu().numpy() for k in ('positions', 'power', 'router_capture')})
        check_power(arrays, config['fiber_slots'])
    snapshot = {k: v.detach().clone() for k, v in model.named_parameters()}
    opt = optimizer_for(model, config)
    loss = F.cross_entropy(model(fields)['logits'], labels, label_smoothing=config['label_smoothing'])
    loss.backward()
    gradients = {k: p.grad.norm().item() if p.grad is not None else 0. for k, p in model.named_parameters()}
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    opt.step()
    updates = {k: (p.detach() - snapshot[k]).abs().max().item() for k, p in model.named_parameters()}
    check(torch.isfinite(loss).item() and all(np.isfinite(v) and v > 0. for v in (*gradients.values(), *updates.values())),
          'Expected finite nonzero phase/router/head gradients and actual updates')
    return {'domain': 'original TRAIN only', 'ids': part['ids'][chosen].astype(str).tolist(),
            'labels': labels.cpu().tolist(), 'loss': loss.item(), 'gradient_l2': gradients, 'update_max_abs': updates,
            'diffraction_control_logits_bitwise': (True if config['arm'] == 'diffraction' else None),
            'shared_router_bitwise': True, 'diagnostic_logits_bitwise': True, 'data': identity,
            'slots_receipt': {k: value.tolist() for k, value in arrays.items()}}


def run(args):
    out, device = args.out, torch.device(args.device)
    config = configuration(args.dataset, args.seed, args.arm, args.epochs)
    gpu = paired.require_authorized_gpu(config['authorized_gpu_uuids']) if device.type == 'cuda' else None
    check(gpu is None or gpu == config['assigned_gpu_uuid'], 'Seed is assigned to the other cohort GPU')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model = make_model(config)
    check(list(model.fiber_core_counts) == config['fiber_core_counts'], 'Runtime expert layout differs from protocol')
    config_id = paired.state_digest(model)
    check(len(model.state_dict()) == 4 and sum(p.numel() for p in model.parameters()) == 513416, 'Optical contract differs')
    sources = paired.source_files()
    commit = os.environ.get('T25_SOURCE_COMMIT') or subprocess.check_output(
        ['git', '-C', str(paired.REPO), 'rev-parse', 'HEAD'], text=True).strip()
    meta = {'arm': args.arm, 'mode': args.arm, 'seed': args.seed, 'fiber_slots': config['fiber_slots'],
            'fiber_core_counts': config['fiber_core_counts'], 'protocol_sha256': config['protocol_sha256'],
            'config': config, 'dataset': config['dataset'], 'source_commit': commit, 'source_sha256': sources,
            'initial_state_sha256': config_id, 'parameter_count': 513416, 'device': str(device),
            'gpu_uuid': gpu, 'gpu_model': torch.cuda.get_device_name(device) if gpu else None, 'pid': os.getpid(),
            'python': platform.python_version(), 'cuda': torch.version.cuda, 'torch': str(torch.__version__),
            'command': [sys.executable, '-m', __spec__.name, *sys.argv[1:]],
            'optimizer': {'name': 'Adam', 'phase_lr': config['phase_lr'], 'head_lr': config['head_lr'],
                          'gradient_clip_norm': 5., 'weight_decay': 0., 'schedule': 'constant'},
            'selection': 'highest original VAL correct count, then lowest VAL CE; earlier exact tie',
            'TEST_use': 'end-of-epoch diagnostics and post-lock evaluation only; original TEST previously used by T17',
            'smoke': args.smoke}
    meta_file = out / 'metadata.json'
    if meta_file.exists():
        old = read(meta_file)
        for k in ('arm', 'config', 'source_sha256', 'initial_state_sha256', 'gpu_uuid', 'device', 'torch', 'cuda',
                  'python', 'optimizer', 'selection', 'smoke'):
            check(old[k] == meta[k], f'Resumed task identity differs: {k}')
        # The exact executing tree is SHA-bound even if unrelated tasks subsequently advance main.
        meta['source_commit'] = commit = old['source_commit']
    else:
        check(not any(p.name != '.writer.lock' for p in out.iterdir()), 'Output contains unrelated files')
    paired.write(meta_file, meta)
    paired.write(out / 'config.json', config)
    meta['config_file_sha256'] = paired.sha256(out / 'config.json')
    paired.write(meta_file, meta)
    paired.write(out / 'status.json', {'state': 'initializing', 'pid': os.getpid()})
    model.to(device)
    if args.smoke:
        evidence = smoke(model, config, device)
        paired.write(out / 'admission.json', {**evidence, 'arm': args.arm, 'source_commit': commit,
                                             'initial_state_sha256': config_id, 'gpu_uuid': gpu,
                                             'fiber_core_counts': config['fiber_core_counts']})
        paired.write(out / 'manifest.json', {n: paired.sha256(out / n) for n in ('config.json', 'metadata.json', 'admission.json')})
        paired.write(out / 'status.json', {'state': 'complete', 'smoke': True, 'pid': os.getpid()})
        return
    inputs, identity = paired.load_inputs(config, device)
    meta['data'] = identity
    paired.write(meta_file, meta)
    opt = optimizer_for(model, config)
    last = out / 'last_checkpoint.pt'
    best = None
    history = []
    rng = np.random.default_rng(args.seed)
    if last.exists():
        state = load_pt(last)
        check(state['config'] == config and state['arm'] == args.arm and state['source_commit'] == commit,
              'Last checkpoint identity differs')
        history, best = state['history'], state['selected_best']
        check(len(history) <= args.epochs and best['epoch'] == max(history, key=lambda r: key(r['validation']))['epoch']
              and best['validation'] == history[best['epoch']-1]['validation'], 'Invalid best or training length')
        rng = verify_training_state(state, history, config, model, opt)
        if gpu:
            torch.cuda.set_rng_state(state['cuda_rng_state'], device)
        materialize_best(out, best)
        paired.write(out / 'history.json', history)
        del state
    for epoch in range(len(history) + 1, args.epochs + 1):
        start = time.monotonic()
        row = train_epoch(model, opt, inputs, config, rng, device)
        val = paired.evaluate(model, inputs['val'], config['batch'], device)
        snapshot = {'model': model.state_dict(), 'epoch': epoch, 'arm': args.arm, 'mode': args.arm,
                    'config': config, 'validation': val, 'source_commit': commit}
        improved = best is None or key(val) > key(best['validation'])
        if improved:
            best = {**snapshot, 'model': {k: v.detach().cpu().clone() for k, v in snapshot['model'].items()}}
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if gpu else []):
            measured = {s: metrics(paired.evaluate(model, inputs[s], config['batch'], device)) for s in ('train', 'test')}
        row.update(epoch=epoch, validation=val, **measured, elapsed_seconds=time.monotonic()-start)
        history.append(row)
        payload = {**snapshot, 'history': history, 'selected_best': best, 'optimizer': opt.state_dict(),
                   'shuffle_rng_state': rng.bit_generator.state, 'torch_rng_state': torch.get_rng_state()}
        if gpu:
            payload['cuda_rng_state'] = torch.cuda.get_rng_state(device)
        paired.save_checkpoint(last, payload)
        if improved:
            paired.save_checkpoint(out / 'best_checkpoint.pt', best)
        paired.write(out / 'history.json', history)
        paired.write(out / 'status.json', {'state': 'training', 'epoch': epoch, 'selected_epoch': best['epoch'],
                                         'target_epoch': args.epochs, 'pid': os.getpid()})
        print(f"epoch={epoch} val={val['correct']}/{val['n']} best={best['epoch']}", flush=True)
    model.load_state_dict(best['model'], strict=True)
    lock = {'arm': args.arm, 'mode': args.arm, 'seed': args.seed, 'source_commit': commit, 'selected_epoch': best['epoch'],
            'best_checkpoint_sha256': paired.sha256(out / 'best_checkpoint.pt'), 'TEST_selection_calls': 0,
            'target_epoch': args.epochs}
    paired.write(out / 'selection_lock.json', lock)
    result = {**lock, 'diagnostic_TEST_calls': args.epochs}
    for split in SPLITS:
        result[split] = final_split(model, split, inputs, config, device, out)
        row = history[best['epoch']-1]['validation' if split == 'val' else split]
        check(result[split]['correct'] == row['correct'], f'{split} differs from selected epoch')
    paired.write(out / 'result.json', result)
    names = ('config.json', 'metadata.json', 'history.json', 'selection_lock.json', 'result.json',
             'best_checkpoint.pt', 'last_checkpoint.pt', *(f'predictions_{s}.npz' for s in SPLITS))
    paired.write(out / 'manifest.json', {k: paired.sha256(out / k) for k in names})
    paired.write(out / 'status.json', {'state': 'complete', 'epoch': args.epochs, 'selected_epoch': best['epoch'],
                                     'target_epoch': args.epochs, 'pid': os.getpid()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATA_CONFIGS, required=True)
    p.add_argument('--arm', choices=read(PROTOCOL)['arms'], required=True)
    p.add_argument('--seed', type=int, choices=(37, 53), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=180)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.out = args.out.resolve()
    base = TASK / 'runs' / ('smoke' if args.smoke else 'simulation')
    if not args.out.is_relative_to(base) or args.out == base or args.epochs <= 0:
        p.error('Use a separate task-local simulation or smoke run and a positive epoch count')
    args.out.mkdir(parents=True, exist_ok=True)
    with writer(args.out):
        run(args)


if __name__ == '__main__':
    main()
