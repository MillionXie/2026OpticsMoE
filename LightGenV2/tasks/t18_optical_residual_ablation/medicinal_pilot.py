"""Bounded natural-image feasibility pilot; train/val only, never reads test arrays."""
import argparse
import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import fast_optical as execution

o, torch = execution.optical, execution.torch
np, r = o.np, o.r


def load_fit_data(path, split, manifest):
    assert split in ('train', 'val'), 'Pilot must never materialize test'
    with np.load(path, allow_pickle=False) as z:
        images = z[split + '_images'].copy()
        labels = z[split + '_labels'].copy()
        ids = z[split + '_ids'].copy()
    assert images.dtype == np.uint8 and images.shape == (len(labels), 150, 150, 3)
    assert np.bincount(labels, minlength=len(manifest['classes'])).tolist() == manifest['supports'][split]
    assert ids.tolist() == manifest['split_ids'][split]
    return torch.from_numpy(images.transpose(0, 3, 1, 2).copy()), torch.from_numpy(labels).long(), ids


@torch.no_grad()
def evaluate(model, data, batch):
    model.eval()
    probabilities, captures = [], []
    for idx in torch.arange(len(data[1])).split(batch):
        p, c = o.forward(model, o.encode(data[0][idx]))
        probabilities.append(p.cpu().numpy()); captures.append(c.cpu().numpy())
    probabilities = np.concatenate(probabilities).astype(np.float64)
    result = o.metrics(data[1].numpy(), probabilities)
    result['detector_capture'] = float(np.concatenate(captures).mean())
    rows = [dict(sample_id=str(sid), label_true=int(label), label_pred=int(p.argmax()),
        **{f'score{k}': float(p[k]) for k in range(len(p))})
        for sid, label, p in zip(data[2], data[1], probabilities)]
    return result, rows


def numerical_gate(data, weights, profile, cfg, batched, micro):
    """Fresh same initialized parameters: serial micro1 vs proposed production execution."""
    r.setseed(cfg['seed'])
    model = execution.build(2, profile, cfg, batched=False)
    identity = o.f.tensor_identity(model)
    def compute(batch):
        model.zero_grad(set_to_none=True)
        probabilities = []
        for start in range(0, 8, batch):
            end = min(start + batch, 8)
            p, c = o.forward(model, o.encode(data[0][start:end]))
            (o.objective(model, p, c, data[1][start:end].cuda(), weights, cfg) * (end-start)/8).backward()
            probabilities.append(p.detach())
        return torch.cat(probabilities), {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    reference, gradients = compute(1)
    if batched:
        execution.enable_expert_batching(model)
    assert identity == o.f.tensor_identity(model)
    assert not any(hasattr(m, 'residual_rho') for n, m in model.named_modules() if 'router' in n)
    result, proposed = compute(micro)
    error = float((reference-result).abs().max())
    relative = max(float((proposed[n]-g).norm()/g.norm().clamp_min(1e-12)) for n, g in gradients.items())
    cosine = min(float(torch.nn.functional.cosine_similarity(proposed[n].flatten(), g.flatten(), dim=0)) for n, g in gradients.items())
    finite = all(bool(torch.isfinite(g).all()) and float(g.norm()) > 0 for g in proposed.values())
    assert error <= 1e-6 and relative <= .02 and cosine >= .999 and finite
    receipt = dict(passed=True, probability_max_error=error, gradient_max_relative_l2=relative,
        gradient_min_cosine=cosine, gradients_finite_nonzero=finite, initialization_unchanged=True,
        router_has_residual=False, initial_parameter_sha256=identity, residual_rho=cfg['residual_rho'],
        expert_batched=batched, microbatch=micro, scope='Fresh seed17 first8 train only; serial micro1 reference')
    del model, reference, gradients, result, proposed
    gc.collect(); torch.cuda.empty_cache()
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--gpu', required=True); p.add_argument('--rho', type=float, choices=[0., .3], required=True)
    p.add_argument('--micro', type=int, choices=[1, 2, 4], required=True)
    p.add_argument('--batched', type=int, choices=[0, 1], required=True)
    p.add_argument('--probe', type=Path, required=True)
    a = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == a.gpu
    a.out.mkdir(parents=True, exist_ok=False)
    profile_path = o.HERE/'medicinal_pilot_profile.json'
    profile = r.read(profile_path)
    gate = r.read(a.probe)
    for rho in [0., .3]:
        assert any(x['depth'] == 2 and x['rho'] == rho and x['expert_batched'] == bool(a.batched)
            and x['microbatch'] == a.micro and x['passed'] for x in gate['results'])
    o.f.gpu_budget.ALLOWED.update(profile['gpu_uuids'])
    gpu_policy = o.f.gpu_budget.configure(torch, a.gpu, shared=True, memory_gib=3)
    torch.set_num_threads(4)
    manifest = r.read(a.data.parent/'data_manifest.json')
    assert manifest['total_records'] == 1835 and len(manifest['classes']) == 30
    assert manifest['cache_sha256'] == r.sha(a.data)
    assert manifest['preparation_source_sha256'] == r.sha(o.HERE/'prepare_medicinal.py')
    assert manifest['grouping_source_sha256'] == r.sha(o.HERE/'prepare_butterfly.py')
    assert manifest['author_index_sha256'] == '922f45590782b91149e24ce2a003b57d1117de8b563d8ace8534a7bee8351fa7'
    assert profile['detector'] == execution.detector(30)
    profile.update(dataset=manifest['dataset'], microbatch=a.micro, evaluation_batch_size=a.micro,
        expert_batched=bool(a.batched), maximum_epochs=30, minimum_epochs=10, patience=6,
        maximum_seconds=1800, scope='Small full30 natural leaf image train/val feasibility pilot, image-level only')
    cfg = o.config(profile, manifest, a.rho, 'base')
    for key in ['initial_epochs', 'maximum_base_epochs', 'maximum_paired_restarts', 'restart_epochs', 'restart_candidates', 'data_revision']:
        cfg.pop(key, None)
    cfg['selection'] = 'EMA minimum validation balanced NLL, first exact tie; pilot never tests'
    sources = o.source_identity(profile_path)
    sources.update({str((o.HERE/name).relative_to(o.ROOT)).replace('\\', '/'): r.sha(o.HERE/name)
        for name in ['fast_optical.py', 'medicinal_pilot.py', 'prepare_medicinal.py', 'probe_fast_optical.py']})
    assert all(hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:'+path])).hexdigest() == digest
        for path, digest in sources.items()), 'Every runtime source must match its published Git blob'
    r.save(a.out/'config.json', cfg)
    r.save(a.out/'metadata.json', dict(pid=os.getpid(), command=sys.argv, sources=sources, gpu_policy=gpu_policy,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        execution_probe_sha256=r.sha(a.probe), manifest_sha256=r.sha(a.data.parent/'data_manifest.json'), time=r.now()))
    data, val = [load_fit_data(a.data, split, manifest) for split in ('train', 'val')]
    weights = (len(data[1])/(30*torch.bincount(data[1], minlength=30).float())).cuda()
    smoke = numerical_gate(data, weights, profile, cfg, bool(a.batched), a.micro)
    r.save(a.out/'smoke.json', smoke)
    r.setseed(cfg['seed'])
    model = execution.build(2, profile, cfg, batched=bool(a.batched))
    initial_sha = o.f.tensor_identity(model)
    assert initial_sha == smoke['initial_parameter_sha256']
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', **cfg['scheduler'])
    history, gradients, orders, transforms = [], [], [], []
    best, stale, reason = float('inf'), 0, 'maximum_epochs'
    started = time.monotonic()
    for epoch in range(1, 31):
        if time.monotonic()-started >= 1800:
            reason = 'time_budget'; break
        epoch_started = time.monotonic()
        model.train()
        order = r.epoch_order(cfg['seed'], epoch, len(data[1]))
        theta = o.b.affine_parameters(len(data[1]), cfg['seed'], epoch, cfg['augmentation'])
        orders.append(r.sha_tensor(order)); transforms.append(r.sha_tensor(theta))
        total = 0.
        for batch_index, idx in enumerate(order.split(16)):
            optimizer.zero_grad(set_to_none=True)
            for micro in idx.split(a.micro):
                probability, capture = o.forward(model, o.encode(data[0][micro], theta[micro]))
                loss = o.objective(model, probability, capture, data[1][micro].cuda(), weights, cfg)
                assert bool(torch.isfinite(loss))
                (loss*len(micro)/len(idx)).backward()
                total += float(loss.detach())*len(micro)
            if batch_index == 0:
                norms = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()}
                assert all(np.isfinite(v) and v > 0 for v in norms.values())
                gradients.append(dict(epoch=epoch, norms=norms))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            with torch.no_grad():
                for ep, mp in zip(ema.parameters(), model.parameters()): ep.lerp_(mp, 1-cfg['ema_decay'])
        vm, rows = evaluate(ema, val, a.micro)
        entry = dict(epoch=epoch, train_loss=total/len(data[1]), lr=optimizer.param_groups[0]['lr'], val=vm)
        if epoch == 1 or epoch % 5 == 0:
            entry['train'] = evaluate(ema, data, a.micro)[0]
        if vm['balanced_nll'] < best:
            best, stale = vm['balanced_nll'], 0
            r.save_torch(a.out/'best_checkpoint.pt', dict(model=ema.state_dict(), epoch=epoch, depth=2,
                config=cfg, sources=sources, data_sha256=manifest['cache_sha256'], validation=vm,
                selected_state_kind='ema', selection_split='val'))
            r.csvwrite(a.out/'val_predictions.csv', rows)
        else: stale += 1
        scheduler.step(vm['balanced_nll'])
        entry.update(seconds=time.monotonic()-started, epoch_seconds=time.monotonic()-epoch_started)
        history.append(entry)
        r.save(a.out/'history.json', history); r.save(a.out/'gradients.json', gradients)
        r.save_torch(a.out/'last_checkpoint.pt', dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch, config=cfg,
            sources=sources, orders=orders, transforms=transforms, initial_parameter_sha256=initial_sha,
            data_sha256=manifest['cache_sha256']))
        status = dict(state='training', depth=2, rho=a.rho, epoch=epoch, seconds=entry['seconds'],
            epoch_seconds=entry['epoch_seconds'], val_accuracy=vm['accuracy'], val_macro_recall=vm['balanced_accuracy'],
            val_balanced_nll=vm['balanced_nll'], test_evaluations=0)
        r.save(a.out/'status.json', status); print(json.dumps(status), flush=True)
        if epoch >= 10 and stale >= 6:
            reason = 'validation_plateau'; break
    if 'train' not in history[-1]:
        history[-1]['train'] = evaluate(ema, data, a.micro)[0]
        r.save(a.out/'history.json', history)
    checkpoint = torch.load(a.out/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    selected = min(history, key=lambda h: h['val']['balanced_nll'])
    assert selected['epoch'] == checkpoint['epoch'] and selected['val'] == checkpoint['validation']
    assert checkpoint['config'] == cfg and checkpoint['sources'] == sources
    ema.load_state_dict(checkpoint['model'])
    tm, rows = evaluate(ema, data, a.micro)
    r.csvwrite(a.out/'train_predictions.csv', rows)
    result = dict(depth=2, rho=a.rho, metrics=dict(train=tm, val=checkpoint['validation']),
        selected_epoch=checkpoint['epoch'], epochs_completed=len(history), stop_reason=reason,
        seconds=time.monotonic()-started, history_sha256=r.sha(a.out/'history.json'),
        checkpoint_sha256=r.sha(a.out/'best_checkpoint.pt'), sources=sources, config=cfg,
        data_sha256=manifest['cache_sha256'], initial_parameter_sha256=initial_sha,
        orders=orders, transforms=transforms, parameters=sum(p.numel() for p in model.parameters()),
        test_evaluations=0, test_used_for_selection=False, selection_split='val',
        accuracy_gate_passed=checkpoint['validation']['accuracy'] >= .7 and checkpoint['validation']['balanced_accuracy'] >= .7,
        speed_gate_passed=float(np.mean([h['epoch_seconds'] for h in history])) <= 60,
        scope='Feasibility pilot, adaptive validation reuse, not a final independent test or residual comparison')
    r.save(a.out/'fit_result.json', result)
    r.save(a.out/'status.json', dict(state='fit_complete', stop_reason=reason, epochs=len(history), test_evaluations=0, time=r.now()))


if __name__ == '__main__': main()
