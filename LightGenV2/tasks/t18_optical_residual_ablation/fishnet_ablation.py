"""Single-GPU paired FishNet adapter; reuse the unchanged T18 optical model."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import run as t

HERE = Path(__file__).resolve().parent
b, r, torch, np = t.b, t.r, t.torch, t.np


def source_identity(profile):
    return dict(optical_parent=t.source_identity(), adapter=r.sha(__file__),
                preparation=r.sha(HERE / 'prepare_fishnet.py'), profile=r.sha(profile))


def preflight(gpu):
    assert gpu == 'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d'
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    assert gpu not in [x.strip() for x in occupied.splitlines()], 'Requested physical GPU is occupied'
    info = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = int(next(x.split(',')[1] for x in info.splitlines() if x.startswith(gpu)))
    assert free >= 22000, 'Insufficient memory for full batch16'
    return dict(gpu_uuid=gpu, exclusive_at_start=True, free_mib_at_start=free)


def config(profile, manifest, rho):
    cfg = t.config(rho)
    for key in ['dataset', 'license', 'epochs', 'batch_size', 'lr', 'ema_decay', 'label_smoothing',
                'capture_weight', 'phase_smooth_weight', 'phase_init_raw_uniform', 'augmentation',
                'selection', 'scope', 'test_development', 'training_reads_test_for_gradients']:
        cfg[key] = profile[key]
    cfg.update(classes=manifest['classes'], deduplication=manifest['split_policy'],
               training_profile='fishnet_common100_testdev', minimum_epochs=100, patience=100,
               source=profile['source'], doi=profile['doi'])
    return cfg


def tensor_identity(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train(a, profile, manifest, cfg, src):
    data, val, test = [t.k.load_data(a.data, split) for split in ['train', 'val', 'test']]
    r.setseed(profile['seed'])
    model = t.build('moe', a.depth, cfg)
    initial_sha = tensor_identity(model)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 10000 + 439848 * (a.depth // 2)
    dest = a.out / f'moe_L{a.depth}_seed{profile["seed"]}'
    dest.mkdir()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'], eta_min=cfg['lr'] * .1)
    weights = len(data[1]) / (8 * torch.bincount(data[1], minlength=8).float())
    history, gradients, orders, transforms = [], [], [], []
    best, chosen_test = (-1., -float('inf')), None
    started = time.time()
    r.save(dest / 'initial_validation.json', b.evaluate(model, val, 'moe', cfg['batch_size'])[0])
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        order = r.epoch_order(profile['seed'], epoch, len(data[1]))
        theta = b.affine_parameters(len(data[1]), profile['seed'], epoch, cfg['augmentation'])
        orders.append(r.sha_tensor(order)); transforms.append(r.sha_tensor(theta))
        total = 0.
        for batch_index, idx in enumerate(order.split(cfg['batch_size'])):
            y = data[1][idx]
            x = b.encode(data[0][idx], theta[idx])
            opt.zero_grad(set_to_none=True)
            prob, capture, _ = b.forward(model, x, 'moe')
            target = torch.nn.functional.one_hot(y, 8) * (1 - cfg['label_smoothing']) + cfg['label_smoothing'] / 8
            loss = (-(target * prob.clamp_min(1e-12).log()).sum(1) * weights[y]).mean()
            loss = loss - cfg['capture_weight'] * capture.clamp_min(1e-12).log().mean() + cfg['phase_smooth_weight'] * b.phase_smoothness(model)
            assert torch.isfinite(loss)
            loss.backward()
            if batch_index == 0:
                norms = {name: float(p.grad.norm()) for name, p in model.named_parameters()}
                assert all(np.isfinite(v) and v > 0 for v in norms.values())
                gradients.append(dict(epoch=epoch, norms=norms))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            with torch.no_grad():
                for ep, mp in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(mp, 1 - cfg['ema_decay'])
            total += float(loss.detach()) * len(idx)
        vm = b.evaluate(ema, val, 'moe', cfg['batch_size'])[0]
        entry = dict(epoch=epoch, train_loss=total / len(data[1]), lr=opt.param_groups[0]['lr'], val=vm)
        if epoch == 1 or epoch % 5 == 0:
            entry['train'] = b.evaluate(ema, data, 'moe', cfg['batch_size'])[0]
        for kind, candidate in [('raw', model), ('ema', ema)]:
            metric, rows = b.evaluate(candidate, test, 'moe', cfg['batch_size'])
            entry['test_development_' + kind] = metric
            score = (metric['accuracy'], -metric['balanced_nll'])
            if score > best:
                best, chosen_test = score, metric
                r.save_torch(dest / 'best_checkpoint.pt', dict(model=candidate.state_dict(), epoch=epoch,
                    arch='moe', depth=a.depth, seed=profile['seed'], config=cfg, sources=src,
                    selected_state_kind=kind, data_sha256=manifest['cache_sha256']))
                r.csvwrite(dest / 'test_predictions.csv', rows)
        history.append(entry)
        scheduler.step()
        r.save(dest / 'history.json', history); r.save(dest / 'gradients.json', gradients)
        r.save_torch(dest / 'last_checkpoint.pt', dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch, arch='moe',
            depth=a.depth, seed=profile['seed'], config=cfg, sources=src, data_sha256=manifest['cache_sha256']))
        status = dict(state='training', depth=a.depth, rho=a.rho, epoch=epoch, epochs=cfg['epochs'],
                      val_accuracy=vm['accuracy'], test_raw=entry['test_development_raw']['accuracy'],
                      test_ema=entry['test_development_ema']['accuracy'], best_test_development=best[0], seconds=time.time() - started)
        r.save(a.out / 'status.json', status); print(json.dumps(status), flush=True)
    ck = torch.load(dest / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    assert ck['config'] == cfg and ck['sources'] == src and ck['data_sha256'] == r.sha(a.data)
    ema.load_state_dict(ck['model'])
    measures = {}
    for split, examples in [('train', data), ('val', val)]:
        measures[split], rows = b.evaluate(ema, examples, 'moe', cfg['batch_size'])
        r.csvwrite(dest / (split + '_predictions.csv'), rows)
    result = dict(depth=a.depth, rho=a.rho, parameters=parameters, initial_parameter_sha256=initial_sha,
                  selected_epoch=ck['epoch'], selected_state_kind=ck['selected_state_kind'], epochs_completed=epoch,
                  metrics=measures, test_development=chosen_test, orders=orders, transforms=transforms,
                  checkpoint_sha256=r.sha(dest / 'best_checkpoint.pt'), checkpoint=str(dest / 'best_checkpoint.pt'),
                  data_sha256=manifest['cache_sha256'], seconds=time.time() - started, scope=cfg['scope'])
    r.save(a.out / 'result.json', result)
    del model, ema, opt, ck, data, val, test
    gc.collect(); torch.cuda.empty_cache()
    r.save(a.out / 'status.json', dict(state='complete', time=r.now(), test_development=True, gpu_released_on_exit=True))


def smoke(a, profile, cfg):
    r.setseed(profile['seed'])
    model = t.original_build('moe', a.depth, cfg)
    model.net.expert_bank.vectorize_homogeneous_d2nn = False
    x = b.encode(t.k.load_data(a.data, 'train')[0][:2])
    initial = tensor_identity(model)
    with torch.no_grad():
        reference = b.forward(model, x, 'moe')[0]
    t.install_residual(model, t.PhaseLayer, 0.)
    with torch.no_grad():
        zero = b.forward(model, x, 'moe')[0]
    assert torch.equal(reference, zero) and tensor_identity(model) == initial
    targets = t.install_residual(model, t.PhaseLayer, .3)
    prob, capture, _ = b.forward(model, x, 'moe')
    (-prob[:, 0].clamp_min(1e-12).log().mean()).backward()
    norms = {name: float(p.grad.norm()) for name, p in model.named_parameters()}
    assert len(targets) == 5 * a.depth and all(np.isfinite(v) and v > 0 for v in norms.values())
    assert all('router' not in name for name in targets) and prob.shape == (2, 8)
    r.save(a.out / 'smoke.json', dict(passed=True, rho0_exact_identity=True, initialization_unchanged=True,
        router_no_residual=True, input_shape=list(x.shape), output_shape=list(prob.shape), gradients=norms))
    r.save(a.out / 'status.json', dict(state='complete', phase='smoke', time=r.now()))


def suite(a, profile):
    a.out.mkdir(parents=True, exist_ok=False)
    r.save(a.out / 'metadata.json', dict(command=sys.argv, pid=os.getpid(), profile=profile,
        sources=source_identity(a.profile), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        gpu_uuid=a.gpu, time=r.now()))
    child = None
    try:
        data = a.data_root / 'fishnet_fixed_split.npz'
        if not data.exists():
            r.save(a.out / 'status.json', dict(state='preparing_data', gpu_processes=0))
            with (a.out / 'preparation.log').open('w') as log:
                subprocess.run([sys.executable, '-u', str(HERE / 'prepare_fishnet.py'), '--out', str(a.data_root),
                                '--profile', str(a.profile)], stdout=log, stderr=subprocess.STDOUT, check=True)
        jobs = [('smoke', 6, .3)] + [('train', depth, rho) for depth in profile['depth_order'] for rho in profile['rhos']]
        results = []
        for phase, depth, rho in jobs:
            preflight(a.gpu)
            folder = a.out / ('smoke' if phase == 'smoke' else f'L{depth}_rho{rho}')
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--phase', phase, '--depth', str(depth),
                   '--rho', str(rho), '--data', str(data), '--profile', str(a.profile), '--out', str(folder), '--gpu', a.gpu]
            with (a.out / (folder.name + '.log')).open('w') as log:
                child = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu), stdout=log, stderr=subprocess.STDOUT)
                r.save(a.out / 'current_process.json', dict(pid=child.pid, command=cmd, gpu_uuid=a.gpu, phase=phase, depth=depth, rho=rho))
                r.save(a.out / 'status.json', dict(state='running', phase=phase, depth=depth, rho=rho, completed=len(results), total=6, child_pid=child.pid))
                assert child.wait() == 0, f'{folder.name} failed; preserve log'
            if phase == 'train':
                results.append(r.read(folder / 'result.json'))
                r.save(a.out / 'results.json', results)
                if rho == .3:
                    left, right = results[-2:]
                    for key in ['depth', 'parameters', 'initial_parameter_sha256', 'orders', 'transforms', 'data_sha256', 'epochs_completed']:
                        assert left[key] == right[key], ('Pair mismatch', depth, key)
                    r.save(a.out / f'pair_lock_L{depth}.json', dict(depth=depth, paired_initialization=True, paired_orders=True,
                        paired_transforms=True, same100_epochs=True, checkpoints=[x['checkpoint_sha256'] for x in [left, right]],
                        selection=profile['selection'], test_development=True))
        r.save(a.out / 'status.json', dict(state='complete', completed=6, gpu_released_on_exit=True, time=r.now()))
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate(); child.wait()
        r.save(a.out / 'status.json', dict(state='failed', traceback=traceback.format_exc(), time=r.now()))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['suite', 'smoke', 'train'], required=True)
    p.add_argument('--profile', type=Path, default=HERE / 'fishnet_profile.json')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--depth', type=int, choices=[2, 4, 6], default=6)
    p.add_argument('--rho', type=float, choices=[0., .3], default=.3)
    p.add_argument('--gpu', required=True)
    a = p.parse_args(); profile = r.read(a.profile)
    assert a.gpu == profile['gpu_uuid'] and profile['test_development'] and profile['epochs'] == 100
    if a.phase == 'suite':
        assert a.data_root
        suite(a, profile)
        return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == a.gpu and a.data
    policy = preflight(a.gpu)
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    manifest = r.read(a.data.parent / 'data_manifest.json')
    assert manifest['dataset'] == profile['dataset'] and manifest['license'] == 'CC BY 4.0'
    assert manifest['classes'] == profile['classes'] and manifest['profile_sha256'] == r.sha(a.profile)
    cfg, src = config(profile, manifest, a.rho), source_identity(a.profile)
    r.save(a.out / 'metadata.json', dict(config=cfg, sources=src, command=sys.argv, pid=os.getpid(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        data_sha256=r.sha(a.data), manifest_sha256=r.sha(a.data.parent / 'data_manifest.json'),
        gpu_policy=policy, environment=t.m.environment(), time=r.now(),
        test_development=True, training_reads_test_for_gradients=False))
    try:
        if a.phase == 'smoke':
            smoke(a, profile, cfg)
        else:
            train(a, profile, manifest, cfg, src)
    except BaseException:
        r.save(a.out / 'status.json', dict(state='failed', traceback=traceback.format_exc(), time=r.now()))
        raise


if __name__ == '__main__':
    main()
