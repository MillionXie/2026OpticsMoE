"""Paired regularization experiment, preserving the archived model and objective.

Train never reads test arrays. Test requires a completed checkpoint/threshold lock.
All transformations are generated on CPU by a separate epoch RNG, then shared by
sample identity across architectures/depths. They are never applied at evaluation.
"""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
TASK = Path(__file__).resolve().parents[1]
ARCHIVE = TASK / 'adrenal_softsign_code_export_20260915_145336/code'
sys.path.insert(0, str(ARCHIVE))
import numpy as np
import torch
from torch.nn import functional as F
import run_experiment as r
from calibration import choose_thresholds, full_metrics
from optical_reference.optics import PhaseLayer


def affine_parameters(n, seed, epoch, cfg):
    g = torch.Generator().manual_seed(seed * 1_000_003 + epoch + 71_000_000)
    u = 2 * torch.rand(n, 4, generator=g) - 1
    angle = u[:, 0] * cfg['degrees'] * math.pi / 180
    scale = 1 + u[:, 1] * cfg['scale_delta']
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = theta[:, 1, 1] = angle.cos() / scale
    theta[:, 0, 1] = -angle.sin() / scale
    theta[:, 1, 0] = angle.sin() / scale
    # Output-to-input sampling coordinates, normalized by the 100-pixel input.
    theta[:, :, 2] = u[:, 2:] * (2 * cfg['translation_pixels'] / 100)
    return theta


@torch.no_grad()
def augment(x, theta):
    grid = F.affine_grid(theta.to(x.device), x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def phase_smoothness(model):
    # Each spatial neighbor edge has equal weight. Neither extra layers nor more
    # expert tensors automatically multiply the regularizer's scale. Circular
    # differences respect the equivalence of phase zero and phase 2*pi.
    total = None
    edges = 0
    for module in model.modules():
        if isinstance(module, PhaseLayer):
            phase = module.get_phase()
            dy, dx = phase[1:] - phase[:-1], phase[:, 1:] - phase[:, :-1]
            value = (1 - dy.cos()).sum() + (1 - dx.cos()).sum()
            total = value if total is None else total + value
            edges += dy.numel() + dx.numel()
    assert total is not None and edges > 0
    return total / edges


def clean_evaluate(model, data, dest, prefix):
    metrics, rows = r.evaluate(model, data, predictions=True)
    r.save(dest / (prefix + '_metrics.json'), metrics)
    r.csvwrite(dest / (prefix + '_predictions.csv'), rows)
    return metrics, rows


def train_variant(v, seed, train, val, protocol, out, source_hashes):
    dest = out / 'runs' / v['id'] / ('seed' + str(seed))
    dest.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(r.CONFIGS[v['id']])
    r.save(dest / 'config.json', cfg)
    r.setseed(seed)
    model = r.build(v['architecture'], cfg).cuda()
    initial = {n: r.sha_tensor(p) for n, p in model.named_parameters()}
    assert all(n.endswith('raw_phase') for n in initial)
    r.save(dest / 'initialization.json', initial)
    opt = torch.optim.Adam(model.parameters(), lr=protocol['lr'], weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=protocol['step_size'], gamma=protocol['gamma'])
    history, orders, transforms = [], [], []
    best_auc, best_mse = -1., float('inf')
    started = time.perf_counter()
    for epoch in range(1, protocol['epochs'] + 1):
        model.train()
        order = r.epoch_order(seed, epoch, len(train[1]))
        theta = affine_parameters(len(train[1]), seed, epoch, protocol['augmentation'])
        orders.append(r.sha_tensor(order))
        transforms.append(r.sha_tensor(theta))
        sums = np.zeros(3)
        for x, y, indices in r.batches(train, order):
            opt.zero_grad(set_to_none=True)
            prediction = model(augment(x, theta[indices]))
            mse = r.objective(prediction, y, model.masks)
            smooth = phase_smoothness(model)
            loss = mse + protocol['phase_smooth_weight'] * smooth
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
            sums += np.array([loss.item(), mse.item(), smooth.item()]) * len(y)
        vm, _ = r.evaluate(model, val)
        row = dict(epoch=epoch, train_total_loss=sums[0]/len(train[1]),
                   train_augmented_mse=sums[1]/len(train[1]), train_phase_smoothness=sums[2]/len(train[1]),
                   val_auroc=vm['auroc'], val_detector_plane_mse=vm['detector_plane_mse'],
                   lr=opt.param_groups[0]['lr'])
        history.append(row)
        r.csvwrite(dest / 'history.csv', history)
        if r.better(vm['auroc'], vm['detector_plane_mse'], best_auc, best_mse):
            best_auc, best_mse = vm['auroc'], vm['detector_plane_mse']
            r.save_torch(dest / 'best_checkpoint.pt', dict(model=model.state_dict(), epoch=epoch,
                         validation=vm, variant=v, seed=seed, source_hashes=source_hashes))
        scheduler.step()
        r.save_torch(dest / 'last_checkpoint.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(),
                     scheduler=scheduler.state_dict(), epoch=epoch, variant=v, seed=seed,
                     source_hashes=source_hashes, history=history, order_sha256=orders,
                     transform_sha256=transforms))
        r.save(out / 'status.json', dict(state='training', variant=v['id'], seed=seed,
               best_val_auroc=best_auc, **row))
        print(json.dumps(dict(variant=v['id'], seed=seed, **row)), flush=True)
    changed = {n: initial[n] != r.sha_tensor(p) for n, p in model.named_parameters()}
    assert all(changed.values())
    last_train, _ = clean_evaluate(model, train, dest, 'last_train')
    ck = torch.load(dest / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    tm, _ = clean_evaluate(model, train, dest, 'selected_train')
    vm, rows = clean_evaluate(model, val, dest, 'selected_val')
    y = np.array([row['label_true'] for row in rows])
    p = np.array([[row['score0'], row['score1']] for row in rows])
    threshold = choose_thresholds(y, p)
    r.save(dest / 'thresholds.json', threshold)
    result = dict(variant=v['id'], seed=seed, selected_epoch=ck['epoch'], train=tm, val=vm,
                  last_train=last_train, seconds=time.perf_counter()-started,
                  parameters=sum(p.numel() for p in model.parameters()), changed_phase_planes=changed,
                  updates=protocol['epochs'] * math.ceil(len(train[1])/protocol['batch_size']),
                  order_sha256=orders, transform_sha256=transforms,
                  checkpoint_sha256=r.sha(dest/'best_checkpoint.pt'),
                  last_checkpoint_sha256=r.sha(dest/'last_checkpoint.pt'),
                  threshold_sha256=r.sha(dest/'thresholds.json'))
    r.save(dest / 'completed.json', result)
    del model, opt, scheduler, ck
    gc.collect()
    torch.cuda.empty_cache()
    return result


def smoke(protocol, train, variants, out):
    cfg = protocol['augmentation']
    a = affine_parameters(8, 17, 1, cfg)
    assert torch.equal(a, affine_parameters(8, 17, 1, cfg))
    assert not torch.equal(a, affine_parameters(8, 17, 2, cfg))
    x = train[0][:8]
    zero = dict(degrees=0., translation_pixels=0., scale_delta=0.)
    identity_error = float((augment(x, affine_parameters(8, 17, 1, zero))-x).abs().max())
    assert identity_error < 1e-5
    transformed = augment(x, a)
    assert transformed.shape == x.shape and transformed.min() >= 0 and transformed.max() <= 1
    layer = PhaseLayer(8, parameterization='sigmoid', init='zeros').cuda()
    assert phase_smoothness(layer).item() == 0
    with torch.no_grad():
        layer.raw_phase[0, 0] = 1
    regularizer = phase_smoothness(layer)
    regularizer.backward()
    assert regularizer > 0 and layer.raw_phase.grad.abs().sum() > 0
    # Duplicating a plane must not double the penalty.
    duplicate = torch.nn.ModuleList([copy.deepcopy(layer), copy.deepcopy(layer)])
    assert torch.allclose(phase_smoothness(duplicate), regularizer)
    rows = []
    for v in variants:
        r.setseed(17)
        model = r.build(v['architecture'], r.CONFIGS[v['id']]).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=protocol['lr'])
        pred = model(transformed)
        loss = r.objective(pred, train[1][:8], model.masks) + protocol['phase_smooth_weight'] * phase_smoothness(model)
        loss.backward()
        grads = {n: float(p.grad.norm()) for n, p in model.named_parameters()}
        assert all(np.isfinite(g) and g > 0 for g in grads.values())
        opt.step()
        rows.append(dict(variant=v['id'], loss=float(loss), gradients=grads))
        del model, opt, pred, loss
        gc.collect()
        torch.cuda.empty_cache()
    r.save(out/'smoke.json', dict(identity_error=identity_error, augmentation_is_deterministic=True,
           regularizer_duplicate_invariance=True, variants=rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['smoke', 'train', 'test'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profile', type=Path, default=Path(__file__).with_name('adrenal_regularization.json'))
    p.add_argument('--seeds', type=int, nargs='+', default=[17])
    args = p.parse_args()
    protocol = r.read(args.profile)
    r.EXP.update({k: protocol[k] for k in ['epochs','batch_size','lr','step_size','gamma','weight_decay']})
    r.EXP['data_npz'] = str(args.data.resolve())
    r.setup()
    sources = {str(f.relative_to(TASK)): r.sha(f) for f in sorted(ARCHIVE.rglob('*'))
               if f.suffix in {'.py','.yaml'} and f.is_file()}
    sources[str(Path(__file__).relative_to(TASK))] = r.sha(__file__)
    sources['profile'] = r.sha(args.profile)
    variants = [v for v in r.VARIANTS if v['activation'] in ['relu_softsign','off']]
    out = args.out.resolve()
    if args.phase == 'test':
        lock = r.read(out/'test_lock.json')
        assert lock['source_hashes'] == sources
        for rel, digest in lock['files'].items():
            assert r.sha(out/rel) == digest, rel
        with np.load(args.data, allow_pickle=False) as z:
            x, y, ids = z['test_images'].copy(), z['test_labels'].reshape(-1).copy(), z['test_ids'].copy()
        assert np.bincount(y).tolist() == [229,69] and len(set(ids)) == 298
        x = F.interpolate(torch.from_numpy(x[:,None]), size=(100,100), mode='bicubic',
                          align_corners=False, antialias=True).clamp(0,1).cuda()
        data = x, torch.from_numpy(y).long().cuda(), ids
        results = []
        for item in lock['models']:
            dest = out/item['directory']
            ck = torch.load(dest/'best_checkpoint.pt', map_location='cpu', weights_only=False)
            cfg = r.read(dest/'config.json')
            model = r.build(ck['variant']['architecture'], cfg).cuda()
            model.load_state_dict(ck['model'])
            metrics, rows = clean_evaluate(model, data, dest, 'test')
            pscore = np.array([[row['score0'],row['score1']] for row in rows])
            t = r.read(dest/'thresholds.json')['policies']['val_balanced']['threshold']
            calibrated = full_metrics(y, pscore, t, metrics['detector_plane_mse'])
            r.save(dest/'test_val_threshold_metrics.json', calibrated)
            results.append(dict(variant=ck['variant']['id'], seed=ck['seed'],
                                selected_epoch=ck['epoch'], fixed=metrics, val_threshold=calibrated))
            del model, ck
            torch.cuda.empty_cache()
        r.save(out/'test_results.json', results)
        r.save(out/'test_execution.json', dict(command=sys.argv, time=r.now(), source_hashes=sources,
               git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()))
        r.save(out/'status.json', dict(state='test_complete'))
        print(json.dumps(results), flush=True)
        return
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(command=sys.argv, protocol=protocol, seeds=args.seeds, source_hashes=sources,
                    data_sha256=r.sha(args.data), git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                    python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True), time=r.now())
    r.save(out/'metadata.json', metadata)
    train, val = r.getdata('train'), r.getdata('val')
    if args.phase == 'smoke':
        smoke(protocol, train, variants, out)
        return
    results, files, models = [], {}, []
    for seed in args.seeds:
        for v in variants:
            results.append(train_variant(v, seed, train, val, protocol, out, sources))
            dest = Path('runs') / v['id'] / ('seed'+str(seed))
            models.append(dict(directory=dest.as_posix()))
            for name in ['best_checkpoint.pt','thresholds.json','config.json']:
                files[(dest/name).as_posix()] = r.sha(out/dest/name)
            r.save(out/'validation_results.json', results)
    # Lock all paired checkpoints and thresholds before any new test access.
    r.save(out/'test_lock.json', dict(source_hashes=sources, files=files, models=models, time=r.now(),
           test_status=protocol['test_status'], selection=protocol['selection']))
    r.save(out/'status.json', dict(state='training_complete_test_not_read'))


if __name__ == '__main__':
    main()
