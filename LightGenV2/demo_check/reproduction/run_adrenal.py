"""Run the archived training functions in an isolated, explicitly limited reproduction.

This fixes the already reported Softsign configuration; it does not repeat activation
selection or claim to reproduce the original 68-run search protocol.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--phase', choices=['smoke', 'train'], required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--depth', type=int, choices=[2, 4, 6], default=2)
    args = parser.parse_args()
    archive = Path(__file__).resolve().parents[1] / 'adrenal_softsign_code_export_20260915_145336/code'
    sys.path.insert(0, str(archive))
    import numpy as np
    import torch
    import torchvision
    import run_experiment as r
    from torch.nn import functional as F

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_hashes = {p.relative_to(archive).as_posix(): r.sha(p) for p in sorted(archive.rglob('*'))
                     if p.is_file() and p.suffix in {'.py', '.yaml'}}
    source_hashes['reproduction/run_adrenal.py'] = r.sha(__file__)
    original_experiment = copy.deepcopy(r.EXP)
    r.EXP['data_npz'] = str(args.data.resolve())
    for cfg in r.CONFIGS.values():
        cfg['experiment']['data_npz'] = str(args.data.resolve())
    r.ROOT = out
    r.run_dir = lambda variant, seed: out / 'runs' / variant['id'] / f'seed{seed}'
    shutil.copy2(archive / 'reference_initialization.yaml', out / 'reference_initialization.yaml')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=archive, text=True).strip()
    metadata = dict(command=sys.argv, git_commit=commit, source_hashes=source_hashes,
                    original_experiment=original_experiment, actual_experiment=r.EXP,
                    phase=args.phase, seed=args.seed, depth=args.depth,
                    scope='fixed Softsign/off subset; no activation re-selection',
                    python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    pip_freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    r.save(out / 'metadata.json', metadata)
    r.save(out / 'configs.json', r.CONFIGS)
    r.setup()
    train, val = r.getdata('train'), r.getdata('val')
    variants = [v for v in r.VARIANTS if v['activation'] in ['relu_softsign', 'off']]
    if args.phase == 'smoke':
        rows = []
        for v in variants:
            r.setseed(args.seed)
            model = r.build(v['architecture'], r.CONFIGS[v['id']]).cuda()
            count = sum(p.numel() for p in model.parameters())
            assert count == r.parameters(v), (v['id'], count, r.parameters(v))
            initial = {n: r.sha_tensor(p) for n, p in model.named_parameters()}
            import yaml
            reference = yaml.safe_load((archive / 'reference_initialization.yaml').read_text())
            assert initial == reference[f"{v['architecture']}_L{v['depth']}_seed{args.seed}"]
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            prediction = model(train[0][:8])
            loss = r.objective(prediction, train[1][:8], model.masks)
            assert torch.isfinite(loss)
            loss.backward()
            gradients = {n: dict(finite=bool(torch.isfinite(p.grad).all()),
                                 norm=float(p.grad.norm()))
                         for n, p in model.named_parameters() if p.grad is not None}
            assert len(gradients) == len(initial)
            assert all(x['finite'] and x['norm'] > 0 for x in gradients.values())
            routing = prediction['route_probabilities']
            if routing is not None:
                assert bool(torch.isfinite(routing).all()) and bool((routing > 0).all())
                assert torch.allclose(routing.sum(1), torch.ones(8, device='cuda'), atol=1e-6)
            torch.cuda.synchronize()
            row = dict(variant=v['id'], parameters=count, loss=float(loss), gradients=gradients,
                       seconds=time.perf_counter()-started,
                       peak_memory_bytes=torch.cuda.max_memory_allocated(),
                       mean_route_probabilities=routing.detach().mean(0).cpu().tolist() if routing is not None else None)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model, prediction, loss, routing
            gc.collect(); torch.cuda.empty_cache()
        r.save(out / 'smoke.json', rows)
    else:
        variants = [v for v in variants if v['depth'] == args.depth]
        for index, v in enumerate(variants, 1):
            r.train_one(v, args.seed, train, val, index, source_hashes)
        # Lock this subset before reading test data. Do not counterfeit the
        # original pipeline's 60-checkpoint lock or copy its historical locks.
        checkpoints = {str(r.run_dir(v, args.seed).relative_to(out) / 'best.pt'):
                       r.sha(r.run_dir(v, args.seed) / 'best.pt') for v in variants}
        r.save(out / 'subset_test_lock.json', dict(source_hashes=source_hashes,
               checkpoints=checkpoints, threshold=0.5, variants=variants, locked_at=r.now()))
        with np.load(r.EXP['data_npz'], allow_pickle=False) as z:
            x, y, ids = z['test_images'].copy(), z['test_labels'].reshape(-1).copy(), z['test_ids'].copy()
        assert np.bincount(y).tolist() == [229, 69]
        xt = F.interpolate(torch.from_numpy(x[:, None]), size=(100, 100), mode='bicubic',
                           align_corners=False, antialias=True).clamp(0, 1).cuda()
        test = xt, torch.from_numpy(y).long().cuda(), ids
        results = []
        for v in variants:
            dest = r.run_dir(v, args.seed)
            assert r.sha(dest / 'best.pt') == checkpoints[str((dest / 'best.pt').relative_to(out))]
            checkpoint = torch.load(dest / 'best.pt', map_location='cpu', weights_only=False)
            model = r.build(v['architecture'], r.CONFIGS[v['id']]).cuda()
            model.load_state_dict(checkpoint['model'], strict=True)
            metrics, predictions = r.evaluate(model, test, predictions=True)
            r.save(dest / 'test_metrics.json', metrics)
            r.csvwrite(dest / 'test_predictions.csv', predictions)
            results.append(dict(variant=v['id'], seed=args.seed, selected_epoch=checkpoint['epoch'], **metrics))
            del checkpoint, model
            torch.cuda.empty_cache()
        r.save(out / 'results.json', results)
        print(json.dumps(results), flush=True)
    r.status(state='complete', phase=args.phase)


if __name__ == '__main__':
    main()
