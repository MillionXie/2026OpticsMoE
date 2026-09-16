"""Pretrain once; freeze identical electronics; train two matched optical models."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import Electronic, FrozenFusion, PURE

spec = importlib.util.spec_from_file_location('phase_only_runner', PURE / 'run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
HERE = Path(__file__).resolve().parent
save, sha, tensors_sha, batches = base.save, base.sha, base.tensors_sha, base.batches


def metric(prob, target, domain):
    pred = prob.argmax(1)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (target, pred), 1)
    return dict(accuracy=float((pred == target).mean()),
                domain_accuracy={str(k): float((pred[domain == k] == target[domain == k]).mean()) for k in (0, 1)},
                loss=float(-np.log(np.maximum(prob[np.arange(len(target)), target], 1e-12)).mean()),
                confusion_matrix=confusion.tolist())


@torch.no_grad()
def evaluate(model, data, batch_size):
    model.eval()
    buckets = {}; routes = []; captures = []
    for images, _, _, _ in batches(*data, torch.arange(len(data[1])), batch_size):
        if isinstance(model, Electronic):
            values = {'electronic': model(images).softmax(1)}
        else:
            out = model(images)
            values = {k: out[v] for k, v in [('fused', 'probabilities'), ('electronic', 'electronic'), ('optical', 'optical')]}
            captures.append(out['detector_capture'].cpu().numpy())
            if out['route_power'] is not None:
                routes.append(out['route_power'].cpu().numpy())
        for key, prob in values.items():
            assert bool(torch.isfinite(prob).all()) and bool((prob >= 0).all())
            assert torch.allclose(prob.sum(1), torch.ones(len(prob), device=prob.device), atol=2e-6)
            buckets.setdefault(key, []).append(prob.cpu().numpy())
    probabilities = {k: np.concatenate(v) for k, v in buckets.items()}
    target, domain = data[1].numpy(), data[2].numpy()
    metrics = {k: metric(v, target, domain) for k, v in probabilities.items()}
    if captures:
        metrics['detector_capture_mean'] = float(np.concatenate(captures).mean())
    if routes:
        q = np.concatenate(routes)
        metrics['routing'] = {str(k): dict(mean_power=q[domain == k].mean(0).tolist(),
                                           std_power=q[domain == k].std(0).tolist(),
                                           largest_expert_counts=np.bincount(q[domain == k].argmax(1), minlength=4).tolist()) for k in (0, 1)}
    return metrics, probabilities


def write_predictions(dest, arrays, probabilities):
    for name, prob in probabilities.items():
        with (dest / ('validation_' + name + '.csv')).open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'domain', 'label', 'prediction'] + [f'p{i}' for i in range(10)])
            writer.writerows([str(i), int(d), int(y), int(p.argmax()), *p.tolist()]
                             for i, d, y, p in zip(arrays['validation_ids'], arrays['validation_domains'], arrays['validation_labels'], prob))


def train_stage(model, stage, train, val, cfg, optical_cfg, out, arrays, frozen_hash=None):
    dest = out / stage
    dest.mkdir()
    is_electronic = stage == 'electronic'
    epochs = cfg['electronic_epochs'] if is_electronic else cfg['optical_epochs']
    lr = cfg['electronic_learning_rate'] if is_electronic else optical_cfg['learning_rate']
    parameters = list(model.parameters()) if is_electronic else list(model.optical.parameters())
    initial = tensors_sha(dict(model.named_parameters()) if is_electronic else dict(model.optical.named_parameters()))
    optimizer = (torch.optim.AdamW(parameters, lr=lr, weight_decay=cfg['electronic_weight_decay']) if is_electronic
                 else torch.optim.Adam(parameters, lr=lr, weight_decay=0.))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*.1)
    initial_validation, _ = evaluate(model, val, cfg['batch_size'])
    save(dest/'initial_validation.json', initial_validation)
    history = []; best = (-1., float('-inf')); started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(train[1]), generator=torch.Generator().manual_seed(cfg['seed']*1000003+epoch))
        augmentation_seed = cfg['seed']*99991+epoch
        loss_sum = 0.; correct = 0; count = 0; first_gradients = None
        for images, labels, _, _ in batches(*train, order, cfg['batch_size'], augment_seed=augmentation_seed):
            optimizer.zero_grad(set_to_none=True)
            if is_electronic:
                logits = model(images); prob = logits.softmax(1); loss = F.cross_entropy(logits, labels)
            else:
                result = model(images); prob = result['probabilities']
                loss = F.nll_loss(prob.clamp_min(1e-12).log(), labels)
            assert bool(torch.isfinite(loss))
            loss.backward()
            if not is_electronic:
                assert all(p.grad is None for p in model.electronic.parameters())
                if first_gradients is None:
                    first_gradients = {n: float(p.grad.norm()) for n, p in model.optical.named_parameters()}
                    assert all(np.isfinite(v) and v > 0 for v in first_gradients.values())
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss)*len(labels); correct += int((prob.argmax(1) == labels).sum()); count += len(labels)
        validation, _ = evaluate(model, val, cfg['batch_size'])
        if not is_electronic:
            assert tensors_sha(model.electronic.state_dict()) == frozen_hash
        chosen = validation['electronic' if is_electronic else 'fused']
        row = dict(epoch=epoch, train_loss=loss_sum/count, train_accuracy=correct/count, validation=validation,
                   learning_rate=optimizer.param_groups[0]['lr'],
                   order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(), augmentation_seed=augmentation_seed,
                   optical_first_batch_gradients=first_gradients, frozen_electronic_verified=not is_electronic,
                   seconds=time.perf_counter()-started)
        history.append(row)
        state = model.state_dict() if is_electronic else model.optical.state_dict()
        payload = dict(model=state, epoch=epoch, config=cfg, optical_config=optical_cfg, stage=stage,
                       frozen_electronic_sha256=frozen_hash, validation=validation)
        key = (chosen['accuracy'], -chosen['loss'])
        if key > best:
            best = key; torch.save(payload, dest/'best_checkpoint.pt')
        scheduler.step()
        torch.save(dict(**payload, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), dest/'last_checkpoint.pt')
        save(dest/'history.json', history)
        save(out/'status.json', dict(state='training', stage=stage, epoch=epoch, epochs=epochs))
        print(json.dumps(dict(stage=stage, epoch=epoch, train_accuracy=correct/count,
                              validation={k: v['accuracy'] for k, v in validation.items() if isinstance(v, dict) and 'accuracy' in v},
                              seconds=row['seconds'])), flush=True)
    ckpt = torch.load(dest/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    (model if is_electronic else model.optical).load_state_dict(ckpt['model'])
    final, probabilities = evaluate(model, val, cfg['batch_size'])
    train_final, _ = evaluate(model, train, cfg['batch_size'])
    write_predictions(dest, arrays, probabilities)
    end = tensors_sha(dict(model.named_parameters()) if is_electronic else dict(model.optical.named_parameters()))
    assert all(end[k] != v for k, v in initial.items())
    summary = dict(stage=stage, selected_epoch=ckpt['epoch'], validation=final, train_unaugmented=train_final,
                   trainable_parameters=sum(p.numel() for p in parameters), initial_parameters_sha256=initial,
                   final_parameters_sha256=end, checkpoint_sha256=sha(dest/'best_checkpoint.pt'),
                   frozen_electronic_sha256=frozen_hash, seconds=time.perf_counter()-started)
    save(dest/'summary.json', summary)
    return summary, probabilities


def smoke(cfg, optical_cfg, out):
    electronic = Electronic([.5]*3, [.2]*3, cfg['seed']).cuda()
    images = torch.randint(1, 255, (8, 56, 56, 3), dtype=torch.uint8, device='cuda')
    labels = torch.arange(8, device='cuda')
    electronic.train(); optimizer = torch.optim.AdamW(electronic.parameters(), lr=.001)
    before = tensors_sha(electronic.state_dict())
    F.cross_entropy(electronic(images), labels).backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    assert before != tensors_sha(electronic.state_dict())
    electronic.eval().requires_grad_(False)
    frozen = tensors_sha(electronic.state_dict()); reports = []
    for architecture in cfg['architectures']:
        model = FrozenFusion(electronic, architecture, optical_cfg).cuda().train()
        assert not model.electronic.training
        assert all(not m.training for m in model.electronic.modules())
        result = model(images)
        assert torch.equal(result['probabilities'], .5*result['electronic']+.5*result['optical'])
        assert torch.allclose(result['probabilities'].sum(1), torch.ones(8, device='cuda'), atol=1e-6)
        loss = F.nll_loss(result['probabilities'].log(), labels); loss.backward()
        grads = {n: float(p.grad.norm()) for n, p in model.optical.named_parameters()}
        assert all(np.isfinite(v) and v > 0 for v in grads.values())
        assert all(p.grad is None and not p.requires_grad for p in electronic.parameters())
        assert frozen == tensors_sha(electronic.state_dict())
        with torch.no_grad():
            expected = model.optical(images)['probabilities']
            assert torch.equal(result['optical'], expected)
            assert torch.allclose(result['input_power'], result['output_power'], atol=1e-5)
        reports.append(dict(architecture=architecture, loss=float(loss), gradients=grads,
                            electronic_parameters=sum(p.numel() for p in electronic.parameters()),
                            optical_parameters=sum(p.numel() for p in model.optical.parameters()),
                            frozen_state_verified=True, normalized_half_fusion_verified=True))
        del model, result, loss; torch.cuda.empty_cache()
    save(out/'smoke.json', reports); print(json.dumps(reports), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'train'], required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    optical_cfg = json.loads((PURE/'config.json').read_text())
    assert cfg['electronic_weight'] == cfg['optical_weight'] == .5
    assert cfg['electronic_dropout'] == .2
    assert cfg['batch_size'] == optical_cfg['batch_size']
    assert cfg['optical_epochs'] == optical_cfg['epochs']
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    files = [*HERE.glob('*.py'), args.config, PURE/'models.py', PURE/'run.py', PURE/'config.json', base.ARCHIVE/'optical_reference/optics.py']
    metadata = dict(config=cfg, optical_config=optical_cfg, command=sys.argv,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
                    source_sha256={str(p.relative_to(HERE.parent)): sha(p) for p in files},
                    python=sys.version, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    environment=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
                    scope='validation-only matched 6000/2000 pilot; no test read')
    save(args.out/'metadata.json', metadata)
    if args.phase == 'smoke':
        smoke(cfg, optical_cfg, args.out)
    else:
        manifest_path = args.data.with_name('manifest.json')
        manifest = json.loads(manifest_path.read_text())
        assert sha(args.data) == manifest['data_sha256']
        assert manifest['protocol']['protocol'] == cfg['dataset_protocol'] == optical_cfg['protocol']
        with np.load(args.data, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        assert not any(k.startswith('test') for k in arrays)
        train = tuple(torch.from_numpy(arrays['train_'+k]) for k in ['images', 'labels', 'domains'])
        val = tuple(torch.from_numpy(arrays['validation_'+k]) for k in ['images', 'labels', 'domains'])
        for split, data in [('train', train), ('validation', val)]:
            assert tuple(data[0].shape[1:]) == (56, 56, 3)
            assert data[0].dtype == torch.uint8
            for domain in (0, 1):
                assert torch.bincount(data[1][data[2] == domain], minlength=10).tolist() == [optical_cfg[split+'_pairs_per_class']]*10
        assert not set(arrays['train_ids']) & set(arrays['validation_ids'])
        # Statistics use training pixels only; accumulated float64 for stability.
        pixels = arrays['train_images'].astype(np.float64)/255.
        mean = pixels.mean(axis=(0, 1, 2)).tolist(); std = pixels.std(axis=(0, 1, 2)).tolist(); del pixels
        assert min(std) > 1e-5
        metadata.update(data_sha256=sha(args.data), data_manifest_sha256=sha(manifest_path),
                        split_sha256=manifest['original_split_sha256'], train_channel_mean=mean, train_channel_std=std)
        save(args.out/'metadata.json', metadata)
        electronic = Electronic(mean, std, cfg['seed']).cuda()
        summary, e_prob = train_stage(electronic, 'electronic', train, val, cfg, optical_cfg, args.out, arrays)
        electronic.zero_grad(set_to_none=True); electronic.eval().requires_grad_(False)
        frozen = tensors_sha(electronic.state_dict())
        save(args.out/'frozen_electronic.json', dict(selected_epoch=summary['selected_epoch'],
             checkpoint_sha256=summary['checkpoint_sha256'], tensors_sha256=frozen, trainable_parameters=0))
        results = [summary]
        for architecture in cfg['architectures']:
            model = FrozenFusion(electronic, architecture, optical_cfg).cuda()
            summary, prob = train_stage(model, architecture, train, val, cfg, optical_cfg, args.out, arrays, frozen)
            assert np.array_equal(prob['electronic'], e_prob['electronic'])
            assert tensors_sha(electronic.state_dict()) == frozen
            results.append(summary)
            save(args.out/'results.json', results)
            del model; torch.cuda.empty_cache()
        histories = [json.loads((args.out/a/'history.json').read_text()) for a in cfg['architectures']]
        assert len({tuple((x['order_sha256'], x['augmentation_seed']) for x in h) for h in histories}) == 1
        save(args.out/'fairness.json', dict(electronic_state_and_predictions_identical=True,
              optical_order_and_augmentation_identical=True, electronic_weight=.5, optical_weight=.5,
              shared_electronic_checkpoint_sha256=sha(args.out/'electronic/best_checkpoint.pt'),
              test_set_used=False))
    save(args.out/'status.json', dict(state='complete', phase=args.phase))


if __name__ == '__main__':
    main()
