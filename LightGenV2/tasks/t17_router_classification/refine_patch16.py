"""Validation-only PATCH16 MoE fine-tuning without changing its inference graph."""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model_four import CONTRACT, FourRouterClassification, encode_rgb
from .prepare_patch16 import digest, write
from .train_patch16 import evaluate, loader, score
from .train_regularized import augment_d4, router_balance_loss, update_ema


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--min-epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--train-eval-every', type=int, default=5)
    p.add_argument('--phase-lr', type=float, default=.001)
    p.add_argument('--head-lr', type=float, default=.0002)
    p.add_argument('--balance-weight', type=float, default=.1)
    p.add_argument('--balance-mode', choices=['dense_kl', 'top2_load'], default='dense_kl')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    a = p.parse_args()
    if not 1 <= a.min_epochs <= a.epochs or a.patience < 1 or a.train_eval_every < 1:
        raise ValueError('invalid epoch budget')
    if a.phase_lr <= 0 or a.head_lr <= 0 or a.balance_weight < 0 or a.batch < 1 or a.workers < 0:
        raise ValueError('invalid optimization settings')
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=False)
    write(a.out / 'status.json', dict(state='loading'))
    try:
        source = a.init_run / 'best_checkpoint.pt'
        parent = torch.load(source, map_location='cpu', weights_only=False)
        prior = parent['config']
        if prior['dataset'] != 'PATCH16' or prior['architecture'] != 'optical':
            raise ValueError('PATCH16 optical MoE checkpoint required')
        if prior['optical_contract'] != CONTRACT or prior['top_k'] != 2 or prior['num_classes'] != 16:
            raise ValueError('optical contract changed')
        data_root = Path(prior['data_root'])
        manifest = json.loads((data_root / 'manifest.json').read_text())
        if digest(data_root / 'manifest.json') != prior['manifest_sha256']:
            raise ValueError('manifest identity changed')
        if digest(data_root / 'data.zip') != manifest['archive_sha256']:
            raise ValueError('dataset archive identity changed')
        if manifest['dataset'] != 'PATCH16' or manifest['license'] != 'CC BY 4.0':
            raise ValueError('wrong dataset or license')
        if manifest['split_counts']['train'] != 88971 or manifest['split_counts']['val'] != 12354:
            raise ValueError('split counts changed')
        cfg = dict(dataset='PATCH16', architecture='optical', num_classes=16,
                   optical_contract=CONTRACT, top_k=2, init_run=str(a.init_run),
                   init_checkpoint_sha256=digest(source), init_epoch=parent['epoch'],
                   init_weights_kind=parent['weights_kind'], parent_git_commit=prior['git_commit'],
                   data_root=str(data_root), manifest_sha256=prior['manifest_sha256'],
                   archive_sha256=manifest['archive_sha256'], skip_test=True,
                   phase_lr=a.phase_lr, head_lr=a.head_lr,
                   balance_weight=a.balance_weight, balance_mode=a.balance_mode,
                   batch=a.batch, epochs=a.epochs, min_epochs=a.min_epochs,
                   patience=a.patience, train_eval_every=a.train_eval_every,
                   seed=a.seed, ema_decay=.995, label_smoothing=.03,
                   class_weight_power=.5, augmentation_probability=.5,
                   selection='full validation (accuracy + macro recall)/2',
                   command=sys.argv, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        model = FourRouterClassification('optical', num_classes=16).cuda()
        model.load_state_dict(parent['model'])
        ema = copy.deepcopy(model).eval()
        ema.requires_grad_(False)
        train = loader(data_root, 'train', manifest, a.batch, a.workers, True, a.seed)
        clean_train = loader(data_root, 'train', manifest, a.batch, a.workers)
        val = loader(data_root, 'val', manifest, a.batch, a.workers)
        counts = torch.tensor(manifest['class_counts']['train'], device='cuda', dtype=torch.float)
        weights = (counts.mean() / counts).pow(.5)
        weights /= weights.mean()
        cfg['loss_class_weights'] = weights.cpu().tolist()
        cfg['total_parameters'] = sum(p.numel() for p in model.parameters())
        write(a.out / 'config.json', cfg)
        baseline, _ = evaluate(model, val)
        write(a.out / 'parent_validation.json', dict(metrics=baseline, score=score(baseline),
                                                   checkpoint_sha256=cfg['init_checkpoint_sha256']))
        groups = []
        for name, parameter in model.named_parameters():
            electronic = 'shared_head' in name
            groups.append(dict(params=[parameter], lr=a.head_lr if electronic else a.phase_lr,
                               weight_decay=.01 if electronic else 0.))
        optimizer = torch.optim.AdamW(groups)
        base_lrs = [g['lr'] for g in optimizer.param_groups]
        best = -1.
        significant_best = score(baseline)
        last_significant = 0
        history = []
        stop_reason = 'maximum candidate budget; no convergence claim'
        for epoch in range(1, a.epochs + 1):
            start = time.monotonic()
            model.train()
            factor = .1 + .9 * .5 * (1 + np.cos(np.pi * (epoch - 1) / max(1, a.epochs - 1)))
            for group, initial_lr in zip(optimizer.param_groups, base_lrs):
                group['lr'] = initial_lr * factor
            seen = 0
            task_loss_sum = 0.
            balance_loss_sum = 0.
            for images, labels, _ in train:
                labels = labels.cuda(non_blocking=True)
                amplitude = encode_rgb(images.cuda(non_blocking=True))
                codes = torch.randint(0, 8, (len(labels),), device='cuda')
                codes = torch.where(torch.rand(len(labels), device='cuda') < .5,
                                    codes, torch.zeros_like(codes))
                amplitude = augment_d4(amplitude, codes)
                optimizer.zero_grad(set_to_none=True)
                output = model(amplitude, return_dense_route=True)
                task_loss = F.cross_entropy(output['logits'], labels, weight=weights,
                                            label_smoothing=.03)
                balance_loss = router_balance_loss(output['dense_route_power'], a.balance_mode)
                loss = task_loss + a.balance_weight * balance_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                update_ema(ema, model, .995)
                n = len(labels)
                seen += n
                task_loss_sum += float(task_loss.detach()) * n
                balance_loss_sum += float(balance_loss.detach()) * n
            if seen != manifest['split_counts']['train']:
                raise ValueError('full training split not traversed')
            online, _ = evaluate(model, val)
            averaged, _ = evaluate(ema, val)
            kind, chosen, metrics = max((('online', model, online), ('ema', ema, averaged)),
                                        key=lambda item: (score(item[2]), -item[2]['cross_entropy']))
            current = score(metrics)
            if current > significant_best + .002:
                significant_best = current
                last_significant = epoch
            if current > best:
                best = current
                torch.save(dict(model=chosen.state_dict(), epoch=epoch,
                                weights_kind=kind, config=cfg), a.out / 'best_checkpoint.pt')
                write(a.out / 'selected_validation.json', dict(epoch=epoch, weights_kind=kind,
                                                               metrics=metrics, selection_score=current))
            row = dict(epoch=epoch, training_samples_seen=seen, task_loss=task_loss_sum / seen,
                       balance_loss=balance_loss_sum / seen, validation=metrics,
                       weights_kind=kind, seconds=time.monotonic() - start)
            if epoch % a.train_eval_every == 0 or epoch == a.epochs:
                row['clean_training'], _ = evaluate(chosen, clean_train)
                row['train_validation_accuracy_gap_pp'] = 100 * (
                    row['clean_training']['accuracy'] - metrics['accuracy'])
                row['seconds'] = time.monotonic() - start
            history.append(row)
            write(a.out / 'metrics.json', history)
            torch.save(dict(model=model.state_dict(), ema_model=ema.state_dict(),
                            optimizer=optimizer.state_dict(), epoch=epoch, config=cfg),
                       a.out / 'last_checkpoint.pt')
            write(a.out / 'status.json', dict(state='training_validation_only', epoch=epoch,
                                            best_validation_score=best, seconds=row['seconds']))
            print(json.dumps(dict(epoch=epoch, score=current, accuracy=metrics['accuracy'],
                                  macro_recall=metrics['balanced_accuracy'],
                                  route=metrics.get('router'), seconds=row['seconds'])), flush=True)
            if epoch >= a.min_epochs and epoch - last_significant >= a.patience:
                stop_reason = 'no >=0.2pp validation selection-score improvement for patience window'
                break
        write(a.out / 'status.json', dict(state='completed_validation_only', epochs=epoch,
                                        stop_reason=stop_reason,
                                        best_validation=json.loads((a.out / 'selected_validation.json').read_text()),
                                        parent_validation_score=score(baseline), test_evaluations=0))
    except Exception as error:
        write(a.out / 'status.json', dict(state='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
