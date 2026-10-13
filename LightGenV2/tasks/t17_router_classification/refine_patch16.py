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
from .gpu_policy import require_non_a100_gpu


def resume_progress(history, checkpoint_epoch, parent_score):
    """Resume the completed prefix, including the original early-stop clock."""
    if checkpoint_epoch < 1 or [row['epoch'] for row in history] != list(range(1, checkpoint_epoch + 1)):
        raise ValueError('history must exactly match the last completed checkpoint')
    best = max(score(row['validation']) for row in history)
    significant_best = parent_score
    last_significant = 0
    for row in history:
        current = score(row['validation'])
        if current > significant_best + .002:
            significant_best = current
            last_significant = row['epoch']
    return best, significant_best, last_significant


def validate_resume_settings(args, config):
    keys = ('init_run', 'epochs', 'min_epochs', 'patience', 'train_eval_every',
            'phase_lr', 'head_lr', 'balance_weight', 'balance_mode',
            'router_warmup_epochs', 'router_warmup_lr', 'batch', 'seed')
    for key in keys:
        value = str(getattr(args, key)) if key == 'init_run' else getattr(args, key)
        if value != config[key]:
            raise ValueError(f'resume must preserve {key}')


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(equal_state(x, y) for x, y in zip(left, right))
    return left == right


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
    p.add_argument('--router-warmup-epochs', type=int, default=0)
    p.add_argument('--router-warmup-lr', type=float, default=.005)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--resume', action='store_true',
                   help='continue an interrupted run from last; no repeated warmup or completed epochs')
    a = p.parse_args()
    if not 1 <= a.min_epochs <= a.epochs or a.patience < 1 or a.train_eval_every < 1:
        raise ValueError('invalid epoch budget')
    if a.phase_lr <= 0 or a.head_lr <= 0 or a.balance_weight < 0 or a.batch < 1 or a.workers < 0:
        raise ValueError('invalid optimization settings')
    if a.router_warmup_epochs < 0 or a.router_warmup_lr <= 0:
        raise ValueError('invalid router warmup settings')
    gpu_uuid = require_non_a100_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    resume = None
    if a.resume:
        status = json.loads((a.out / 'status.json').read_text())
        if not status['state'].startswith('interrupted'):
            raise ValueError('resume requires an explicitly interrupted run')
        stored_config = json.loads((a.out / 'config.json').read_text())
        validate_resume_settings(a, stored_config)
        resume = torch.load(a.out / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        if not 1 <= resume['epoch'] < a.epochs:
            raise ValueError('no remaining completed-prefix training budget')
        if resume['config'] != stored_config or resume.get('ema_model') is None:
            raise ValueError('resume checkpoint/config or EMA identity mismatch')
    else:
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
                   router_warmup_epochs=a.router_warmup_epochs,
                   router_warmup_lr=a.router_warmup_lr,
                   batch=a.batch, epochs=a.epochs, min_epochs=a.min_epochs,
                   patience=a.patience, train_eval_every=a.train_eval_every,
                   seed=a.seed, ema_decay=.995, label_smoothing=.03,
                   class_weight_power=.5, augmentation_probability=.5,
                   selection='full validation (accuracy + macro recall)/2',
                   command=sys.argv, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        model = FourRouterClassification('optical', num_classes=16).cuda()
        model.load_state_dict(resume['model'] if resume is not None else parent['model'])
        ema = copy.deepcopy(model).eval()
        ema.requires_grad_(False)
        if resume is not None:
            ema.load_state_dict(resume['ema_model'])
        train = loader(data_root, 'train', manifest, a.batch, a.workers, True, a.seed)
        clean_train = loader(data_root, 'train', manifest, a.batch, a.workers)
        val = loader(data_root, 'val', manifest, a.batch, a.workers)
        counts = torch.tensor(manifest['class_counts']['train'], device='cuda', dtype=torch.float)
        weights = (counts.mean() / counts).pow(.5)
        weights /= weights.mean()
        cfg['loss_class_weights'] = weights.cpu().tolist()
        cfg['total_parameters'] = sum(p.numel() for p in model.parameters())
        runtime_commit = cfg['git_commit']
        if resume is not None:
            if cfg['init_checkpoint_sha256'] != stored_config['init_checkpoint_sha256']:
                raise ValueError('parent checkpoint identity changed')
            cfg = stored_config
            parent_report = json.loads((a.out / 'parent_validation.json').read_text())
            if parent_report['checkpoint_sha256'] != cfg['init_checkpoint_sha256']:
                raise ValueError('parent validation identity changed')
            baseline = parent_report['metrics']
        else:
            write(a.out / 'config.json', cfg)
            baseline, _ = evaluate(model, val)
            write(a.out / 'parent_validation.json', dict(metrics=baseline, score=score(baseline),
                                                       checkpoint_sha256=cfg['init_checkpoint_sha256']))
        if a.router_warmup_epochs and resume is None:
            router_optimizer = torch.optim.Adam([model.router_phase], lr=a.router_warmup_lr)
            for warmup_epoch in range(1, a.router_warmup_epochs + 1):
                model.train()
                seen = 0
                loss_sum = 0.
                for images, labels, _ in train:
                    amplitude = encode_rgb(images.cuda(non_blocking=True))
                    codes = torch.randint(0, 8, (len(labels),), device='cuda')
                    codes = torch.where(torch.rand(len(labels), device='cuda') < .5,
                                        codes, torch.zeros_like(codes))
                    amplitude = augment_d4(amplitude, codes)
                    router_optimizer.zero_grad(set_to_none=True)
                    dense, _ = model.dense_route(amplitude)
                    loss = router_balance_loss(dense, 'dense_kl')
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite router warmup loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([model.router_phase], 5.)
                    router_optimizer.step()
                    n = len(labels)
                    seen += n
                    loss_sum += float(loss.detach()) * n
                if seen != manifest['split_counts']['train']:
                    raise ValueError('router warmup did not traverse full training split')
                warmup_metrics, _ = evaluate(model, val)
                write(a.out / 'warmup_validation.json', dict(epoch=warmup_epoch,
                    training_samples_seen=seen, router_loss=loss_sum / seen,
                    metrics=warmup_metrics, selection_score=score(warmup_metrics)))
            torch.save(dict(model=model.state_dict(), config=cfg,
                            warmup_epochs=a.router_warmup_epochs),
                       a.out / 'router_warmup_checkpoint.pt')
            ema.load_state_dict(model.state_dict())
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
        start_epoch = 1
        if resume is not None:
            history = json.loads((a.out / 'metrics.json').read_text())
            best, significant_best, last_significant = resume_progress(history, resume['epoch'], score(baseline))
            selected = json.loads((a.out / 'selected_validation.json').read_text())
            if abs(selected['selection_score'] - best) > 1e-10 or selected['epoch'] > resume['epoch']:
                raise ValueError('best selection does not match completed history')
            optimizer.load_state_dict(resume['optimizer'])
            if not (equal_state(model.state_dict(), resume['model']) and
                    equal_state(ema.state_dict(), resume['ema_model']) and
                    equal_state(optimizer.state_dict(), resume['optimizer'])):
                raise ValueError('restored model/EMA/optimizer tensors changed')
            start_epoch = resume['epoch'] + 1
            rng_restored = all(key in resume for key in ('torch_rng', 'cuda_rng', 'loader_rng', 'numpy_rng'))
            if rng_restored:
                torch.set_rng_state(resume['torch_rng'])
                torch.cuda.set_rng_state(resume['cuda_rng'])
                train.generator.set_state(resume['loader_rng'])
                np.random.set_state(resume['numpy_rng'])
            else:
                recovery_seed = a.seed + 100003 * resume['epoch']
                torch.manual_seed(recovery_seed)
                np.random.seed(recovery_seed)
                train.generator.manual_seed(recovery_seed)
            events_file = a.out / 'resume_events.json'
            events = json.loads(events_file.read_text()) if events_file.exists() else []
            events.append(dict(resumed_after_epoch=resume['epoch'],next_epoch=start_epoch,
                               gpu_uuid=gpu_uuid,runtime_git_commit=runtime_commit,
                               last_checkpoint_sha256=digest(a.out / 'last_checkpoint.pt'),
                               optimizer_restored=True,ema_restored=True,
                               raw_model_restored=True,restored_tensors_verified=True,
                               warmup_repeated=False,completed_epochs_repeated=False,
                               rng_restored=rng_restored,bit_exact_replay=False,
                               stochastic_note='legacy last did not save RNG; restart seeded explicitly' if not rng_restored
                               else 'RNG states restored; worker restart/device change is not bit-exact replay',
                               timestamp=time.time(),command=sys.argv))
            write(events_file,events)
        stop_reason = 'maximum candidate budget; no convergence claim'
        for epoch in range(start_epoch, a.epochs + 1):
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
                            optimizer=optimizer.state_dict(), epoch=epoch, config=cfg,
                            runtime_git_commit=runtime_commit,torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(),loader_rng=train.generator.get_state(),
                            numpy_rng=np.random.get_state()),
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
