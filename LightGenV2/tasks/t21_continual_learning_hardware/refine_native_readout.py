"""Bounded A-only readout refinement; frozen optics, full TRAIN, VAL selection.

Independent of active capture/train entry points. No SDK, TEST teacher, change
to the model graph, or reuse of old CCD under altered optical weights.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .model import build_model
from .train_native import sha, put, tensor_sha, gpu_gate, evaluate, summary
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_lifelong_moe import load_dataset


def upstream_state(model):
    return {k: tensor_sha(v) for k, v in model.state_dict().items()
            if k != 'shared_head.weight'}


def fixed_data(folder, split):
    p = Path(folder) / 'eurosat'
    manifest = json.loads((p / 'manifest.json').read_text())
    if manifest['task'] != 'eurosat' or manifest['split'] != split:
        raise ValueError('Changed fixed dataset split')
    if sha(p / 'inputs.npz') != manifest['inputs_sha256']:
        raise ValueError('Input archive SHA mismatch')
    with np.load(p / 'inputs.npz', allow_pickle=False) as a:
        x = torch.from_numpy(a['amplitudes'].copy()).float()
        y = torch.from_numpy(a['labels'].copy()).long()
    if x.shape != (1000, 224, 224) or y.shape != (1000,):
        raise ValueError('Fixed split must contain exactly 1000 records')
    return dict(x=x, y=y, manifest=manifest)


@torch.no_grad()
def feature_scores(head, x, y):
    logits = head(x)
    prediction = logits.argmax(1)
    recall = [float((prediction[y == k] == k).float().mean())
              for k in range(head.out_features)]
    return dict(accuracy=float((prediction == y).float().mean()),
                balanced_accuracy=float(np.mean(recall)), per_class_recall=recall)


def fit_head(head, x, y, val_x, val_y, cfg, record):
    """Only TRAIN enters backward; VAL is consumed by a no-grad callback."""
    head = head.cpu()
    initial = head.weight.detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    n = len(y)
    record('baseline', 0, None)
    for epoch in range(1, cfg['epochs'] + 1):
        fraction = (epoch - 1) / max(1, cfg['epochs'] - 1)
        rate = cfg['minimum_learning_rate'] + .5 * (
            cfg['learning_rate'] - cfg['minimum_learning_rate']) * (1 + math.cos(math.pi * fraction))
        optimizer.param_groups[0]['lr'] = rate
        permutation = torch.randperm(n)
        total = 0.
        head.train()
        for start in range(0, n, cfg['batch_size']):
            ix = permutation[start:start + cfg['batch_size']]
            batch = x[ix]
            batch = F.dropout(batch, p=cfg['TRAIN_feature_dropout'], training=True)
            batch = batch * (1 + torch.randn_like(batch) * cfg['TRAIN_feature_noise_std'])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(batch), y[ix], label_smoothing=cfg['label_smoothing'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite TRAIN readout loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        record('adamw', epoch, total / n)
    # Deterministic convex full-batch polish, not another model or TEST search.
    optimizer = torch.optim.LBFGS(head.parameters(), lr=.5, max_iter=1,
                                  history_size=20, line_search_fn='strong_wolfe')
    for step in range(1, cfg['LBFGS_steps'] + 1):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x), y, label_smoothing=cfg['label_smoothing'])
            loss = loss + cfg['LBFGS_anchor_strength'] * (head.weight - initial).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite full-batch readout loss')
            loss.backward()
            return loss
        loss = float(optimizer.step(closure()).detach())
        record('lbfgs', step, loss)
    return head


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config', 'checkpoint', 'train_protocols', 'val', 'test', 'out'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cfg = json.loads(args.config.read_text())
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    try:
        gpu = gpu_gate(cfg, args.gpu_uuid)
        device = torch.device('cuda:0')
        if sha(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError('Parent A checkpoint SHA mismatch')
        parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if parent['architecture'] != 'moe' or parent['stage'] != 1:
            raise ValueError('This bounded refinement only accepts LightGen A')
        model = build_model('moe')
        model.load_state_dict(parent['model'], strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name == 'shared_head.weight')
        before = upstream_state(model)
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != 7840:
            raise ValueError('Changed original readout parameter count')
        paths = {k: Path(v) for k, v in json.loads(args.train_protocols.read_text()).items()}
        identity = json.loads((Path(__file__).parent / 'source_identity.json').read_text())
        expected = identity['models'][3]['config']['protocol_sha256']['eurosat']
        if sha(paths['eurosat']) != expected:
            raise ValueError('Full original TRAIN protocol changed')
        val = fixed_data(args.val, 'val')
        if sha(paths['vision']) != val['manifest']['vision_checkpoint_sha256']:
            raise ValueError('Frozen frontend SHA changed')
        put(args.out / 'protocol.json', dict(config=cfg, parent_sha256=args.checkpoint_sha256,
            source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            command=__import__('sys').argv, gpu=gpu, TRAIN_protocol_sha256=expected,
            VAL_manifest=val['manifest'], TEST_gradients=False, TEST_selection=False,
            CCD_compatibility='all upstream state bitwise frozen; terminal head only',
            chain_scope='A-only candidate, not an independently spliced A-D curve'))
        model.to(device).eval()
        train = load_dataset(paths, 'eurosat', 'train', vision_checkpoint=paths['vision'], device=device)
        if len(train) != 15998:
            raise ValueError('Full original TRAIN count changed')
        labels = torch.as_tensor(np.asarray(train.labels), dtype=torch.long)
        features = {}
        for split, data, count in (('train', train, len(train)), ('val', val, 1000)):
            rows = []
            with torch.no_grad():
                for start in range(0, count, cfg['cache_batch_size']):
                    stop = min(count, start + cfg['cache_batch_size'])
                    amplitude = (data.get_batch(np.arange(start, stop), device) if split == 'train'
                                 else data['x'][start:stop].to(device))
                    result = model(amplitude)
                    feature = result['ccd_features']
                    error = float((model.shared_head(feature) - result['logits']).abs().max())
                    if error > 1e-6 or not torch.isfinite(feature).all():
                        raise RuntimeError('Frozen feature replay mismatch')
                    rows.append(feature.cpu())
                    if start % 2000 == 0:
                        put(args.out / 'status.json', dict(state='caching', split=split,
                            completed=stop, total=count, updated=time.time()))
            features[split] = torch.cat(rows)
        np.savez(args.out / 'TRAIN_VAL_features.npz', TRAIN=features['train'].numpy(),
                 TRAIN_labels=labels.numpy(), VAL=features['val'].numpy(), VAL_labels=val['y'].numpy())
        del train
        model.cpu()
        torch.cuda.empty_cache()
        best = -1.
        best_tag = None
        history = []
        def save(name, method, step):
            saved = copy.deepcopy(parent)
            saved.update(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         parent_checkpoint_sha256=args.checkpoint_sha256,
                         selection='VAL only; frozen optics terminal refinement',
                         refinement=dict(method=method, step=step, frozen_upstream_SHA256=before))
            torch.save(saved, args.out / name)
        def record(method, step, loss):
            nonlocal best, best_tag
            score = feature_scores(model.shared_head, features['val'], val['y'])
            value = score['balanced_accuracy']
            if value > best + 1e-10:
                best, best_tag = value, dict(method=method, step=step)
                save('best_checkpoint.pt', method, step)
            save('last_checkpoint.pt', method, step)
            row = dict(method=method, step=step, TRAIN_loss=loss, VAL=score,
                       best_VAL=best, best_tag=best_tag, updated=time.time())
            history.append(row)
            put(args.out / 'history.json', history)
            put(args.out / 'status.json', dict(state='readout_training', **row))
            if step % 10 == 0:
                print(json.dumps(row), flush=True)
        fit_head(model.shared_head, features['train'], labels, features['val'], val['y'], cfg, record)
        if upstream_state(model) != before:
            raise RuntimeError('Upstream state changed')
        # First access to TEST occurs only after all choices are fixed on VAL.
        selected = torch.load(args.out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(selected['model'], strict=True)
        if upstream_state(model) != before:
            raise RuntimeError('Saved selected upstream mismatch')
        test = fixed_data(args.test, 'test')
        scores = evaluate(model, {'eurosat': test}, ['eurosat'], torch.device('cpu'), 4)
        reloaded = build_model('moe')
        reloaded.load_state_dict(selected['model'], strict=True)
        with torch.no_grad():
            a = model(test['x'][:4])['logits']
            b = reloaded(test['x'][:4])['logits']
        if not torch.equal(a, b):
            raise RuntimeError('Strict saved CPU replay failed')
        put(args.out / 'selected_test.json', dict(scores=scores, strict_CPU=True,
            TEST_gradients=False, TEST_selection=False))
        put(args.out / 'frozen_audit.json', dict(passed=True, parent_upstream_SHA256=before,
            selected_upstream_SHA256=upstream_state(model), readout_parameters=7840))
        report = dict(passed=True, best_tag=best_tag, best_is_parent=(best_tag['method'] == 'baseline'),
            VAL=best, TEST=summary(scores), best_sha256=sha(args.out / 'best_checkpoint.pt'),
            last_sha256=sha(args.out / 'last_checkpoint.pt'), frozen_upstream=True,
            strict_CPU=True, TEST_selection=False, seconds=time.time() - started)
        put(args.out / 'report.json', report)
        put(args.out / 'status.json', dict(state='complete', **report))
        print(json.dumps(report), flush=True)
    except BaseException:
        put(args.out / 'failure.json', dict(traceback=traceback.format_exc(), updated=time.time()))
        raise


if __name__ == '__main__':
    main()
