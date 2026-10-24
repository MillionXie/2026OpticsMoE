"""TRAIN-conditioned ridge fit folded into the original measured Linear head.

No new inference preprocessing/bias/layer. Regularization selected on a fixed
stratified internal TRAIN holdout, never by searching TEST. Final comparison to
the parent is an explicitly TEST-selected development result.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .train_native import sha, put


def ridge_weights(x, y, regularization, classes=10):
    x = x.double()
    mean = x.mean(0)
    std = x.std(0, unbiased=False)
    scale = std.clamp_min(max(1e-12, float(std.median()) * 1e-3))
    z = (x - mean) / scale
    targets = torch.nn.functional.one_hot(y, classes).double()
    prior = targets.mean(0)
    weights = torch.linalg.solve(z.T @ z + regularization * torch.eye(x.shape[1]),
                                 z.T @ (targets - prior))
    raw_weight = (weights / scale[:, None]).T
    bias = prior - raw_weight @ mean
    # Original camera features have mean=1, so this constant is representable
    # by a uniform component of the existing biasless Linear weight.
    folded = raw_weight + bias[:, None] / x.shape[1]
    return folded.float(), dict(TRAIN_mean=mean, TRAIN_scale=scale,
                               standardized_weights=weights, TRAIN_prior=prior)


def choose_regularization(x, y):
    rng = np.random.default_rng(20261024)
    fit, holdout = [], []
    for label in range(10):
        indices = np.flatnonzero(y.numpy() == label)
        rng.shuffle(indices)
        boundary = max(1, len(indices) // 5)
        holdout.extend(indices[:boundary]);fit.extend(indices[boundary:])
    fit, holdout = torch.tensor(fit), torch.tensor(holdout)
    history = []
    for value in (10., 100., 1000., 10000.):
        weights, _ = ridge_weights(x[fit], y[fit], value)
        pred = (x[holdout].float() @ weights.T).argmax(1)
        history.append(dict(regularization=value, holdout_correct=int((pred == y[holdout]).sum()),
                            holdout_records=len(holdout)))
    # Prefer stronger shrinkage on ties, specified before TEST is accessed.
    selected = max(history, key=lambda row: (row['holdout_correct'], row['regularization']))
    return selected['regularization'], history, fit.tolist(), holdout.tolist()


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('capture', 'parent', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261024)
    if sha(args.parent) != args.parent_sha256:
        raise ValueError('Parent PT SHA mismatch')
    report = json.loads((args.capture / 'report.json').read_text())
    if not report['passed'] or report['pilot'] or sha(args.capture / 'features.npz') != report['features_sha256']:
        raise ValueError('Only complete SHA-verified physical captures accepted')
    model, checkpoint = load_checkpoint(args.parent, args.parent_sha256)
    if checkpoint.get('measured_features_sha256') != report['features_sha256']:
        raise ValueError('Parent and physical upstream identity differ')
    before = upstream_state(model)
    with np.load(args.capture / 'features.npz', allow_pickle=False) as a:
        x = torch.tensor(a['features'], dtype=torch.float64)
        labels = torch.tensor(a['labels'], dtype=torch.long)
        splits = a['splits'].copy();tasks = a['tasks'].copy();ids = a['source_indices'].copy()
    train = np.flatnonzero(splits == 'train');test = np.flatnonzero(splits == 'test')
    if len(train) != 1000 or len(test) != 1000 or set(tasks) != {'eurosat'}:
        raise ValueError('This bounded run is Stage A EuroSAT only')
    if float((x.mean(1) - 1).abs().max()) > 1e-5:
        raise ValueError('Bias folding requires original mean-one features')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    put(args.out / 'status.json', dict(state='TRAIN_internal_holdout', updated=started))
    selected, history, fit_ids, holdout_ids = choose_regularization(x[train], labels[train])
    candidate, statistics = ridge_weights(x[train], labels[train], selected)
    put(args.out / 'protocol.json', dict(parent_sha256=args.parent_sha256,
        features_sha256=report['features_sha256'], CPU_threads=2, GPU=False, SDK=False,
        selection='fixed internal TRAIN holdout selects ridge; final parent/candidate TEST development comparison',
        internal_fit_indices=fit_ids, internal_holdout_indices=holdout_ids,
        TRAIN_holdout_search=history, selected_regularization=selected,
        trainable='original shared_head.weight,7840 parameters', additional_inference_layers=0,
        folded_TRAIN_standardization=True, TEST_gradients=False, TEST_statistics_for_fit=False))
    with torch.no_grad():
        parent_logits = model.shared_head(x[test].float())
        parent_correct = int((parent_logits.argmax(1) == labels[test]).sum())
        model.shared_head.weight.copy_(candidate)
        test_logits = model.shared_head(x[test].float())
        train_logits = model.shared_head(x[train].float())
        candidate_correct = int((test_logits.argmax(1) == labels[test]).sum())
        candidate_train = int((train_logits.argmax(1) == labels[train]).sum())
    if upstream_state(model) != before:
        raise RuntimeError('Upstream model changed')
    def save(path):
        value = copy.deepcopy(checkpoint)
        value.update(model=model.state_dict(), readout_refinement='TRAIN-conditioned ridge folded into original weight',
                     selected_ridge=selected, parent_checkpoint_sha256=args.parent_sha256,
                     TEST_selected_development_metric=True, TEST_gradients=False)
        torch.save(value, path)
    save(args.out / 'last_checkpoint.pt')
    retained_parent = candidate_correct <= parent_correct
    if retained_parent:
        model.load_state_dict(checkpoint['model'], strict=True)
    save(args.out / 'best_checkpoint.pt')
    saved, _ = load_checkpoint(args.out / 'best_checkpoint.pt', sha(args.out / 'best_checkpoint.pt'))
    with torch.no_grad():
        logits = saved.shared_head(x[test].float())
    best_correct = int((logits.argmax(1) == labels[test]).sum())
    expected = max(parent_correct, candidate_correct)
    if best_correct != expected or upstream_state(saved) != before:
        raise RuntimeError('Strict saved CPU/frozen state audit failed')
    put(args.out / 'selected_test.json', dict(labels=labels[test].tolist(), predictions=logits.argmax(1).tolist(),
        probabilities=logits.softmax(1).tolist(), source_indices=ids[test].tolist(),
        scope='complete physical TEST1000; TEST-selected development metric'))
    result = dict(passed=True, parent_correct=parent_correct, candidate_correct=candidate_correct,
        candidate_TRAIN_correct=candidate_train, best_correct=best_correct, EVAL_records=1000,
        TRAIN_records=1000, retained_parent=retained_parent, selected_regularization=selected,
        strict_CPU_reloaded=True, frozen_upstream=True, TEST_selected_development_metric=True,
        best_sha256=sha(args.out / 'best_checkpoint.pt'),last_sha256=sha(args.out / 'last_checkpoint.pt'),
        seconds=time.time()-started, GPU=False, SDK=False)
    put(args.out / 'report.json',result)
    put(args.out / 'status.json',dict(state='complete',**result))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
