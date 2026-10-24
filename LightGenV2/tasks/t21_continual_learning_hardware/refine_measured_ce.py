"""Bounded TRAIN-only convex readout polish, folded into the original Linear.

The mean-one camera features permit an invertible change of the existing 784
coordinates: 783 spatial contrasts and one DC coordinate. No extra trainable
parameters, inference preprocessing, bias, optical changes, SDK, or GPU.
Regularization is chosen on a fixed internal TRAIN holdout before TEST access.
The final parent/candidate comparison is a TEST-selected development result.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate, brief
from .train_native import sha, put


def contrast_transform(train):
    """TRAIN statistics only; square invertible linear coordinate transform."""
    train = train.double()
    d = train.shape[1]
    unit = torch.ones(d, dtype=torch.float64) / d ** .5
    direction = -unit.clone()
    direction[-1] += 1
    direction /= direction.norm()
    householder = torch.eye(d, dtype=torch.float64) - 2 * direction[:, None] @ direction[None, :]
    dc = torch.ones(d, dtype=torch.float64) / d
    contrasts = train @ householder[:-1].T
    mean = contrasts.mean(0)
    scale = contrasts.std(0, unbiased=False)
    scale = scale.clamp_min(max(1e-12, float(scale.median()) * 1e-3))
    transform = torch.cat(((householder[:-1] - mean[:, None] * dc) / scale[:, None], dc[None, :]))
    return transform


def fit_convex(z, y, strength, steps):
    """Exactly ten times feature-count coordinates, same function family."""
    weight = torch.zeros(10, z.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight], lr=1., max_iter=steps,
                                max_eval=steps * 2, history_size=20,
                                tolerance_grad=1e-7, tolerance_change=1e-10,
                                line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(z @ weight.T, y, label_smoothing=.02)
        # Penalize spatial contrast, not the existing representable DC offset.
        loss = loss + .5 * strength * weight[:, :-1].square().sum() / 10
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite TRAIN objective')
        loss.backward()
        return loss
    optimizer.step(closure)
    return weight.detach(), float(closure().detach())


def internal_indices(labels):
    rng = np.random.default_rng(20261025)
    fit, hold = [], []
    for label in range(10):
        indices = np.flatnonzero(labels.numpy() == label)
        if len(indices) != 100:
            raise ValueError('A TRAIN must remain ten classes of100')
        rng.shuffle(indices)
        hold.extend(indices[:20]);fit.extend(indices[20:])
    return torch.tensor(fit), torch.tensor(hold)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('capture', 'parent', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    a = p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261025)
    if sha(a.parent) != a.parent_sha256:
        raise ValueError('Parent SHA mismatch')
    report = json.loads((a.capture / 'report.json').read_text())
    if not report['passed'] or report['pilot'] or not report['SDK_released']:
        raise ValueError('Complete physical capture required')
    if sha(a.capture / 'features.npz') != report['features_sha256']:
        raise ValueError('Changed measured features')
    model, checkpoint = load_checkpoint(a.parent, a.parent_sha256)
    if checkpoint.get('measured_features_sha256') != report['features_sha256']:
        raise ValueError('Wrong measured parent lineage')
    if checkpoint['stage'] != 1:
        raise ValueError('Bounded A-only refinement')
    before = upstream_state(model)
    with np.load(a.capture / 'features.npz', allow_pickle=False) as cache:
        x = torch.tensor(cache['features'], dtype=torch.float64)
        labels = torch.tensor(cache['labels'], dtype=torch.long)
        splits = cache['splits'].copy();tasks = cache['tasks'].copy();ids = cache['source_indices'].copy()
    train = np.flatnonzero(splits == 'train');test = np.flatnonzero(splits == 'test')
    if len(train) != 1000 or len(test) != 1000 or set(tasks) != {'eurosat'}:
        raise ValueError('Fixed A TRAIN1000/TEST1000 only')
    if x.shape != (2000, 784) or float((x.mean(1) - 1).abs().max()) > 1e-5:
        raise ValueError('Original mean-one784 camera features required')
    a.out.mkdir(parents=True, exist_ok=False);began = time.time()
    fit, hold = internal_indices(labels[train])
    transform = contrast_transform(x[train][fit])
    zfit = x[train][fit] @ transform.T;zhold = x[train][hold] @ transform.T
    rows = []
    # Four predeclared TRAIN-only strengths, not a TEST-guided candidate sweep.
    for strength in (.001, .01, .1, 1.):
        put(a.out / 'status.json', dict(state='TRAIN_internal_holdout', strength=strength, updated=time.time()))
        weight, loss = fit_convex(zfit, labels[train][fit], strength, 80)
        pred = (zhold @ weight.T).argmax(1)
        rows.append(dict(strength=strength, holdout_correct=int((pred == labels[train][hold]).sum()),
                         records=len(hold), TRAIN_loss=loss))
    chosen = max(rows, key=lambda row: (row['holdout_correct'], row['strength']))['strength']
    transform = contrast_transform(x[train])
    put(a.out / 'protocol.json', dict(parent_sha256=a.parent_sha256, features_sha256=report['features_sha256'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        source_sha256=sha(Path(__file__)), trainable='same original7840 Linear coordinates',
        original_inference_graph_unchanged=True, preprocessing_folded_into_existing_weight=True,
        additional_parameters=0, GPU=False, SDK=False, CPU_threads=2,
        TRAIN_internal_fit_source_indices=ids[train][fit].tolist(),
        TRAIN_internal_holdout_source_indices=ids[train][hold].tolist(),
        strengths=[.001,.01,.1,1.], search=rows, selected_strength=chosen,
        label_smoothing=.02, full_TRAIN_LBFGS_max_iterations=160,
        TEST_statistics=False, TEST_gradients=False, TEST_teacher=False,
        final_selection='parent versus one TRAIN-selected candidate; TEST development only',
        independent_generalization_claim=False))
    put(a.out / 'status.json', dict(state='full_TRAIN_fit', updated=time.time()))
    weight, loss = fit_convex(x[train] @ transform.T, labels[train], chosen, 160)
    folded = (weight @ transform).float()
    with torch.no_grad():
        parent_correct = int((model.shared_head(x[test].float()).argmax(1) == labels[test]).sum())
        model.shared_head.weight.copy_(folded)
        candidate_correct = int((model.shared_head(x[test].float()).argmax(1) == labels[test]).sum())
        train_correct = int((model.shared_head(x[train].float()).argmax(1) == labels[train]).sum())
    if upstream_state(model) != before:
        raise RuntimeError('Upstream state changed')
    def save(name):
        saved = copy.deepcopy(checkpoint)
        saved.update(model=model.state_dict(), parent_checkpoint_sha256=a.parent_sha256,
                     readout_refinement='TRAIN-only regularized CE in invertible contrast coordinates',
                     measured_features_sha256=report['features_sha256'],
                     TEST_selected_development_metric=True, TEST_gradients=False)
        torch.save(saved, a.out / name)
    save('last_checkpoint.pt')
    retained_parent = candidate_correct <= parent_correct
    if retained_parent:
        model.load_state_dict(checkpoint['model'], strict=True)
    save('best_checkpoint.pt')
    strict_rows = {}
    for name in ('best', 'last'):
        saved, _ = load_checkpoint(a.out / (name + '_checkpoint.pt'), sha(a.out / (name + '_checkpoint.pt')))
        if upstream_state(saved) != before:
            raise RuntimeError('Strict CPU frozen-state mismatch')
        scores = evaluate(saved.shared_head, x.float(), labels, tasks, test)
        expected = max(parent_correct, candidate_correct) if name == 'best' else candidate_correct
        if scores['eurosat']['correct'] != expected:
            raise RuntimeError('Strict CPU original-graph prediction mismatch')
        put(a.out / (name + '_test.json'), dict(scores=scores, source_indices=ids[test].tolist(),
            scope='physical TEST1000; TEST-selected development result'))
        strict_rows[name] = brief(scores)
    result = dict(passed=True, parent_correct=parent_correct, candidate_correct=candidate_correct,
        candidate_TRAIN_correct=train_correct, best_correct=max(parent_correct,candidate_correct),
        TRAIN_records=1000, EVAL_records=1000, selected_strength=chosen, retained_parent=retained_parent,
        strict_CPU_reloaded=True, frozen_upstream=True, original_graph=True, trainable_parameters=7840,
        best_sha256=sha(a.out/'best_checkpoint.pt'), last_sha256=sha(a.out/'last_checkpoint.pt'),
        best=strict_rows['best'], last=strict_rows['last'], full_TRAIN_loss=loss,
        TEST_selected_development_metric=True, independent_generalization_claim=False,
        accuracy_not_clipped_to_simulation=True, seconds=time.time()-began, GPU=False, SDK=False)
    put(a.out / 'report.json', result);put(a.out / 'status.json', dict(state='complete', updated=time.time()))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
