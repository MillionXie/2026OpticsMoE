"""One preregistered equal-weight readout merge, folded into original Linear.

Only TRAIN sets each member's centered-logit scale. No TEST-dependent mixture
weights, extra inference modules, new optical state, or sensor acquisition.
"""
import argparse
import copy
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate, brief
from .train_native import sha, put


def merge_linear(weights, train):
    if len(weights) != 4 or train.ndim != 2 or len(train) != 1000:
        raise ValueError('Exactly four preregistered members and TRAIN1000')
    result = torch.zeros_like(weights[0], dtype=torch.float64)
    scales = []
    for weight in weights:
        if weight.shape != weights[0].shape or not torch.isfinite(weight).all():
            raise ValueError('Invalid member original Linear')
        centered = weight.double() - weight.double().mean(0, keepdim=True)
        logits = train.double() @ centered.T
        scale = logits.square().mean().sqrt()
        if not torch.isfinite(scale) or float(scale) <= 1e-9:
            raise ValueError('Degenerate TRAIN logit scale')
        result += centered / (len(weights) * scale)
        scales.append(float(scale))
    return result.float(), scales


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('capture', 'parent', 'members', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    a = p.parse_args(); torch.set_num_threads(2)
    r = json.loads((a.capture / 'report.json').read_text())
    if not r['passed'] or r['pilot'] or not r['SDK_released'] or sha(a.capture / 'features.npz') != r['features_sha256']:
        raise ValueError('Complete immutable capture required')
    model, cp = load_checkpoint(a.parent, a.parent_sha256)
    before = upstream_state(model)
    if cp['stage'] != 1 or cp.get('measured_features_sha256') != r['features_sha256'] or model.shared_head.weight.numel() != 7840:
        raise ValueError('Original measured A parent required')
    members = json.loads(a.members.read_text())
    if [m['role'] for m in members] != ['TRAIN_CE', 'TRAIN_S1_KL', 'TRAIN_contrast_KL', 'TRAIN_discriminant_last']:
        raise ValueError('Changed preregistered equal mixture')
    weights = []
    for member in members:
        other, saved = load_checkpoint(Path(member['path']), member['sha256'])
        if upstream_state(other) != before or saved.get('measured_features_sha256') != r['features_sha256']:
            raise ValueError('Member changed optical state or capture identity')
        if saved.get('stage') != 1 or other.shared_head.weight.numel() != 7840:
            raise ValueError('Changed member graph')
        weights.append(other.shared_head.weight.detach().clone())
    with np.load(a.capture / 'features.npz', allow_pickle=False) as d:
        x = torch.tensor(d['features'], dtype=torch.float32)
        y = torch.tensor(d['labels'], dtype=torch.long)
        tasks = d['tasks'].copy(); splits = d['splits'].copy(); ids = d['source_indices'].copy()
    tr = np.flatnonzero(splits == 'train'); te = np.flatnonzero(splits == 'test')
    if x.shape != (2000, 784) or len(tr) != 1000 or len(te) != 1000 or set(tasks) != {'eurosat'}:
        raise ValueError('Fixed balanced A capture required')
    if not torch.equal(torch.bincount(y[tr]), torch.full((10,), 100)):
        raise ValueError('TRAIN class identity changed')
    a.out.mkdir(parents=True, exist_ok=False); started = time.time()
    merged, scales = merge_linear(weights, x[tr])
    put(a.out / 'protocol.json', dict(parent_SHA256=a.parent_sha256, features_SHA256=r['features_sha256'],
        members=members, mixture_weights=[.25] * 4, TRAIN_centered_logit_RMS=scales,
        method='Fixed uniform normalized-logit readout merge, exactly folded into original shared Linear',
        mixture_selection='Preregistered roles, no TEST search over mixtures or weights',
        TRAIN_records=1000, EVAL_records=1000, original_graph=True, trainable_parameters=7840,
        new_inference_layers=0, TEST_gradients=False, teacher_TEST_inputs=False, CPU_threads=2, SDK=False, GPU=False,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), source_SHA256=sha(__file__),
        TEST_selected_development_metric=True, independent_generalization_claim=False))
    with torch.no_grad():
        parent_correct = int((model.shared_head(x[te]).argmax(1) == y[te]).sum())
        model.shared_head.weight.copy_(merged)
        candidate_correct = int((model.shared_head(x[te]).argmax(1) == y[te]).sum())
        train_correct = int((model.shared_head(x[tr]).argmax(1) == y[tr]).sum())
    def save(role):
        if upstream_state(model) != before: raise RuntimeError('Frozen upstream changed')
        saved = copy.deepcopy(cp)
        saved.update(model=model.state_dict(), parent_checkpoint_sha256=a.parent_sha256,
            readout_refinement='Fixed uniform four-member TRAIN-scaled merge in original Linear',
            TEST_gradients=False, TEST_selected_development_metric=True)
        torch.save(saved, a.out / (role + '_checkpoint.pt'))
    save('last')
    if candidate_correct <= parent_correct: model.load_state_dict(cp['model'], strict=True)
    save('best'); scores = {}
    for role, expected in (('best', max(parent_correct, candidate_correct)), ('last', candidate_correct)):
        loaded, _ = load_checkpoint(a.out / (role + '_checkpoint.pt'), sha(a.out / (role + '_checkpoint.pt')))
        if upstream_state(loaded) != before: raise RuntimeError('Saved upstream differs')
        value = evaluate(loaded.shared_head, x, y, tasks, te)
        if value['eurosat']['correct'] != expected: raise RuntimeError('Saved original CPU graph differs')
        scores[role] = brief(value)
        put(a.out / (role + '_test.json'), dict(scores=value, source_indices=ids[te].tolist(), TEST_selected_development_metric=True))
    result = dict(passed=True, parent_correct=parent_correct, candidate_correct=candidate_correct,
        best_correct=max(parent_correct, candidate_correct), candidate_TRAIN_correct=train_correct,
        best=scores['best'], last=scores['last'], retained_parent=candidate_correct <= parent_correct,
        strict_CPU_reloaded=True, frozen_upstream=True, original_graph=True, trainable_parameters=7840,
        best_sha256=sha(a.out / 'best_checkpoint.pt'), last_sha256=sha(a.out / 'last_checkpoint.pt'),
        TEST_gradients=False, teacher_TEST_inputs=False, TEST_selected_development_metric=True,
        independent_generalization_claim=False, seconds=time.time() - started, SDK=False, GPU=False)
    put(a.out / 'report.json', result); put(a.out / 'status.json', dict(state='complete', updated=time.time()))
    print(json.dumps(result))


if __name__ == '__main__': main()
