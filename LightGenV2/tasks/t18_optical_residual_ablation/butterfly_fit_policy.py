"""Bounded, paired adaptation using train/val only; never accepts test metrics."""
import math


def diagnose(history):
    assert len(history) >= 20
    assert all('test' not in str(row.keys()).lower() for row in history)
    assert all(math.isfinite(row['val']['balanced_nll']) for row in history)
    best = min(history, key=lambda x: x['val']['balanced_nll'])
    last_train = next(x['train'] for x in reversed(history) if 'train' in x)
    previous = sum(x['val']['balanced_nll'] for x in history[-20:-10]) / 10
    recent = sum(x['val']['balanced_nll'] for x in history[-10:]) / 10
    improvement = (previous - recent) / max(previous, 1e-12)
    gap = last_train['accuracy'] - history[-1]['val']['accuracy']
    deterioration = recent / max(best['val']['balanced_nll'], 1e-12) - 1
    age = history[-1]['epoch'] - best['epoch']
    return dict(epochs=len(history), best_epoch=best['epoch'], train_val_accuracy_gap=gap,
        train_balanced_accuracy=last_train['balanced_accuracy'], recent_relative_improvement=improvement,
        recent_relative_deterioration=deterioration, best_age=age,
        still_improving=improvement > .005 and age < 10,
        overfit_signal=gap >= .10 and deterioration >= .05 and age >= 20,
        stalled_low_fit=last_train['balanced_accuracy'] < .85 and gap < .10 and improvement <= .005,
        scope='Heuristic fit signals; low train accuracy is not proof of architecture capacity or optimization failure')


def pair_action(histories):
    assert len(histories) == 2 and len(histories[0]) == len(histories[1])
    diagnostics = [diagnose(h) for h in histories]
    epochs = len(histories[0])
    assert epochs in (100, 150, 200)
    if any(x['still_improving'] for x in diagnostics) and epochs < 200:
        return dict(action='extend_both', target_epochs=epochs + 50, diagnostics=diagnostics)
    if any(x['overfit_signal'] for x in diagnostics):
        return dict(action='one_paired_restart', candidate='regularized', target_epochs=100, diagnostics=diagnostics)
    if all(x['stalled_low_fit'] for x in diagnostics):
        return dict(action='one_paired_restart', candidate='optimization', target_epochs=100, diagnostics=diagnostics)
    return dict(action='lock_pair', diagnostics=diagnostics)


def choose_pair(candidates):
    """Select ONE regimen for both rhos by mean best validation loss; first tie."""
    assert candidates and len(candidates) <= 2
    values = []
    for name, results in candidates:
        assert len(results) == 2
        assert all(result['test_evaluations'] == 0 for result in results)
        value = sum(x['selected_validation']['balanced_nll'] for x in results) / 2
        assert math.isfinite(value)
        values.append(value)
    chosen = min(range(len(values)), key=values.__getitem__)
    return dict(candidate=candidates[chosen][0], pair_mean_best_validation_balanced_nll=dict(zip([c[0] for c in candidates], values)),
                selection_split='val', test_used_for_selection=False)
